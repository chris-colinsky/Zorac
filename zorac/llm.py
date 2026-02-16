"""
LLM interaction and conversation summarization.

This module handles the core challenge of working with LLMs that have limited
context windows: automatic conversation summarization.

The problem:
  LLMs can only process a fixed number of tokens at once (the "context window").
  As a conversation grows longer, the total tokens eventually exceed what the
  model can handle. Without management, the conversation would simply fail.

The solution: Sliding window with summarization
  When the token count exceeds MAX_INPUT_TOKENS, we:
  1. Keep the system message (always first, always preserved)
  2. Send older messages to the LLM for summarization into a concise paragraph
  3. Keep the N most recent messages (KEEP_RECENT_MESSAGES) intact
  4. Replace the old messages with the summary

  This preserves important context (key facts, decisions) while dramatically
  reducing token usage. The result is a message list structured like:
    [system_message, summary_message, recent_msg_1, recent_msg_2, ...]

Why summarize instead of just truncating?
  Simply dropping old messages loses context that might be important later
  (e.g., "earlier you said you prefer Python 3.13"). Summarization extracts
  the key information and compresses it, maintaining conversational continuity.

Trade-offs:
  - Summarization requires an extra LLM call, adding latency
  - The summary is lossy — details are inevitably lost
  - Summary quality depends on the model's summarization ability
  - If summarization fails (API error), we fall back to keeping only recent
    messages (losing context is better than crashing)
"""

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import KEEP_RECENT_MESSAGES, VLLM_MODEL
from .console import console


async def summarize_old_messages(
    client: AsyncOpenAI, messages: list[ChatCompletionMessageParam], auto=True
) -> list[ChatCompletionMessageParam]:
    """Summarize old messages to reduce token count while preserving context.

    This function implements the sliding window summarization strategy described
    in the module docstring. It can be triggered automatically (when token limit
    is approached) or manually (via the /summarize command).

    Message list structure before summarization:
      [system, msg1, msg2, msg3, ..., msgN-5, msgN-4, msgN-3, msgN-2, msgN-1, msgN]
       ^                                      ^--- KEEP_RECENT_MESSAGES (6) ---^
       |--- kept ---|--- old (summarized) ---|--- kept (recent) --------------|

    Message list structure after summarization:
      [system, summary, msgN-5, msgN-4, msgN-3, msgN-2, msgN-1, msgN]

    Args:
        client: The AsyncOpenAI client for making the summarization API call.
        messages: The full conversation history to potentially summarize.
        auto: If True, shows "Token limit approaching" warning. If False
              (manual /summarize), shows "Summarizing as requested" message.
              This distinction helps the user understand why summarization
              is happening.

    Returns:
        A new message list — either summarized (if there were enough messages
        to summarize) or unchanged (if the conversation is too short).
    """
    # Guard clause: don't summarize if there aren't enough messages.
    # We need at least KEEP_RECENT_MESSAGES + 1 (system message) + 1 (at least
    # one old message to summarize). Without this check, we'd try to summarize
    # an empty list of "old" messages.
    if len(messages) <= KEEP_RECENT_MESSAGES + 1:  # +1 for system message
        return messages

    # Split the message list into three segments:
    # 1. System message — always preserved as-is (contains instructions for the LLM)
    # 2. Old messages — everything between system message and recent messages (to be summarized)
    # 3. Recent messages — the last N messages (preserved for conversational continuity)
    system_message = messages[0]
    old_messages = messages[1:-KEEP_RECENT_MESSAGES]
    recent_messages = messages[-KEEP_RECENT_MESSAGES:]

    # Display appropriate status message based on whether this was triggered
    # automatically (token limit) or manually (/summarize command).
    if auto:
        console.print(
            "\n[yellow]⚠ Token limit approaching. Summarizing conversation history...[/yellow]"
        )
    else:
        console.print("\n[green]ℹ Summarizing conversation history as requested...[/green]")

    # Show a spinner while summarizing — this involves an LLM API call that
    # can take several seconds, so visual feedback is important.
    with console.status("[bold green]Summarizing...[/bold green]", spinner="dots"):
        # Format old messages into a readable text block for the summarization prompt.
        # We include the role (USER/ASSISTANT) so the summarizer understands
        # the conversation flow and who said what.
        conversation_text = "\n\n".join(
            [f"{msg['role'].upper()}: {msg.get('content', '')}" for msg in old_messages]
        )

        # Make a separate API call to summarize the conversation.
        # We use a dedicated system prompt ("You are a helpful assistant that creates
        # concise summaries") rather than the main system prompt, because we want
        # the model to focus purely on summarization, not on being "Zorac".
        # Low temperature (0.1) ensures consistent, factual summaries.
        # stream=False because we need the complete summary before proceeding.
        try:
            summary_response = await client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates concise summaries.",
                    },
                    {
                        "role": "user",
                        "content": f"Please create a concise summary of this conversation history, preserving key facts, decisions, and context:\n\n{conversation_text}",
                    },
                ],
                temperature=0.1,
                stream=False,
            )

            summary = summary_response.choices[0].message.content

            # Construct the new message list with the summary replacing old messages.
            # The summary is stored as a system message with a "Previous conversation
            # summary:" prefix so it can be identified later (e.g., by the /summary
            # command) and distinguished from the main system message.
            summary_message: ChatCompletionMessageParam = {
                "role": "system",
                "content": f"Previous conversation summary: {summary}",
            }
            new_messages: list[ChatCompletionMessageParam] = [
                system_message,
                summary_message,
            ] + recent_messages

            console.print(
                f"[green]✓ Summarized {len(old_messages)} messages. Kept {KEEP_RECENT_MESSAGES} recent messages.[/green]\n"
            )
            return new_messages

        except Exception as e:
            # If summarization fails (network error, model error, etc.), we fall
            # back to simply keeping the system message + recent messages.
            # This loses the old message context but keeps the app functional.
            # This is better than crashing or keeping messages that exceed the
            # token limit (which would cause the next chat call to fail).
            console.print(f"[red]Error during summarization: {e}[/red]")
            return [system_message] + recent_messages
