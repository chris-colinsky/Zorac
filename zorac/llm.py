from openai.types.chat import ChatCompletionMessageParam

from .config import KEEP_RECENT_MESSAGES
from .console import console


def summarize_old_messages(
    client, messages: list[ChatCompletionMessageParam], *, model: str, auto: bool = True
) -> list[ChatCompletionMessageParam]:
    """Summarize old messages to reduce token count while preserving context"""
    if len(messages) <= KEEP_RECENT_MESSAGES + 1:  # +1 for system message
        return messages

    # Separate system message, old messages, and recent messages
    system_message = messages[0]
    old_messages = messages[1:-KEEP_RECENT_MESSAGES]
    recent_messages = messages[-KEEP_RECENT_MESSAGES:]

    # Create a summary of old messages
    if auto:
        console.print(
            "\n[yellow]⚠ Token limit approaching. Summarizing conversation history...[/yellow]"
        )
    else:
        console.print("\n[green]ℹ Summarizing conversation history as requested...[/green]")

    with console.status("[bold green]Summarizing...[/bold green]", spinner="dots"):
        # Build conversation text for summarization
        conversation_text = "\n\n".join(
            [f"{msg['role'].upper()}: {msg.get('content', '')}" for msg in old_messages]
        )

        # Request summarization
        try:
            summary_response = client.chat.completions.create(
                model=model,
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

            # Create new messages list with summary
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
            console.print(f"[red]Error during summarization: {e}[/red]")
            # If summarization fails, just keep recent messages
            return [system_message] + recent_messages
