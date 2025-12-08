import atexit
import json
import os
import readline
import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

# Load environment variables from .env file
load_dotenv()

# Configuration (can be overridden via environment variables)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").strip().rstrip("/")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = os.getenv("VLLM_MODEL", "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ")
SESSION_FILE = Path(os.getenv("ZORAC_SESSION_FILE", str(Path.home() / ".zorac_session.json")))
HISTORY_FILE = Path(os.getenv("ZORAC_HISTORY_FILE", str(Path.home() / ".zorac_history")))

# Token limits
MAX_INPUT_TOKENS = 12000  # System + Chat history
MAX_OUTPUT_TOKENS = 4000  # Response generation
KEEP_RECENT_MESSAGES = 6  # Keep this many recent messages when summarizing

# Initialize Rich Console
console = Console()


def count_tokens(messages: list[ChatCompletionMessageParam], model="gpt-4"):
    """Count tokens in messages using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for _, value in message.items():
            if isinstance(value, str):
                num_tokens += len(encoding.encode(value))
            elif isinstance(value, list):
                # Handle list of content parts if necessary, for now just stringify
                for part in value:
                    if isinstance(part, dict) and "text" in part:
                        num_tokens += len(encoding.encode(str(part["text"])))
    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens


def save_session(messages: list[ChatCompletionMessageParam], filepath=SESSION_FILE):
    """Save conversation history to a JSON file"""
    try:
        with open(filepath, "w") as f:
            json.dump(messages, f, indent=2)
        return True
    except Exception as e:
        console.print(f"[red]Error saving session: {e}[/red]")
        return False


def load_session(filepath=SESSION_FILE):
    """Load conversation history from a JSON file"""
    try:
        if filepath.exists():
            with open(filepath) as f:
                messages = json.load(f)
            return messages
        return None
    except Exception as e:
        console.print(f"[red]Error loading session: {e}[/red]")
        return None


def print_header():
    """Print welcome header with ASCII art logo"""
    # ASCII art logo for Zorac (purple color matching terminal prompt)
    logo = """[bold purple]
     ███████╗ ██████╗ ██████╗  █████╗  ██████╗`
     ╚══███╔╝██╔═══██╗██╔══██╗██╔══██╗██╔════╝
       ███╔╝ ██║   ██║██████╔╝███████║██║
      ███╔╝  ██║   ██║██╔══██╗██╔══██║██║
     ███████╗╚██████╔╝██║  ██║██║  ██║╚██████╗
     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
[/bold purple]
        [dim italic]intelligence running on localhost[/dim italic]
"""

    header_text = f"""[dim]Connected to: {VLLM_BASE_URL}[/dim]
[dim]Model: {VLLM_MODEL}[/dim]

[bold]Commands:[/bold]
  [green]/clear[/green]     - Clear conversation history
  [green]/save[/green]      - Manually save session
  [green]/load[/green]      - Reload session from disk
  [green]/tokens[/green]    - Show current token usage
  [green]/summarize[/green] - Force summarization of conversation
  [green]/summary[/green]   - Show current conversation summary
  [green]/quit[/green]      - Exit the chat"""

    console.print(logo)
    console.print(Panel(header_text, box=box.ROUNDED, expand=False))


def summarize_old_messages(
    client, messages: list[ChatCompletionMessageParam], auto=True
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


def check_connection(client) -> bool:
    """Verify connection to the vLLM server"""
    try:
        with console.status("[bold green]Verifying connection...[/bold green]"):
            client.models.list()
        console.print("[green]✓ Connection verified[/green]")
        return True
    except Exception as e:
        console.print("[bold red]✗ Connection failed![/bold red]")
        console.print(f"[red]Error: {e}[/red]")
        console.print(f"[yellow]Target URL: {client.base_url}[/yellow]")
        console.print(
            "[yellow]Review your .env file and ensure the VLLM_BASE_URL is correct and the server is running.[/yellow]"
        )
        return False


def setup_readline():
    """Setup command history and persistent storage"""
    try:
        if HISTORY_FILE.exists():
            readline.read_history_file(str(HISTORY_FILE))

        # Save history on exit
        atexit.register(readline.write_history_file, str(HISTORY_FILE))

        # Optional: Enable tab completion if desired (simple default)
        readline.parse_and_bind("tab: complete")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not setup command history: {e}[/yellow]")


def main():
    # Print welcome message first
    print_header()

    # Setup readline history
    setup_readline()

    # Initialize client pointing to vLLM server
    client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    # Verify connection on startup
    if not check_connection(client):
        console.print(
            "\n[bold yellow]Proceeding offline (some features may not work)...[/bold yellow]"
        )

    # Try to load existing session
    loaded_messages = load_session()
    if loaded_messages:
        messages: list[ChatCompletionMessageParam] = loaded_messages
        token_count = count_tokens(messages)
        console.print(
            f"[green]✓ Loaded previous session ({len(messages)} messages, ~{token_count} tokens)[/green]"
        )
    else:
        # Initialize conversation memory with system message
        messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # Interactive loop
    while True:
        try:
            # Get user input
            try:
                # Use ANSI codes for prompt so readline handles it correctly (can't be deleted, redraws correctly)
                # \001 and \002 mark non-printing characters for readline
                # Note: We print \n separately to prevent readline refresh issues on macOS
                print()
                prompt = "\001\033[1;34m\002You:\001\033[0m\002 "
                user_input = input(prompt).strip()
            except EOFError:
                break

            # Handle empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["/quit", "/exit"]:
                # Save session before exiting
                save_session(messages)
                console.print("\n[green]✓ Session saved. Goodbye![/green]\n")
                break

            if user_input.lower() == "/clear":
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                save_session(messages)
                console.print("\n[green]✓ Conversation history cleared and saved![/green]\n")
                continue

            if user_input.lower() == "/save":
                if save_session(messages):
                    console.print(f"\n[green]✓ Session saved to {SESSION_FILE}[/green]\n")
                continue

            if user_input.lower() == "/load":
                loaded = load_session()
                if loaded:
                    messages = loaded
                    token_count = count_tokens(messages)
                    console.print(
                        f"\n[green]✓ Session reloaded ({len(messages)} messages, ~{token_count} tokens)[/green]\n"
                    )
                else:
                    console.print("\n[red]✗ No saved session found[/red]\n")
                continue

            if user_input.lower() == "/tokens":
                token_count = count_tokens(messages)
                console.print("\n[bold]📊 Token usage:[/bold]")
                console.print(f"   Current: ~{token_count} tokens")
                console.print(f"   Limit: {MAX_INPUT_TOKENS} tokens")
                console.print(f"   Remaining: ~{MAX_INPUT_TOKENS - token_count} tokens")
                console.print(f"   Messages: {len(messages)}\n")
                continue

            if user_input.lower() == "/summarize":
                if len(messages) <= KEEP_RECENT_MESSAGES + 1:
                    console.print(
                        f"\n[yellow]⚠ Not enough messages to summarize. Need more than {KEEP_RECENT_MESSAGES + 1} messages.[/yellow]\n"
                    )
                else:
                    messages = summarize_old_messages(client, messages, auto=False)
                    save_session(messages)
                    console.print("[green]✓ Session saved with summary[/green]\n")
                continue

            if user_input.lower() == "/summary":
                # Look for existing summary (should be at index 1 if it exists)
                summary_found = False
                if len(messages) > 1 and messages[1].get("role") == "system":
                    content = messages[1].get("content", "")
                    if isinstance(content, str) and content.startswith(
                        "Previous conversation summary:"
                    ):
                        summary_text = content.replace("Previous conversation summary:", "").strip()
                        console.print("\n[bold]📝 Current Conversation Summary:[/bold]\n")
                        console.print(Panel(Markdown(summary_text), box=box.ROUNDED, expand=False))
                        console.print()
                        summary_found = True

                if not summary_found:
                    console.print(
                        "\n[yellow]ℹ No summary exists yet. Use /summarize to create one.[/yellow]\n"
                    )
                continue

            # Add user message to conversation history
            messages.append({"role": "user", "content": user_input})

            # Check token count and summarize if needed
            current_tokens = count_tokens(messages)
            if current_tokens > MAX_INPUT_TOKENS:
                messages = summarize_old_messages(client, messages)

            # Start the timer
            start_time = time.time()

            # Prepare streaming loop
            console.print("\n[bold purple]Assistant:[/bold purple]")

            full_content = ""
            first_chunk_received = False

            # Show loading animation while waiting for response
            with console.status(
                "[bold purple]Thinking...[/bold purple]", spinner="dots", spinner_style="purple"
            ) as status:
                try:
                    stream = client.chat.completions.create(
                        model=VLLM_MODEL,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=MAX_OUTPUT_TOKENS,
                        stream=True,
                    )

                    # Process the stream
                    with Live("", refresh_per_second=10, vertical_overflow="visible") as live:
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                # Stop the loading animation on first chunk
                                if not first_chunk_received:
                                    status.stop()
                                    first_chunk_received = True

                                content_chunk = chunk.choices[0].delta.content
                                full_content += content_chunk
                                live.update(Markdown(full_content))

                except Exception as e:
                    console.print(f"[red]Error receiving response: {e}[/red]")
                    full_content += f"\n[Error: {e}]"

            # Stop the timer
            end_time = time.time()
            duration = end_time - start_time

            # Extract collected tokens (approximate for streaming)
            # For exact count we count the full string afterwards
            tokens = len(tiktoken.get_encoding("cl100k_base").encode(full_content))

            tps = tokens / duration if duration > 0 else 0

            # Add assistant response to conversation history
            messages.append({"role": "assistant", "content": full_content})

            # Auto-save session after each interaction
            save_session(messages)

            # Update token count
            current_tokens = count_tokens(messages)

            # Stats line
            console.print(
                Panel(
                    f"[dim]Stats: {tokens} tokens in {duration:.2f}s ({tps:.2f} tok/s) | "
                    f"Total msgs: {len(messages)} | Tokens: ~{current_tokens}/{MAX_INPUT_TOKENS}[/dim]",
                    box=box.SIMPLE,
                    expand=False,
                )
            )

        except KeyboardInterrupt:
            console.print(
                "\n\n[yellow]Interrupted. Type /quit to exit or continue chatting.[/yellow]\n"
            )
            continue
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            continue


if __name__ == "__main__":
    main()
