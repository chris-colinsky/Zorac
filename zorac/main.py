import atexit
import contextlib
import readline
import time

import tiktoken
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from rich import box
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from .commands import get_help_text, get_system_prompt_commands
from .config import (
    CONFIG_FILE,
    DEFAULT_CONFIG,
    HISTORY_FILE,
    KEEP_RECENT_MESSAGES,
    MAX_INPUT_TOKENS,
    SESSION_FILE,
    ensure_zorac_dir,
    get_bool_setting,
    get_float_setting,
    get_int_setting,
    get_setting,
    is_first_run,
    load_config,
    run_first_time_setup,
    save_config,
)
from .console import console
from .llm import summarize_old_messages
from .session import load_session, save_session
from .utils import check_connection, count_tokens, print_header


def get_initial_system_message() -> str:
    """
    Get the initial system message with command information.

    Returns:
        System message content including assistant role and command info.
    """
    base_message = "You are a helpful assistant."
    command_info = get_system_prompt_commands()
    return f"{base_message}{command_info}"


def setup_readline():
    """Setup command history and persistent storage"""
    try:
        ensure_zorac_dir()

        # Try to read history file if it exists
        if HISTORY_FILE.exists():
            with contextlib.suppress(OSError, PermissionError):
                readline.read_history_file(str(HISTORY_FILE))

        # Try to register history saving on exit
        with contextlib.suppress(OSError, PermissionError):
            atexit.register(readline.write_history_file, str(HISTORY_FILE))

        # Optional: Enable tab completion if desired (simple default)
        # readline.parse_and_bind may fail in some environments (Operation not permitted)
        with contextlib.suppress(OSError, AttributeError):
            readline.parse_and_bind("tab: complete")
    except Exception as e:
        # Only show warning for unexpected errors
        console.print(f"[yellow]Warning: Could not setup command history: {e}[/yellow]")


def get_multiline_input(session: PromptSession) -> str:
    """
    Get multi-line input from the user using prompt_toolkit.

    Users can:
    - Press Enter to submit
    - Paste multi-line text directly (newlines preserved)

    Returns the complete input as a single string.
    """
    print()
    # Create formatted prompt with blue color
    formatted_prompt = FormattedText(
        [
            ("ansiblue bold", "You:"),
            ("", " "),
        ]
    )

    try:
        user_input = session.prompt(formatted_prompt)
        return str(user_input).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def main():
    # Print welcome message first (always show logo and tagline)
    print_header()

    # Check if this is the first run and run setup wizard
    if is_first_run():
        run_first_time_setup()

    # Get configuration settings (these can be updated at runtime via /config)
    VLLM_BASE_URL = get_setting("VLLM_BASE_URL", "http://localhost:8000/v1")
    VLLM_API_KEY = get_setting("VLLM_API_KEY", "EMPTY")
    VLLM_MODEL = get_setting("VLLM_MODEL", "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ")

    # Load model parameters (configurable via .env or ~/.zorac/config.json)
    temperature = get_float_setting("TEMPERATURE", 0.1)
    max_output_tokens = get_int_setting("MAX_OUTPUT_TOKENS", 4000)
    stream_enabled = get_bool_setting("STREAM", True)

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
        # Initialize conversation memory with system message including command info
        messages = [{"role": "system", "content": get_initial_system_message()}]

    # Create prompt session for input
    # Pasted multi-line text is preserved, Enter submits
    prompt_session: PromptSession = PromptSession()

    # Interactive loop
    while True:
        try:
            # Get user input (supports pasting multi-line text)
            try:
                user_input = get_multiline_input(prompt_session)
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

            if user_input.lower() == "/help":
                # Display all available commands
                console.print(f"\n{get_help_text()}\n")
                continue

            if user_input.lower() == "/clear":
                messages = [{"role": "system", "content": get_initial_system_message()}]
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

            if user_input.lower().startswith("/config"):
                parts = user_input.split()
                if len(parts) == 1 or parts[1] == "list":
                    # List Current Settings (Effective)
                    console.print("\n[bold]Configuration:[/bold]")
                    console.print(f"  VLLM_BASE_URL:      [cyan]{VLLM_BASE_URL}[/cyan]")
                    console.print(f"  VLLM_MODEL:         [cyan]{VLLM_MODEL}[/cyan]")
                    console.print(f"  VLLM_API_KEY:       [dim]{VLLM_API_KEY[:4]}...[/dim]")
                    console.print(f"  MAX_INPUT_TOKENS:   [cyan]{MAX_INPUT_TOKENS}[/cyan]")
                    console.print(f"  MAX_OUTPUT_TOKENS:  [cyan]{max_output_tokens}[/cyan]")
                    console.print(f"  KEEP_RECENT_MESSAGES: [cyan]{KEEP_RECENT_MESSAGES}[/cyan]")
                    console.print(f"  TEMPERATURE:        [cyan]{temperature}[/cyan]")
                    console.print(f"  STREAM:             [cyan]{stream_enabled}[/cyan]")
                    console.print(f"  Config File:        [dim]{CONFIG_FILE}[/dim]\n")

                elif parts[1] == "set" and len(parts) >= 4:
                    key = parts[2].upper()
                    value = " ".join(parts[3:])

                    if key not in DEFAULT_CONFIG:
                        console.print(f"\n[red]Unknown setting: {key}[/red]")
                        console.print(f"Available keys: {', '.join(DEFAULT_CONFIG.keys())}\n")
                    else:
                        config = load_config()
                        config[key] = value
                        if save_config(config):
                            console.print(f"\n[green]✓ Updated {key} in {CONFIG_FILE}[/green]")

                            # Note: We also need to update the runtime variables if needed
                            # This is a bit simplistic but works for immediate feedback
                            if key == "VLLM_BASE_URL":
                                VLLM_BASE_URL = value.strip().rstrip("/")
                                client.base_url = VLLM_BASE_URL
                                console.print(
                                    "[yellow]⚠ Base URL updated. Connection will be verified on next message.[/yellow]"
                                )
                            elif key == "VLLM_MODEL":
                                VLLM_MODEL = value
                            elif key == "VLLM_API_KEY":
                                VLLM_API_KEY = value
                                client.api_key = VLLM_API_KEY
                            elif key == "TEMPERATURE":
                                try:
                                    temperature = float(value)
                                    console.print(
                                        "[green]✓ Temperature will take effect on next message.[/green]"
                                    )
                                except ValueError:
                                    console.print(
                                        f"[red]✗ Invalid temperature value: {value}[/red]"
                                    )
                            elif key == "MAX_OUTPUT_TOKENS":
                                try:
                                    max_output_tokens = int(value)
                                    console.print(
                                        "[green]✓ Max output tokens will take effect on next message.[/green]"
                                    )
                                except ValueError:
                                    console.print(
                                        f"[red]✗ Invalid max output tokens value: {value}[/red]"
                                    )
                            elif key == "STREAM":
                                stream_enabled = value.lower() in ("true", "1", "yes", "on")
                                console.print(
                                    f"[green]✓ Streaming {'enabled' if stream_enabled else 'disabled'}.[/green]"
                                )

                            console.print("\n")

                elif parts[1] == "get" and len(parts) >= 3:
                    key = parts[2].upper()
                    settings_map = {
                        "VLLM_BASE_URL": VLLM_BASE_URL,
                        "VLLM_MODEL": VLLM_MODEL,
                        "VLLM_API_KEY": VLLM_API_KEY,
                        "MAX_INPUT_TOKENS": MAX_INPUT_TOKENS,
                        "MAX_OUTPUT_TOKENS": max_output_tokens,
                        "KEEP_RECENT_MESSAGES": KEEP_RECENT_MESSAGES,
                        "TEMPERATURE": temperature,
                        "STREAM": stream_enabled,
                    }
                    if key in settings_map:
                        console.print(f"\n{key} = {settings_map[key]}\n")
                    else:
                        console.print(f"\n[red]Unknown setting: {key}[/red]\n")
                        console.print(f"Available settings: {', '.join(settings_map.keys())}\n")

                else:
                    console.print("\n[bold]Usage:[/bold]")
                    console.print("  /config list               - Show current configuration")
                    console.print("  /config set <KEY> <VALUE>  - set a configuration value")
                    console.print(
                        "  /config get <KEY>          - Get a specific configuration value\n"
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
                    # Process the response
                    if stream_enabled:
                        # Streaming mode
                        stream_response = client.chat.completions.create(
                            model=VLLM_MODEL,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_output_tokens,
                            stream=True,
                        )
                        with Live("", refresh_per_second=10, vertical_overflow="visible") as live:
                            for chunk in stream_response:
                                if chunk.choices[0].delta.content:
                                    # Stop the loading animation on first chunk
                                    if not first_chunk_received:
                                        status.stop()
                                        first_chunk_received = True

                                    content_chunk = chunk.choices[0].delta.content
                                    full_content += content_chunk
                                    live.update(Markdown(full_content))
                    else:
                        # Non-streaming mode
                        completion_response = client.chat.completions.create(
                            model=VLLM_MODEL,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_output_tokens,
                            stream=False,
                        )
                        status.stop()
                        full_content = completion_response.choices[0].message.content or ""
                        console.print(Markdown(full_content))

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
