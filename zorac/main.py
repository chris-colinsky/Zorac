import asyncio
import contextlib
import datetime
import time
from collections.abc import Callable, Coroutine
from typing import Any

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich import box
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.live import Live
from rich.panel import Panel

from .commands import get_help_text, get_system_prompt_commands
from .config import (
    CODE_THEME,
    CONFIG_FILE,
    DEFAULT_CONFIG,
    HISTORY_FILE,
    KEEP_RECENT_MESSAGES,
    MAX_INPUT_TOKENS,
    SESSION_FILE,
    TIKTOKEN_ENCODING,
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
from .markdown_custom import LeftAlignedMarkdown as Markdown
from .session import load_session, save_session
from .utils import check_connection, count_tokens, print_header

# Content width as percentage of console width (60% width, 40% remains unused on right)
CONTENT_WIDTH_PCT = 0.6


class ConstrainedWidth:
    """Wrapper that constrains renderable to a specific width percentage"""

    def __init__(self, renderable: RenderableType, width_pct: float = 0.6):
        self.renderable = renderable
        self.width_pct = width_pct

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render with constrained width"""
        # Calculate target width as percentage of console width
        max_width = int(console.width * self.width_pct)
        # Update options with constrained width
        new_options = options.update(max_width=max_width)
        # Render the wrapped content with new width constraint
        yield from console.render(self.renderable, new_options)


class SlashCommandCompleter(Completer):
    """Completer that only suggests commands when input starts with /"""

    def __init__(self, commands: list[str]):
        self.commands = sorted(commands)

    def get_completions(self, document, _complete_event):
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return
        for command in self.commands:
            if command.lower().startswith(text.lower()):
                yield Completion(command, start_position=-len(text))


def get_initial_system_message() -> str:
    """Get the initial system message with command information and current date."""
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    base_message = f"You are Zorac, a helpful AI assistant. Today's date is {today}."
    command_info = get_system_prompt_commands()
    return f"{base_message}{command_info}"


class Zorac:
    """Main application class for Zorac CLI"""

    def __init__(self):
        self.vllm_base_url = ""
        self.vllm_api_key = ""
        self.vllm_model = ""
        self.temperature = 0.1
        self.max_output_tokens = 4000
        self.stream_enabled = True
        self.tiktoken_encoding = TIKTOKEN_ENCODING
        self.code_theme = CODE_THEME
        self.encoding = tiktoken.get_encoding(self.tiktoken_encoding)

        self.client: AsyncOpenAI | None = None
        self.is_connected = False
        self.messages: list[ChatCompletionMessageParam] = []
        self.prompt_session: PromptSession | None = None
        self.running = True
        self.session_start_time: float = 0.0
        self._current_date: datetime.date | None = None

        # Command registry mapping triggers to handler methods
        self.command_handlers: dict[str, Callable[[list[str]], Coroutine[Any, Any, None]]] = {
            "/quit": self.cmd_quit,
            "/exit": self.cmd_quit,
            "/help": self.cmd_help,
            "/clear": self.cmd_clear,
            "/save": self.cmd_save,
            "/load": self.cmd_load,
            "/tokens": self.cmd_tokens,
            "/summarize": self.cmd_summarize,
            "/summary": self.cmd_summary,
            "/config": self.cmd_config,
            "/reconnect": self.cmd_reconnect,
        }

    async def setup(self):
        """Initial application setup"""
        print_header()

        if is_first_run():
            run_first_time_setup()

        self.load_configuration()
        ensure_zorac_dir()
        self.setup_prompt_session()

        self.client = AsyncOpenAI(base_url=self.vllm_base_url, api_key=self.vllm_api_key)
        self.is_connected = await check_connection(self.client)

        if not self.is_connected:
            console.print(
                "\n[bold yellow]Proceeding offline (some features may not work)...[/bold yellow]"
            )

        self.session_start_time = time.time()

        loaded_messages = load_session()
        if loaded_messages:
            self.messages = loaded_messages
            token_count = count_tokens(self.messages)
            console.print(
                f"[green]✓ Loaded previous session ({len(self.messages)} messages, ~{token_count} tokens)[/green]"
            )
        else:
            self.messages = [{"role": "system", "content": get_initial_system_message()}]

        # Ensure system message has current date (updates loaded sessions from previous days)
        self._update_system_message()

    def load_configuration(self):
        """Load settings from configuration"""
        self.vllm_base_url = get_setting("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.vllm_api_key = get_setting("VLLM_API_KEY", "EMPTY")
        self.vllm_model = get_setting(
            "VLLM_MODEL", "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ"
        )
        self.temperature = get_float_setting("TEMPERATURE", 0.1)
        self.max_output_tokens = get_int_setting("MAX_OUTPUT_TOKENS", 4000)
        self.stream_enabled = get_bool_setting("STREAM", True)
        self.tiktoken_encoding = get_setting("TIKTOKEN_ENCODING", TIKTOKEN_ENCODING)
        self.code_theme = get_setting("CODE_THEME", CODE_THEME)
        try:
            self.encoding = tiktoken.get_encoding(self.tiktoken_encoding)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def setup_prompt_session(self):
        """Setup prompt_toolkit session with history and completion"""
        # Collect all command triggers for auto-completion
        all_triggers = list(self.command_handlers.keys())

        self.prompt_session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            completer=SlashCommandCompleter(all_triggers),
        )

    def _update_system_message(self):
        """Update the system message with current date."""
        self._current_date = datetime.date.today()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": get_initial_system_message()}

    def _refresh_system_date(self):
        """Check if date has changed since last update and refresh if needed."""
        today = datetime.date.today()
        if today != self._current_date:
            self._update_system_message()

    async def get_multiline_input(self) -> str:
        """Get multi-line input from the user using prompt_toolkit."""
        assert self.prompt_session is not None, "Prompt session not initialized."
        formatted_prompt = FormattedText([("ansiblue bold", "You:"), ("", " ")])
        try:
            # Use prompt_async for non-blocking input
            user_input = await self.prompt_session.prompt_async(formatted_prompt)
            return str(user_input).strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    async def run(self):
        """Main interactive loop"""
        await self.setup()

        while self.running:
            try:
                # Use patch_stdout to ensure rich output doesn't break prompt-toolkit
                with patch_stdout():
                    print()
                    user_input = await self.get_multiline_input()

                if not user_input:
                    if not self.running:
                        break
                    continue

                # Process commands
                if user_input.startswith("/"):
                    parts = user_input.split()
                    cmd = parts[0].lower()
                    if cmd in self.command_handlers:
                        await self.command_handlers[cmd](parts)
                        continue
                    else:
                        console.print(
                            f"\n[red]Unknown command: {cmd}. Type /help for available commands.[/red]\n"
                        )
                        continue

                # Process normal chat
                await self.handle_chat(user_input)

            except EOFError:
                await self.cmd_quit([])
            except KeyboardInterrupt:
                console.print(
                    "\n\n[yellow]Interrupted. Type /quit to exit or continue chatting.[/yellow]\n"
                )
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]\n")

    async def handle_chat(self, user_input: str):
        """Process a standard chat interaction"""
        assert self.client is not None, "Client not initialized. Call setup() first."

        # Keep system message date current for long-running sessions
        self._refresh_system_date()

        if not self.is_connected:
            console.print()
            self.is_connected = await check_connection(self.client)
            if not self.is_connected:
                console.print(
                    "\n[bold yellow]Still offline. Use /reconnect to retry or check your server.[/bold yellow]\n"
                )
                return

        self.messages.append({"role": "user", "content": user_input})

        # Check token count and summarize if needed
        current_tokens = count_tokens(self.messages)
        if current_tokens > MAX_INPUT_TOKENS:
            self.messages = await summarize_old_messages(self.client, self.messages)

        start_time = time.time()
        console.print("\n[bold purple]Assistant:[/bold purple]")

        full_content = ""
        first_chunk_received = False

        with console.status(
            "[bold purple]Thinking...[/bold purple]", spinner="dots", spinner_style="purple"
        ) as status:
            try:
                if self.stream_enabled:
                    stream_response = await self.client.chat.completions.create(
                        model=self.vllm_model,
                        messages=self.messages,
                        temperature=self.temperature,
                        max_tokens=self.max_output_tokens,
                        stream=True,
                    )
                    with Live("", refresh_per_second=10) as live:
                        async for chunk in stream_response:
                            if chunk.choices[0].delta.content:
                                if not first_chunk_received:
                                    status.stop()
                                    first_chunk_received = True

                                content_chunk = chunk.choices[0].delta.content
                                full_content += content_chunk
                                markdown_content = Markdown(
                                    full_content, justify="left", code_theme=self.code_theme
                                )
                                constrained_content = ConstrainedWidth(
                                    markdown_content, CONTENT_WIDTH_PCT
                                )
                                live.update(constrained_content)
                else:
                    completion_response = await self.client.chat.completions.create(
                        model=self.vllm_model,
                        messages=self.messages,
                        temperature=self.temperature,
                        max_tokens=self.max_output_tokens,
                        stream=False,
                    )
                    status.stop()
                    full_content = completion_response.choices[0].message.content or ""
                    markdown_content = Markdown(
                        full_content, justify="left", code_theme=self.code_theme
                    )
                    constrained_content = ConstrainedWidth(markdown_content, CONTENT_WIDTH_PCT)
                    console.print(constrained_content)

            except Exception as e:
                console.print(f"[red]Error receiving response: {e}[/red]")
                full_content += f"\n[Error: {e}]"

        end_time = time.time()
        duration = end_time - start_time
        tokens = len(self.encoding.encode(full_content))
        tps = tokens / duration if duration > 0 else 0

        self.messages.append({"role": "assistant", "content": full_content})
        save_session(self.messages)
        current_tokens = count_tokens(self.messages)

        console.print(
            f"\n[dim]Stats: {tokens} tokens in {duration:.2f}s ({tps:.2f} tok/s) | "
            f"Total: {len(self.messages)} msgs, ~{current_tokens}/{MAX_INPUT_TOKENS} tokens[/dim]\n"
        )

    # --- Command Handlers ---

    async def cmd_quit(self, _args: list[str]):
        """Handler for /quit and /exit"""
        save_session(self.messages)
        console.print("\n[green]✓ Session saved. Goodbye![/green]\n")
        self.running = False

    async def cmd_help(self, _args: list[str]):
        """Handler for /help"""
        console.print(f"\n{get_help_text()}\n")

    async def cmd_clear(self, _args: list[str]):
        """Handler for /clear"""
        self.messages = [{"role": "system", "content": get_initial_system_message()}]
        self._current_date = datetime.date.today()
        save_session(self.messages)
        console.print("\n[green]✓ Conversation history cleared and saved![/green]\n")

    async def cmd_save(self, _args: list[str]):
        """Handler for /save"""
        if save_session(self.messages):
            console.print(f"\n[green]✓ Session saved to {SESSION_FILE}[/green]\n")

    async def cmd_load(self, _args: list[str]):
        """Handler for /load"""
        loaded = load_session()
        if loaded:
            self.messages = loaded
            token_count = count_tokens(self.messages)
            console.print(
                f"\n[green]✓ Session reloaded ({len(self.messages)} messages, ~{token_count} tokens)[/green]\n"
            )
        else:
            console.print("\n[red]✗ No saved session found[/red]\n")

    async def cmd_tokens(self, _args: list[str]):
        """Handler for /tokens"""
        token_count = count_tokens(self.messages)
        console.print("\n[bold] Token usage:[/bold]")
        console.print(f"   Current: ~{token_count} tokens")
        console.print(f"   Limit: {MAX_INPUT_TOKENS} tokens")
        console.print(f"   Remaining: ~{MAX_INPUT_TOKENS - token_count} tokens")
        console.print(f"   Messages: {len(self.messages)}\n")

    async def cmd_summarize(self, _args: list[str]):
        """Handler for /summarize"""
        assert self.client is not None, "Client not initialized."
        if len(self.messages) <= KEEP_RECENT_MESSAGES + 1:
            console.print(
                f"\n[yellow]⚠ Not enough messages to summarize. Need more than {KEEP_RECENT_MESSAGES + 1} messages.[/yellow]\n"
            )
        else:
            self.messages = await summarize_old_messages(self.client, self.messages, auto=False)
            save_session(self.messages)
            console.print("[green]✓ Session saved with summary[/green]\n")

    async def cmd_summary(self, _args: list[str]):
        """Handler for /summary"""
        summary_found = False
        if len(self.messages) > 1 and self.messages[1].get("role") == "system":
            content = self.messages[1].get("content", "")
            if isinstance(content, str) and content.startswith("Previous conversation summary:"):
                summary_text = content.replace("Previous conversation summary:", "").strip()
                console.print("\n[bold] Current Conversation Summary:[/bold]\n")
                markdown_content = Markdown(
                    summary_text, justify="left", code_theme=self.code_theme
                )
                constrained_content = ConstrainedWidth(markdown_content, CONTENT_WIDTH_PCT)
                console.print(Panel(constrained_content, box=box.ROUNDED, expand=True))
                console.print()
                summary_found = True

        if not summary_found:
            console.print(
                "\n[yellow]ℹ No summary exists yet. Use /summarize to create one.[/yellow]\n"
            )

    async def cmd_reconnect(self, _args: list[str]):
        """Handler for /reconnect"""
        assert self.client is not None, "Client not initialized."
        self.is_connected = await check_connection(self.client)
        if not self.is_connected:
            console.print("\n[bold yellow]Connection failed. Still offline.[/bold yellow]\n")
        else:
            console.print()

    async def cmd_config(self, args: list[str]):
        """Handler for /config"""
        assert self.client is not None, "Client not initialized."
        if len(args) == 1 or args[1] == "list":
            console.print("\n[bold]Configuration:[/bold]")
            console.print(f"  VLLM_BASE_URL:      [cyan]{self.vllm_base_url}[/cyan]")
            console.print(f"  VLLM_MODEL:         [cyan]{self.vllm_model}[/cyan]")
            console.print(f"  VLLM_API_KEY:       [dim]{self.vllm_api_key[:4]}...[/dim]")
            console.print(f"  MAX_INPUT_TOKENS:   [cyan]{MAX_INPUT_TOKENS}[/cyan]")
            console.print(f"  MAX_OUTPUT_TOKENS:  [cyan]{self.max_output_tokens}[/cyan]")
            console.print(f"  KEEP_RECENT_MESSAGES: [cyan]{KEEP_RECENT_MESSAGES}[/cyan]")
            console.print(f"  TEMPERATURE:        [cyan]{self.temperature}[/cyan]")
            console.print(f"  STREAM:             [cyan]{self.stream_enabled}[/cyan]")
            console.print(f"  TIKTOKEN_ENCODING:  [cyan]{self.tiktoken_encoding}[/cyan]")
            console.print(f"  CODE_THEME:         [cyan]{self.code_theme}[/cyan]")
            console.print(f"  Config File:        [dim]{CONFIG_FILE}[/dim]\n")

        elif args[1] == "set" and len(args) >= 4:
            key = args[2].upper()
            value = " ".join(args[3:])

            if key not in DEFAULT_CONFIG:
                console.print(f"\n[red]Unknown setting: {key}[/red]")
                console.print(f"Available keys: {', '.join(DEFAULT_CONFIG.keys())}\n")
            else:
                # Validate before saving
                if key == "TEMPERATURE":
                    try:
                        float(value)
                    except ValueError:
                        console.print(f"\n[red]✗ Invalid temperature value: {value}[/red]\n")
                        return
                elif key == "MAX_OUTPUT_TOKENS":
                    try:
                        int(value)
                    except ValueError:
                        console.print(f"\n[red]✗ Invalid max output tokens value: {value}[/red]\n")
                        return
                elif key == "TIKTOKEN_ENCODING":
                    try:
                        tiktoken.get_encoding(value)
                    except ValueError:
                        console.print(f"\n[red]✗ Invalid tiktoken encoding: {value}[/red]\n")
                        return

                config = load_config()
                config[key] = value
                if save_config(config):
                    console.print(f"\n[green]✓ Updated {key} in {CONFIG_FILE}[/green]")

                    if key == "VLLM_BASE_URL":
                        self.vllm_base_url = value.strip().rstrip("/")
                        self.client.base_url = self.vllm_base_url
                        console.print(
                            "[yellow]⚠ Base URL updated. Connection will be verified on next message.[/yellow]"
                        )
                    elif key == "VLLM_MODEL":
                        self.vllm_model = value
                    elif key == "VLLM_API_KEY":
                        self.vllm_api_key = value
                        self.client.api_key = self.vllm_api_key
                    elif key == "TEMPERATURE":
                        self.temperature = float(value)
                        console.print(
                            "[green]✓ Temperature will take effect on next message.[/green]"
                        )
                    elif key == "MAX_OUTPUT_TOKENS":
                        self.max_output_tokens = int(value)
                        console.print(
                            "[green]✓ Max output tokens will take effect on next message.[/green]"
                        )
                    elif key == "STREAM":
                        self.stream_enabled = value.lower() in ("true", "1", "yes", "on")
                        console.print(
                            f"[green]✓ Streaming {'enabled' if self.stream_enabled else 'disabled'}.[/green]"
                        )
                    elif key == "TIKTOKEN_ENCODING":
                        self.tiktoken_encoding = value
                        try:
                            self.encoding = tiktoken.get_encoding(value)
                            console.print(f"[green]✓ Tiktoken encoding updated to {value}.[/green]")
                        except Exception:
                            self.encoding = tiktoken.get_encoding("cl100k_base")
                            console.print(
                                f"[yellow]⚠ Invalid encoding {value}, falling back to cl100k_base.[/yellow]"
                            )
                    elif key == "CODE_THEME":
                        self.code_theme = value
                        console.print(f"[green]✓ Code theme updated to {value}.[/green]")
                    console.print("\n")

        elif args[1] == "get" and len(args) >= 3:
            key = args[2].upper()
            settings_map = {
                "VLLM_BASE_URL": self.vllm_base_url,
                "VLLM_MODEL": self.vllm_model,
                "VLLM_API_KEY": self.vllm_api_key,
                "MAX_INPUT_TOKENS": MAX_INPUT_TOKENS,
                "MAX_OUTPUT_TOKENS": self.max_output_tokens,
                "KEEP_RECENT_MESSAGES": KEEP_RECENT_MESSAGES,
                "TEMPERATURE": self.temperature,
                "STREAM": self.stream_enabled,
                "TIKTOKEN_ENCODING": self.tiktoken_encoding,
                "CODE_THEME": self.code_theme,
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
            console.print("  /config get <KEY>          - Get a specific configuration value\n")


def main():
    """Application entry point"""
    app = Zorac()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(app.run())


if __name__ == "__main__":
    main()
