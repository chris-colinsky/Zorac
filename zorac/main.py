"""
Main application module for Zorac — the interactive CLI chat client.

This is the heart of Zorac. It orchestrates all other modules into a cohesive
interactive experience built on Textual, a modern TUI framework.

Architecture: Textual TUI
  ZoracApp extends textual.app.App to provide a full terminal user interface with:
    - A scrollable chat log area for conversation history
    - A persistent stats bar pinned to the bottom that never scrolls away
    - An input bar with autocomplete suggestions for slash commands
    - Real-time streaming markdown rendering during LLM responses

  The layout uses Textual's widget system:
    - VerticalScroll#chat-log: Contains all chat messages as Static/Markdown widgets
    - Input#user-input: Text input with SuggestFromList for slash command completion
    - Static#stats-bar: Always-visible status bar showing performance metrics

Command processing:
  Commands (starting with /) are handled by a dispatch table (self.command_handlers)
  that maps command strings to async handler methods.

  Normal chat messages (not starting with /) go through handle_chat(), which:
    1. Checks/manages the connection
    2. Appends the user message to history
    3. Checks token limits and triggers summarization if needed
    4. Streams the LLM response with live Markdown rendering
    5. Saves the session and displays performance stats
"""

import contextlib
import datetime
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.suggester import SuggestFromList
from textual.widgets import Input, Markdown, Static

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
from .llm import summarize_old_messages
from .session import load_session, save_session
from .utils import count_tokens, get_version


def get_initial_system_message() -> str:
    """Build the initial system message that defines the LLM's behavior.

    The system message serves two purposes:
      1. Identity & behavior: "You are Zorac, a helpful AI assistant"
      2. Command awareness: Includes all available commands so the LLM can
         suggest them naturally when users ask about functionality

    Including the current date allows the LLM to answer time-sensitive questions
    and provides temporal context for the conversation.

    Returns:
        The complete system message string to use as messages[0].
    """
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    base_message = f"You are Zorac, a helpful AI assistant. Today's date is {today}."
    command_info = get_system_prompt_commands()
    return f"{base_message}{command_info}"


class ZoracApp(App):
    """Main Textual application for the Zorac CLI chat client.

    This class extends Textual's App to provide a full TUI with a scrollable
    chat log, persistent stats bar, and input bar with autocomplete.

    The lifecycle is:
        app = ZoracApp()
        app.run()  # Textual handles the event loop

    Key state:
      - self.messages: The conversation history (list of OpenAI message dicts)
      - self.client: The async OpenAI client connected to the vLLM server
      - self.is_connected: Whether we have a working server connection
    """

    CSS = """
    #chat-log {
        padding: 0 1;
    }
    #bottom-bar {
        dock: bottom;
        height: auto;
    }
    #user-input {
        background: #1a1a2e;
    }
    #stats-bar {
        height: auto;
        padding: 1 1;
        background: #0f0f1a;
        color: #888888;
    }
    """

    BINDINGS = [
        ("ctrl+c", "cancel_stream", "Cancel"),
        ("ctrl+d", "quit_app", "Quit"),
    ]

    def __init__(self):
        super().__init__()

        # Server and model settings — populated by load_configuration()
        self.vllm_base_url = ""
        self.vllm_api_key = ""
        self.vllm_model = ""
        self.temperature = 0.1
        self.max_output_tokens = 4000
        self.stream_enabled = True
        self.tiktoken_encoding = TIKTOKEN_ENCODING
        self.code_theme = CODE_THEME

        # tiktoken encoder instance
        self.encoding = tiktoken.get_encoding(self.tiktoken_encoding)

        # Runtime state
        self.client: AsyncOpenAI | None = None
        self.is_connected = False
        self.messages: list[ChatCompletionMessageParam] = []
        self.session_start_time: float = 0.0
        self._current_date: datetime.date | None = None
        self._streaming = False  # True while streaming a response

        # Stats from the most recent chat interaction
        self.stats: dict[str, int | float] = {
            "tokens": 0,
            "duration": 0.0,
            "tps": 0.0,
            "total_msgs": 0,
            "current_tokens": 0,
        }

        # Command history (replaces prompt_toolkit FileHistory)
        self._history: list[str] = []
        self._history_index: int = -1
        self._history_temp: str = ""

        # Command dispatch table
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

    def compose(self) -> ComposeResult:
        """Build the UI layout: chat log, input bar, stats bar."""
        all_triggers = sorted(self.command_handlers.keys())
        yield VerticalScroll(id="chat-log")
        yield Vertical(
            Input(
                id="user-input",
                placeholder="Type your message or /<command>",
                suggester=SuggestFromList(all_triggers, case_sensitive=False),
            ),
            Static(" Ready ", id="stats-bar"),
            id="bottom-bar",
        )

    async def on_mount(self) -> None:
        """Initialize the application after the UI is mounted."""
        await self._setup()

    async def _setup(self) -> None:
        """Initialize: load config, connect to server, restore session."""
        self.load_configuration()
        ensure_zorac_dir()

        self._load_history()

        # Write header first so status messages appear below logo
        self._write_header()

        self.client = AsyncOpenAI(base_url=self.vllm_base_url, api_key=self.vllm_api_key)
        self.is_connected = await self._check_connection()

        if not self.is_connected:
            self._log_system(
                "Proceeding offline (some features may not work)...",
                style="bold yellow",
            )

        self.session_start_time = time.time()

        loaded_messages = load_session()
        if loaded_messages:
            self.messages = loaded_messages
            token_count = count_tokens(self.messages)
            self._log_system(
                f"Loaded previous session ({len(self.messages)} messages, ~{token_count} tokens)",
                style="green",
            )
        else:
            self.messages = [{"role": "system", "content": get_initial_system_message()}]

        self._update_system_message()
        self.query_one("#user-input", Input).focus()

    def load_configuration(self):
        """Load all settings from the configuration system."""
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

    # -------------------------------------------------------------------
    # UI Helpers
    # -------------------------------------------------------------------

    def _log_system(self, text: str, style: str = "dim") -> None:
        """Add a system message to the chat log."""
        chat_log = self.query_one("#chat-log", VerticalScroll)
        widget = Static(f"[{style}]{text}[/{style}]")
        chat_log.mount(widget)
        widget.scroll_visible()

    def _log_user(self, text: str) -> None:
        """Add a user message to the chat log."""
        chat_log = self.query_one("#chat-log", VerticalScroll)
        widget = Static(f"\n[bold blue]You:[/bold blue] {text}")
        chat_log.mount(widget)
        widget.scroll_visible()

    def _update_stats_bar(self) -> None:
        """Update the stats bar with current stats."""
        stats_bar = self.query_one("#stats-bar", Static)
        if self.stats["tokens"] > 0:
            stats_bar.update(
                f" Stats: {int(self.stats['tokens'])} tokens in "
                f"{self.stats['duration']:.1f}s ({self.stats['tps']:.1f} tok/s)"
                f" | Total: {int(self.stats['total_msgs'])} msgs, "
                f"~{int(self.stats['current_tokens'])}/{MAX_INPUT_TOKENS} tokens "
            )
        elif len(self.messages) > 1:
            token_count = count_tokens(self.messages)
            stats_bar.update(f" Session: {len(self.messages)} msgs, ~{token_count} tokens ")
        else:
            stats_bar.update(" Ready ")

    def _write_header(self) -> None:
        """Write the logo and header info to the chat log."""
        chat_log = self.query_one("#chat-log", VerticalScroll)

        logo_text = (
            "\n"
            "[bold purple]"
            "     ███████╗ ██████╗ ██████╗  █████╗  ██████╗\n"
            "     ╚══███╔╝██╔═══██╗██╔══██╗██╔══██╗██╔════╝\n"
            "       ███╔╝ ██║   ██║██████╔╝███████║██║\n"
            "      ███╔╝  ██║   ██║██╔══██╗██╔══██║██║\n"
            "     ███████╗╚██████╔╝██║  ██║██║  ██║╚██████╗\n"
            "     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝"
            "[/bold purple]\n"
            "        [dim italic]intelligence running on localhost[/dim italic]\n"
        )
        chat_log.mount(Static(logo_text))

        config_text = (
            f"[dim]Zorac v{get_version()}[/dim]\n"
            f"[dim]Connected to: {self.vllm_base_url}[/dim]\n"
            f"[dim]Model: {self.vllm_model}[/dim]"
        )
        chat_log.mount(Static(config_text))

        commands_text = (
            "\n[bold]Commands:[/bold]\n"
            "  [green]/clear[/green]     - Clear conversation history\n"
            "  [green]/save[/green]      - Manually save session\n"
            "  [green]/load[/green]      - Reload session from disk\n"
            "  [green]/tokens[/green]    - Show current token usage\n"
            "  [green]/summarize[/green] - Force summarization of conversation\n"
            "  [green]/summary[/green]   - Show current conversation summary\n"
            "  [green]/reconnect[/green] - Retry server connection\n"
            "  [green]/config[/green]    - Manage configuration\n"
            "  [green]/quit[/green]      - Exit the chat\n"
        )
        chat_log.mount(Static(commands_text))

    async def _check_connection(self) -> bool:
        """Verify connection to the vLLM server (TUI-aware)."""
        assert self.client is not None
        stats_bar = self.query_one("#stats-bar", Static)
        stats_bar.update(" Verifying connection... ")
        try:
            await self.client.models.list()
            self._log_system("Connection verified", style="green")
            stats_bar.update(" Ready ")
            return True
        except Exception as e:
            self._log_system("Connection failed!", style="bold red")
            self._log_system(f"Error: {e}", style="red")
            self._log_system(f"Target URL: {self.client.base_url}", style="yellow")
            self._log_system(
                "Review your .env file and ensure the VLLM_BASE_URL is correct "
                "and the server is running.",
                style="yellow",
            )
            stats_bar.update(" Offline ")
            return False

    # -------------------------------------------------------------------
    # History Management
    # -------------------------------------------------------------------

    def _load_history(self) -> None:
        """Load command history from the history file."""
        try:
            if HISTORY_FILE.exists():
                lines = Path(HISTORY_FILE).read_text().splitlines()
                for line in lines:
                    # Handle prompt_toolkit's + prefix format for migration
                    entry = line.lstrip("+").strip()
                    if entry and (not self._history or self._history[-1] != entry):
                        self._history.append(entry)
        except Exception:
            pass

    def _save_history(self) -> None:
        """Save command history to the history file."""
        try:
            ensure_zorac_dir()
            Path(HISTORY_FILE).write_text("\n".join(self._history[-500:]) + "\n")
        except Exception:
            pass

    # -------------------------------------------------------------------
    # Input Handling
    # -------------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value.strip()
        event.input.value = ""
        self._history_index = -1
        self._history_temp = ""

        if not user_input:
            return

        # Save to history
        if not self._history or self._history[-1] != user_input:
            self._history.append(user_input)
            self._save_history()

        # Route input: commands start with "/", everything else is chat
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            if cmd in self.command_handlers:
                await self.command_handlers[cmd](parts)
            else:
                self._log_system(
                    f"Unknown command: {cmd}. Type /help for available commands.",
                    style="red",
                )
        else:
            await self.handle_chat(user_input)

    async def on_key(self, event) -> None:
        """Handle key events for history navigation."""
        input_widget = self.query_one("#user-input", Input)
        if not input_widget.has_focus:
            return

        if event.key == "up":
            event.prevent_default()
            if not self._history:
                return
            if self._history_index == -1:
                self._history_temp = input_widget.value
                self._history_index = len(self._history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            else:
                return
            input_widget.value = self._history[self._history_index]
            input_widget.cursor_position = len(input_widget.value)

        elif event.key == "down":
            event.prevent_default()
            if self._history_index == -1:
                return
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                input_widget.value = self._history[self._history_index]
            else:
                self._history_index = -1
                input_widget.value = self._history_temp
            input_widget.cursor_position = len(input_widget.value)

    # -------------------------------------------------------------------
    # Chat
    # -------------------------------------------------------------------

    async def handle_chat(self, user_input: str) -> None:
        """Process a standard chat interaction with the LLM."""
        assert self.client is not None

        self._refresh_system_date()

        # Lazy connection check
        if not self.is_connected:
            self.is_connected = await self._check_connection()
            if not self.is_connected:
                self._log_system(
                    "Still offline. Use /reconnect to retry or check your server.",
                    style="bold yellow",
                )
                return

        self._log_user(user_input)
        self.messages.append({"role": "user", "content": user_input})

        current_tokens = count_tokens(self.messages)
        if current_tokens > MAX_INPUT_TOKENS:
            await self._summarize_messages()

        # Disable input during streaming
        input_widget = self.query_one("#user-input", Input)
        input_widget.disabled = True
        self._streaming = True

        self._stream_response()

    @work(exclusive=True, group="stream")
    async def _stream_response(self) -> None:
        """Stream the LLM response (runs as a Textual worker)."""
        assert self.client is not None
        from textual.worker import get_current_worker

        worker = get_current_worker()
        chat_log = self.query_one("#chat-log", VerticalScroll)
        stats_bar = self.query_one("#stats-bar", Static)
        input_widget = self.query_one("#user-input", Input)

        # Add assistant label
        label = Static("\n[bold purple]Assistant:[/bold purple]")
        chat_log.mount(label)
        label.scroll_visible()

        stats_bar.update(" Thinking... ")

        full_content = ""
        start_time = time.time()

        try:
            if self.stream_enabled:
                stream_response = await self.client.chat.completions.create(
                    model=self.vllm_model,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    stream=True,
                )

                # Create a Markdown widget for the response
                md_widget = Markdown("")
                chat_log.mount(md_widget)
                md_widget.scroll_visible()

                stream = Markdown.get_stream(md_widget)
                stream_tokens = 0

                async for chunk in stream_response:
                    if worker.is_cancelled:
                        break

                    if chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        full_content += content_chunk
                        stream_tokens += len(self.encoding.encode(content_chunk))
                        elapsed = time.time() - start_time
                        tps = stream_tokens / elapsed if elapsed > 0 else 0

                        await stream.write(content_chunk)
                        chat_log.scroll_end(animate=False)

                        stats_bar.update(
                            f" {stream_tokens} tokens | {elapsed:.1f}s | {tps:.1f} tok/s "
                        )

                await stream.stop()
                chat_log.scroll_end()

            else:
                # Non-streaming mode
                completion_response = await self.client.chat.completions.create(
                    model=self.vllm_model,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    stream=False,
                )
                full_content = completion_response.choices[0].message.content or ""
                md_widget = Markdown(full_content)
                chat_log.mount(md_widget)
                md_widget.scroll_visible()

        except Exception as e:
            if not worker.is_cancelled:
                self._log_system(f"Error receiving response: {e}", style="red")
                full_content += f"\n[Error: {e}]"

        # Performance metrics
        end_time = time.time()
        duration = end_time - start_time
        tokens = len(self.encoding.encode(full_content)) if full_content else 0
        tps = tokens / duration if duration > 0 else 0

        # Save response
        if full_content and not worker.is_cancelled:
            self.messages.append({"role": "assistant", "content": full_content})
            save_session(self.messages)

        current_tokens = count_tokens(self.messages)
        self.stats = {
            "tokens": tokens,
            "duration": duration,
            "tps": tps,
            "total_msgs": len(self.messages),
            "current_tokens": current_tokens,
        }
        self._update_stats_bar()

        # Re-enable input
        self._streaming = False
        input_widget.disabled = False
        input_widget.focus()

    # -------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------

    def action_cancel_stream(self) -> None:
        """Handle Ctrl+C — cancel streaming or ignore."""
        if self._streaming:
            self.workers.cancel_group(self, "stream")
            self._streaming = False
            input_widget = self.query_one("#user-input", Input)
            input_widget.disabled = False
            input_widget.focus()
            self._log_system("Response interrupted.", style="yellow")
        # If not streaming, ignore Ctrl+C (don't quit)

    def action_quit_app(self) -> None:
        """Handle Ctrl+D — save and quit."""
        save_session(self.messages)
        self.exit()

    # -------------------------------------------------------------------
    # System message management
    # -------------------------------------------------------------------

    def _update_system_message(self):
        """Update the system message with today's date."""
        self._current_date = datetime.date.today()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": get_initial_system_message()}

    def _refresh_system_date(self):
        """Check if the date has changed and update the system message if needed."""
        today = datetime.date.today()
        if today != self._current_date:
            self._update_system_message()

    # -------------------------------------------------------------------
    # Summarization
    # -------------------------------------------------------------------

    async def _summarize_messages(self, auto: bool = True) -> None:
        """Summarize messages with TUI feedback."""
        assert self.client is not None
        stats_bar = self.query_one("#stats-bar", Static)

        if auto:
            self._log_system(
                "Token limit approaching. Summarizing conversation history...",
                style="yellow",
            )
        else:
            self._log_system(
                "Summarizing conversation history as requested...",
                style="green",
            )

        stats_bar.update(" Summarizing... ")
        self.messages = await summarize_old_messages(self.client, self.messages, auto=auto)
        self._update_stats_bar()

    # -------------------------------------------------------------------
    # Command Handlers
    # -------------------------------------------------------------------

    async def cmd_quit(self, _args: list[str]):
        """Handle /quit and /exit — save session and exit."""
        save_session(self.messages)
        self._log_system("Session saved. Goodbye!", style="green")
        self.exit()

    async def cmd_help(self, _args: list[str]):
        """Handle /help — display all available commands."""
        self._log_system(get_help_text())

    async def cmd_clear(self, _args: list[str]):
        """Handle /clear — reset conversation to a fresh state."""
        self.messages = [{"role": "system", "content": get_initial_system_message()}]
        self._current_date = datetime.date.today()
        save_session(self.messages)

        chat_log = self.query_one("#chat-log", VerticalScroll)
        await chat_log.remove_children()
        self._write_header()
        self._log_system("Conversation history cleared and saved!", style="green")

        self.stats = {
            "tokens": 0,
            "duration": 0.0,
            "tps": 0.0,
            "total_msgs": 0,
            "current_tokens": 0,
        }
        self._update_stats_bar()

    async def cmd_save(self, _args: list[str]):
        """Handle /save — manually save the current session."""
        if save_session(self.messages):
            self._log_system(f"Session saved to {SESSION_FILE}", style="green")

    async def cmd_load(self, _args: list[str]):
        """Handle /load — reload the session from disk."""
        loaded = load_session()
        if loaded:
            self.messages = loaded
            token_count = count_tokens(self.messages)
            self._log_system(
                f"Session reloaded ({len(self.messages)} messages, ~{token_count} tokens)",
                style="green",
            )
            self._update_stats_bar()
        else:
            self._log_system("No saved session found", style="red")

    async def cmd_tokens(self, _args: list[str]):
        """Handle /tokens — display current token usage statistics."""
        token_count = count_tokens(self.messages)
        self._log_system(
            f"Token usage:\n"
            f"   Current: ~{token_count} tokens\n"
            f"   Limit: {MAX_INPUT_TOKENS} tokens\n"
            f"   Remaining: ~{MAX_INPUT_TOKENS - token_count} tokens\n"
            f"   Messages: {len(self.messages)}"
        )

    async def cmd_summarize(self, _args: list[str]):
        """Handle /summarize — force conversation summarization."""
        assert self.client is not None
        if len(self.messages) <= KEEP_RECENT_MESSAGES + 1:
            self._log_system(
                f"Not enough messages to summarize. Need more than "
                f"{KEEP_RECENT_MESSAGES + 1} messages.",
                style="yellow",
            )
        else:
            await self._summarize_messages(auto=False)
            save_session(self.messages)
            self._log_system("Session saved with summary", style="green")

    async def cmd_summary(self, _args: list[str]):
        """Handle /summary — display the current conversation summary."""
        summary_found = False
        if len(self.messages) > 1 and self.messages[1].get("role") == "system":
            content = self.messages[1].get("content", "")
            if isinstance(content, str) and content.startswith("Previous conversation summary:"):
                summary_text = content.replace("Previous conversation summary:", "").strip()
                chat_log = self.query_one("#chat-log", VerticalScroll)
                chat_log.mount(Static("\n[bold]Current Conversation Summary:[/bold]"))
                md_widget = Markdown(summary_text)
                chat_log.mount(md_widget)
                md_widget.scroll_visible()
                summary_found = True

        if not summary_found:
            self._log_system(
                "No summary exists yet. Use /summarize to create one.",
                style="yellow",
            )

    async def cmd_reconnect(self, _args: list[str]):
        """Handle /reconnect — retry connection to the vLLM server."""
        assert self.client is not None
        self.is_connected = await self._check_connection()
        if not self.is_connected:
            self._log_system("Connection failed. Still offline.", style="bold yellow")

    async def cmd_config(self, args: list[str]):
        """Handle /config — manage configuration settings at runtime."""
        assert self.client is not None

        if len(args) == 1 or args[1] == "list":
            self._log_system(
                f"Configuration:\n"
                f"  VLLM_BASE_URL:      {self.vllm_base_url}\n"
                f"  VLLM_MODEL:         {self.vllm_model}\n"
                f"  VLLM_API_KEY:       {self.vllm_api_key[:4]}...\n"
                f"  MAX_INPUT_TOKENS:   {MAX_INPUT_TOKENS}\n"
                f"  MAX_OUTPUT_TOKENS:  {self.max_output_tokens}\n"
                f"  KEEP_RECENT_MESSAGES: {KEEP_RECENT_MESSAGES}\n"
                f"  TEMPERATURE:        {self.temperature}\n"
                f"  STREAM:             {self.stream_enabled}\n"
                f"  TIKTOKEN_ENCODING:  {self.tiktoken_encoding}\n"
                f"  CODE_THEME:         {self.code_theme}\n"
                f"  Config File:        {CONFIG_FILE}"
            )

        elif args[1] == "set" and len(args) >= 4:
            key = args[2].upper()
            value = " ".join(args[3:])

            if key not in DEFAULT_CONFIG:
                self._log_system(
                    f"Unknown setting: {key}\nAvailable keys: {', '.join(DEFAULT_CONFIG.keys())}",
                    style="red",
                )
            else:
                # Validate
                if key == "TEMPERATURE":
                    try:
                        float(value)
                    except ValueError:
                        self._log_system(f"Invalid temperature value: {value}", style="red")
                        return
                elif key == "MAX_OUTPUT_TOKENS":
                    try:
                        int(value)
                    except ValueError:
                        self._log_system(f"Invalid max output tokens value: {value}", style="red")
                        return
                elif key == "TIKTOKEN_ENCODING":
                    try:
                        tiktoken.get_encoding(value)
                    except ValueError:
                        self._log_system(f"Invalid tiktoken encoding: {value}", style="red")
                        return

                config = load_config()
                config[key] = value
                if save_config(config):
                    self._log_system(f"Updated {key} in {CONFIG_FILE}", style="green")

                    if key == "VLLM_BASE_URL":
                        self.vllm_base_url = value.strip().rstrip("/")
                        self.client.base_url = self.vllm_base_url
                        self._log_system(
                            "Base URL updated. Connection will be verified on next message.",
                            style="yellow",
                        )
                    elif key == "VLLM_MODEL":
                        self.vllm_model = value
                    elif key == "VLLM_API_KEY":
                        self.vllm_api_key = value
                        self.client.api_key = self.vllm_api_key
                    elif key == "TEMPERATURE":
                        self.temperature = float(value)
                        self._log_system(
                            "Temperature will take effect on next message.", style="green"
                        )
                    elif key == "MAX_OUTPUT_TOKENS":
                        self.max_output_tokens = int(value)
                        self._log_system(
                            "Max output tokens will take effect on next message.", style="green"
                        )
                    elif key == "STREAM":
                        self.stream_enabled = value.lower() in ("true", "1", "yes", "on")
                        self._log_system(
                            f"Streaming {'enabled' if self.stream_enabled else 'disabled'}.",
                            style="green",
                        )
                    elif key == "TIKTOKEN_ENCODING":
                        self.tiktoken_encoding = value
                        try:
                            self.encoding = tiktoken.get_encoding(value)
                            self._log_system(
                                f"Tiktoken encoding updated to {value}.", style="green"
                            )
                        except Exception:
                            self.encoding = tiktoken.get_encoding("cl100k_base")
                            self._log_system(
                                f"Invalid encoding {value}, falling back to cl100k_base.",
                                style="yellow",
                            )
                    elif key == "CODE_THEME":
                        self.code_theme = value
                        self._log_system(f"Code theme updated to {value}.", style="green")

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
                self._log_system(f"{key} = {settings_map[key]}")
            else:
                self._log_system(
                    f"Unknown setting: {key}\nAvailable settings: {', '.join(settings_map.keys())}",
                    style="red",
                )
        else:
            self._log_system(
                "Usage:\n"
                "  /config list               - Show current configuration\n"
                "  /config set <KEY> <VALUE>  - Set a configuration value\n"
                "  /config get <KEY>          - Get a specific configuration value"
            )


def main():
    """Application entry point — creates and runs the ZoracApp.

    First-time setup runs BEFORE Textual enters the alternate screen,
    since run_first_time_setup() uses input() which doesn't work in
    Textual's alternate screen.
    """
    if is_first_run():
        from .utils import print_header

        print_header()
        run_first_time_setup()

    app = ZoracApp()
    with contextlib.suppress(KeyboardInterrupt):
        app.run()


if __name__ == "__main__":
    main()
