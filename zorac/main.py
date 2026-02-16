"""
Main application module for Zorac — the interactive CLI chat client.

This is the heart of Zorac. It orchestrates all other modules into a cohesive
interactive experience built on Textual, a modern TUI framework.

The main components are:

  1. **ZoracApp class**: The Textual application that manages state, handles
     user input, processes commands, and coordinates chat interactions. It
     extends textual.app.App (with CommandHandlersMixin, StreamingMixin, and
     HistoryMixin) to get lifecycle management, widget composition, key
     bindings, and worker threads for free.

  2. **main()**: The entry point that runs first-time setup (before Textual's
     alternate screen) and launches the TUI.

Architecture: Textual TUI
  ZoracApp extends textual.app.App to provide a full terminal user interface with:
    - A scrollable chat log area for conversation history
    - A persistent stats bar pinned to the bottom that never scrolls away
    - An input bar with autocomplete suggestions for slash commands
    - Real-time streaming markdown rendering during LLM responses

  The layout uses Textual's widget system:
    - VerticalScroll#chat-log: Contains all chat messages as Static/Markdown widgets
    - ChatInput#user-input: Multiline text input with command suggestions (Enter submits, Shift+Enter for newlines)
    - Static#stats-bar: Always-visible status bar showing performance metrics

  The async architecture follows this flow:
    main() → ZoracApp().run() → Textual event loop → on_mount/_setup → event handlers

Command processing:
  Commands (starting with /) are handled by a dispatch table (self.command_handlers)
  that maps command strings to async handler methods. This pattern is cleaner than
  a long if/elif chain and makes it easy to add new commands — just add the handler
  method and the mapping entry.

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
from typing import Any

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

from .commands import get_initial_system_message
from .config import (
    CODE_THEME,
    MAX_INPUT_TOKENS,
    TIKTOKEN_ENCODING,
    ensure_zorac_dir,
    get_bool_setting,
    get_float_setting,
    get_int_setting,
    get_setting,
    is_first_run,
    run_first_time_setup,
)
from .handlers import CommandHandlersMixin
from .history import HistoryMixin
from .llm import summarize_old_messages
from .session import load_session, save_session
from .streaming import StreamingMixin
from .utils import count_tokens, get_version
from .widgets import ChatInput


class ZoracApp(CommandHandlersMixin, StreamingMixin, HistoryMixin, App):
    """Main Textual application for the Zorac CLI chat client.

    This class follows the "application controller" pattern — it owns all
    application state and coordinates between the various subsystems (config,
    session, LLM, UI). Extending Textual's App gives us:
      - Automatic event loop management (no manual asyncio.run())
      - Widget composition via compose() for declarative UI layout
      - Key bindings for Ctrl+C/Ctrl+D handling
      - Worker threads via @work decorator for non-blocking streaming
      - CSS-based styling for consistent visual design

    The lifecycle is:
        app = ZoracApp()
        app.run()  # Textual handles the event loop, calls on_mount → _setup

    Key state:
      - self.messages: The conversation history (list of OpenAI message dicts)
      - self.client: The async OpenAI client connected to the vLLM server
      - self.is_connected: Whether we have a working server connection
      - self._streaming: Whether a response is currently being streamed
    """

    # Textual CSS for layout and styling. Textual uses a CSS subset (TCSS) that
    # supports properties like dock, padding, background, and color. This replaces
    # the prompt_toolkit PtStyle dict from the previous architecture.
    CSS = """
    #chat-log {
        padding: 0 1;
    }
    #bottom-bar {
        dock: bottom;
        height: auto;
    }
    #user-input {
        height: 3;
        min-height: 3;
        max-height: 7;
        border: tall $accent-darken-3;
        padding: 0 1;
    }
    #stats-bar {
        height: auto;
        padding: 1 1;
        background: #0f0f1a;
        color: #888888;
    }
    """

    # Key bindings map keyboard shortcuts to action methods (action_*).
    # Ctrl+C cancels the current stream (not quit), Ctrl+D saves and exits.
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

        # tiktoken encoder instance — used for counting tokens in responses.
        # Initialized here and updated if the user changes TIKTOKEN_ENCODING
        # via /config. We keep a pre-loaded encoder to avoid the cost of
        # loading it on every token count operation.
        self.encoding = tiktoken.get_encoding(self.tiktoken_encoding)

        # Runtime state
        self.client: AsyncOpenAI | None = None  # Set during _setup()
        self.is_connected = False  # Updated by _check_connection()
        self.messages: list[ChatCompletionMessageParam] = []  # Conversation history
        self.session_start_time: float = 0.0  # For calculating session duration
        self._current_date: datetime.date | None = None  # Tracks date for system message refresh
        self._streaming = False  # True while streaming a response (disables input)

        # Stats from the most recent chat interaction, displayed in the stats bar
        self.stats: dict[str, int | float] = {
            "tokens": 0,
            "duration": 0.0,
            "tps": 0.0,
            "total_msgs": 0,
            "current_tokens": 0,
        }

        # Command history — replaces prompt_toolkit's FileHistory with a simple
        # in-memory list that's loaded from / saved to ~/.zorac/history.
        # _history_index tracks position during Up/Down navigation (-1 = not navigating).
        # _history_temp stores the user's in-progress input when they start navigating.
        self._history: list[str] = []
        self._history_index: int = -1
        self._history_temp: str = ""

        # Command dispatch table: maps command strings to async handler methods.
        self.command_handlers: dict[str, Callable[[list[str]], Coroutine[Any, Any, None]]] = {
            "/quit": self.cmd_quit,
            "/exit": self.cmd_quit,  # Alias — both map to the same handler
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
        # Collect all command triggers for the autocomplete suggestion list
        all_triggers = sorted(self.command_handlers.keys())
        yield VerticalScroll(id="chat-log")
        yield Vertical(
            ChatInput(
                commands=all_triggers,
                id="user-input",
                placeholder="Type your message or /command",
            ),
            Static(" Ready ", id="stats-bar"),
            id="bottom-bar",
        )

    async def on_mount(self) -> None:
        """Initialize the application after the UI is mounted."""
        await self._setup()

    async def _setup(self) -> None:
        """Initialize the application: load config, connect to server, restore session.

        Setup sequence:
          1. Load configuration from all sources (env, file, defaults)
          2. Ensure data directory exists (~/.zorac/)
          3. Load command history for Up/Down arrow navigation
          4. Display welcome header (ASCII art + connection info)
          5. Create OpenAI client and verify connection
          6. Load previous session or create a fresh one
          7. Update system message with current date
        """
        self.load_configuration()
        ensure_zorac_dir()

        self._load_history()

        # Write header first so status messages appear below the logo
        self._write_header()

        # Create the async OpenAI client. We use OpenAI's client library because
        # vLLM implements the OpenAI-compatible API, so we get a well-tested,
        # maintained client for free.
        self.client = AsyncOpenAI(base_url=self.vllm_base_url, api_key=self.vllm_api_key)
        self.is_connected = await self._check_connection()

        if not self.is_connected:
            self._log_system(
                "Proceeding offline (some features may not work)...",
                style="bold yellow",
            )

        self.session_start_time = time.time()

        # Try to restore the previous conversation session from disk.
        loaded_messages = load_session()
        if loaded_messages:
            self.messages = loaded_messages
            token_count = count_tokens(self.messages)
            self._log_system(
                f"Loaded previous session ({len(self.messages)} messages, ~{token_count} tokens)",
                style="green",
            )
        else:
            # First run or no saved session — start with just the system message
            self.messages = [{"role": "system", "content": get_initial_system_message()}]

        # Update the system message's date in case the session was saved yesterday
        self._update_system_message()
        self.query_one("#user-input", ChatInput).focus()

    def load_configuration(self):
        """Load all settings from the configuration system.

        Reads each setting using the three-tier priority system
        (env var > config file > default). This method is called during
        _setup() and could be called again to reload configuration.
        """
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

        # Update the tiktoken encoder to match the configured encoding.
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
        if "\n" in text:
            # Indent continuation lines to align under the message text
            lines = text.split("\n")
            formatted = lines[0] + "\n" + "\n".join("      " + line for line in lines[1:])
            widget = Static(f"\n[bold blue]You:[/bold blue] {formatted}")
        else:
            widget = Static(f"\n[bold blue]You:[/bold blue] {text}")
        chat_log.mount(widget)
        widget.scroll_visible()

    def _update_stats_bar(self) -> None:
        """Update the bottom stats bar with contextual information."""
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

    def _write_header(self, show_commands: bool = True) -> None:
        """Write the welcome header with ASCII art logo and connection info.

        Args:
            show_commands: Whether to display the commands list. True at startup,
                False after /clear to reduce noise.
        """
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

        if show_commands:
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
        """Verify connection to the vLLM server using the models endpoint.

        Returns:
            True if the server is reachable and responding, False otherwise.
        """
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
    # Input Handling
    # -------------------------------------------------------------------

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle user input when Enter is pressed.

        Routes commands (starting with /) via the command_handlers table,
        and normal text to handle_chat() for LLM interaction.
        """
        user_input = event.value.strip()
        event.input.clear()
        event.input._auto_resize()
        self._history_index = -1
        self._history_temp = ""

        if not user_input:
            return

        # Save to history, deduplicating consecutive identical entries
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
            # Not a command — process as a regular chat message
            await self.handle_chat(user_input)

    # -------------------------------------------------------------------
    # Chat
    # -------------------------------------------------------------------

    async def handle_chat(self, user_input: str) -> None:
        """Process a standard chat interaction with the LLM.

        This is the core chat flow:
          1. Check/refresh server connection (handles midnight date changes too)
          2. Append user message to conversation history
          3. Check token count → trigger auto-summarization if over limit
          4. Disable input and launch the streaming worker

        Args:
            user_input: The user's chat message (already confirmed to not be a command).
        """
        assert self.client is not None

        # Keep system message date current for long-running sessions (across midnight)
        self._refresh_system_date()

        # Lazy connection check: if we're offline, try to reconnect before chatting.
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

        # Check if we've exceeded the token limit and need to summarize.
        current_tokens = count_tokens(self.messages)
        if current_tokens > MAX_INPUT_TOKENS:
            await self._summarize_messages()

        # Disable input during streaming to prevent overlapping requests
        input_widget = self.query_one("#user-input", ChatInput)
        input_widget.disabled = True
        self._streaming = True

        self._stream_response()

    # -------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------

    def action_cancel_stream(self) -> None:
        """Handle Ctrl+C — cancel the current streaming response."""
        if self._streaming:
            self.workers.cancel_group(self, "stream")
            self._streaming = False
            input_widget = self.query_one("#user-input", ChatInput)
            input_widget.disabled = False
            input_widget.focus()
            self._log_system("Response interrupted.", style="yellow")
        # If not streaming, ignore Ctrl+C (don't quit)

    def action_quit_app(self) -> None:
        """Handle Ctrl+D — save the session and exit the application."""
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
        """Summarize messages with TUI-appropriate feedback.

        Args:
            auto: If True, shows "Token limit approaching" warning (triggered
                  automatically). If False, shows "as requested" message
                  (triggered by /summarize command).
        """
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


def main():
    """Application entry point — creates and runs the ZoracApp.

    First-time setup runs BEFORE Textual enters the alternate screen,
    since run_first_time_setup() uses input() which doesn't work in
    Textual's alternate screen. Once setup is complete (or skipped for
    returning users), we launch the Textual app.

    This function is referenced in pyproject.toml as the console script
    entry point: [project.scripts] zorac = "zorac.main:main"
    """
    # First-time setup: guide new users through initial configuration.
    if is_first_run():
        from .utils import print_header

        print_header()
        run_first_time_setup()

    app = ZoracApp()
    with contextlib.suppress(KeyboardInterrupt):
        app.run()


if __name__ == "__main__":
    main()
