"""
Main application module for Zorac — the interactive CLI chat client.

This is the heart of Zorac. It orchestrates all other modules into a cohesive
interactive experience built on Textual, a modern TUI framework.

The main components are:

  1. **ZoracApp class**: The Textual application that manages state, handles
     user input, processes commands, and coordinates chat interactions. It
     extends textual.app.App to get lifecycle management, widget composition,
     key bindings, and worker threads for free.

  2. **get_initial_system_message()**: Builds the system prompt that defines
     the LLM's identity and command awareness.

  3. **main()**: The entry point that runs first-time setup (before Textual's
     alternate screen) and launches the TUI.

Architecture: Textual TUI
  ZoracApp extends textual.app.App to provide a full terminal user interface with:
    - A scrollable chat log area for conversation history
    - A persistent stats bar pinned to the bottom that never scrolls away
    - An input bar with autocomplete suggestions for slash commands
    - Real-time streaming markdown rendering during LLM responses

  Why Textual instead of prompt_toolkit + Rich?
    The previous architecture used prompt_toolkit for input and Rich for output,
    which required careful coordination (patch_stdout, Live context managers).
    Textual unifies both into a single widget-based framework where the entire
    UI is reactive — widgets update in place without flickering, scrolling is
    automatic, and streaming content "just works" via Markdown.get_stream().

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Markdown, Static, TextArea

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


class ChatInput(TextArea):
    """Multiline input widget with Enter to submit and Shift+Enter for newlines.

    Replaces the single-line Input widget to support multiline messages (code
    blocks, formatted text). Enter submits the message, Shift+Enter inserts a
    newline. Handles both "shift+enter" (Kitty keyboard protocol) and "ctrl+j"
    (iTerm2 and other terminals that send a raw newline for Shift+Enter).
    In terminals where Shift+Enter is indistinguishable from Enter (e.g.,
    macOS Terminal.app), Enter always submits — multiline input is still
    possible via paste (bracket paste mode).

    Auto-resizes from 1 to 5 lines based on content. Provides inline command
    suggestions for /commands, accepted with Tab.
    """

    @dataclass
    class Submitted(Message):
        """Posted when the user presses Enter to submit their message."""

        input: "ChatInput"
        value: str

    BINDINGS = [
        Binding("tab", "accept_suggestion", "Accept suggestion", show=False),
    ]

    # Border adds 2 rows (top + bottom). Heights must include this overhead.
    _BORDER_OVERHEAD = 2

    def __init__(
        self,
        commands: list[str] | None = None,
        *,
        id: str | None = None,
        placeholder: str = "",
    ) -> None:
        super().__init__(
            id=id,
            show_line_numbers=False,
            soft_wrap=True,
            tab_behavior="focus",
            highlight_cursor_line=False,
            theme="monokai",
        )
        self._commands = commands or []
        self._placeholder = placeholder
        self._suggestion: str = ""

    def on_mount(self) -> None:
        """Set placeholder text and apply custom theme after mount."""
        self.placeholder = self._placeholder
        # Override the monokai theme's background to match the Zorac UI while
        # keeping its text color for visibility. This is needed because TextArea
        # themes paint their own background over the CSS background property.
        from rich.style import Style

        if self._theme:
            self._theme.base_style = Style(color="#f8f8f2", bgcolor="#1a1a2e")

    async def _on_key(self, event: events.Key) -> None:
        """Handle Enter (submit) and Shift+Enter (newline).

        Shift+Enter is detected as "shift+enter" in terminals with Kitty
        keyboard protocol (kitty, WezTerm) or as "ctrl+j" in terminals
        that send a raw newline character (iTerm2).
        """
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(input=self, value=text))
            return
        elif event.key in ("shift+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            self._auto_resize()
            return
        await super()._on_key(event)

    def _on_text_area_changed(self) -> None:
        """Auto-resize and update suggestion when text changes."""
        self._auto_resize()
        self._update_suggestion()

    def _auto_resize(self) -> None:
        """Resize height from 1 to 5 content lines, plus border overhead."""
        line_count = self.document.line_count
        content_height = max(1, min(5, line_count))
        self.styles.height = content_height + self._BORDER_OVERHEAD

    def _update_suggestion(self) -> None:
        """Show inline suggestion for /commands."""
        text = self.text
        self._suggestion = ""
        if text.startswith("/") and "\n" not in text:
            for cmd in self._commands:
                if cmd.startswith(text) and cmd != text:
                    self._suggestion = cmd
                    break

    def action_accept_suggestion(self) -> None:
        """Accept the current suggestion if one exists."""
        if self._suggestion:
            self.clear()
            self.insert(self._suggestion)
            self._suggestion = ""


class ZoracApp(App):
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

    Why a class instead of module-level functions?
      - Encapsulates mutable state (messages, connection status, config values)
      - Makes testing easier (create isolated instances with different state)
      - Command handlers can access shared state via self
      - Textual requires an App subclass for its lifecycle management

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
    # This mirrors common terminal conventions where Ctrl+C is interrupt
    # and Ctrl+D is EOF/exit.
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
        # This pattern replaces a long if/elif chain with a clean dictionary lookup.
        # Adding a new command is as simple as:
        #   1. Add the handler method (async def cmd_foo)
        #   2. Add the mapping here: "/foo": self.cmd_foo
        #   3. Add the command info to commands.py COMMANDS list
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
        """Build the UI layout: chat log, input bar, stats bar.

        Textual calls compose() once during app initialization to build the
        widget tree. The layout is:
          - VerticalScroll#chat-log: Scrollable container for the entire conversation.
            New messages are mounted as child widgets (Static for user/system,
            Markdown for assistant responses).
          - Vertical#bottom-bar: Docked to the bottom, containing:
            - ChatInput#user-input: Multiline text entry with command suggestions.
              Enter submits, Shift+Enter inserts newline. Auto-resizes 1-5 lines.
            - Static#stats-bar: Shows "Ready", session info, or performance metrics.

        The bottom-bar is docked (CSS: dock: bottom) so it stays pinned even
        as the chat log grows and scrolls.
        """
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

        This is separated from __init__ because it performs async operations
        (connection check) and I/O (session loading). Python's __init__ can't
        be async, so Textual's on_mount() calls this method after the UI is ready.

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
        # maintained client for free. The client handles HTTP connection pooling,
        # retries, streaming, and error handling.
        self.client = AsyncOpenAI(base_url=self.vllm_base_url, api_key=self.vllm_api_key)
        self.is_connected = await self._check_connection()

        if not self.is_connected:
            self._log_system(
                "Proceeding offline (some features may not work)...",
                style="bold yellow",
            )

        self.session_start_time = time.time()

        # Try to restore the previous conversation session from disk.
        # If successful, the user can continue where they left off.
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
        # (or earlier). This ensures the LLM always knows today's date.
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
        # Falls back to cl100k_base if the configured encoding is invalid.
        try:
            self.encoding = tiktoken.get_encoding(self.tiktoken_encoding)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    # -------------------------------------------------------------------
    # UI Helpers
    # -------------------------------------------------------------------

    def _log_system(self, text: str, style: str = "dim") -> None:
        """Add a system message to the chat log.

        System messages are status updates, warnings, and command output —
        anything that isn't a user message or assistant response. The style
        parameter accepts any Rich markup style (e.g., "green", "bold yellow",
        "red") for contextual color coding.

        Messages are mounted as Static widgets in the chat log and automatically
        scrolled into view. Unlike console.print() in the old architecture,
        this integrates seamlessly with Textual's widget tree.
        """
        chat_log = self.query_one("#chat-log", VerticalScroll)
        widget = Static(f"[{style}]{text}[/{style}]")
        chat_log.mount(widget)
        widget.scroll_visible()

    def _log_user(self, text: str) -> None:
        """Add a user message to the chat log.

        Displays the user's message with a blue "You:" prefix to visually
        distinguish it from assistant output (purple) and system messages (dim).
        For multiline messages, indents continuation lines for readability.
        """
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
        """Update the bottom stats bar with contextual information.

        The stats bar shows different content depending on the application state:
          - After a chat: Full performance metrics (tokens, duration, tok/s) and
            conversation totals (message count, token usage vs limit)
          - With a loaded session (no recent chat): Session summary (msg count, tokens)
          - Fresh start: "Ready"

        This replaces the prompt_toolkit bottom toolbar from the old architecture.
        The stats bar is a Static widget docked to the bottom, so it persists
        across all interactions without cluttering the scrollback.
        """
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
        """Write the welcome header with ASCII art logo and connection info.

        This is displayed once at startup (and again after /clear). The header
        serves dual purposes:
          1. Brand identity — the ASCII art logo makes Zorac visually distinctive
          2. Quick reference — shows version, connection info, and available commands

        The ASCII art uses Unicode box-drawing characters (█, ╔, ║, etc.) styled
        with Rich markup for purple coloring. Unlike the old architecture which
        used Rich's Panel widget, the Textual version mounts Static widgets
        directly into the chat log.
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

        Uses the OpenAI-compatible /v1/models endpoint to verify the server is
        reachable and responding. This is a lightweight health check — the models
        list endpoint is fast and doesn't require loading the model into GPU memory.

        Unlike the standalone check_connection() in utils.py (which uses Rich's
        console.status spinner), this TUI-aware version updates the stats bar
        and logs messages to the chat log widget tree.

        Broad exception handling is intentional because connection failures can
        manifest as many different exception types: ConnectionError, TimeoutError,
        httpx.ConnectError, etc. We display the specific error for debugging.

        Returns:
            True if the server is reachable and responding, False otherwise.
        """
        assert self.client is not None
        stats_bar = self.query_one("#stats-bar", Static)
        stats_bar.update(" Verifying connection... ")
        try:
            # client.models.list() calls GET /v1/models on the vLLM server.
            # If this succeeds, the server is running and accepting requests.
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
        """Load command history from ~/.zorac/history for Up/Down arrow navigation.

        Reads the history file line by line, deduplicating consecutive entries.
        Handles migration from prompt_toolkit's format (which uses a "+" prefix
        on each line) by stripping the prefix. Multiline entries are stored with
        literal \\n escapes and unescaped on load.

        Silently ignores errors (missing file, permissions) — history is a
        convenience feature that shouldn't prevent the app from starting.
        """
        try:
            if HISTORY_FILE.exists():
                lines = Path(HISTORY_FILE).read_text().splitlines()
                for line in lines:
                    # Handle prompt_toolkit's + prefix format for migration
                    entry = line.lstrip("+").strip()
                    # Unescape multiline entries
                    entry = entry.replace("\\n", "\n")
                    if entry and (not self._history or self._history[-1] != entry):
                        self._history.append(entry)
        except Exception:
            pass

    def _save_history(self) -> None:
        """Save command history to the history file.

        Keeps only the last 500 entries to prevent unbounded growth.
        Called after every input submission so history persists across sessions.
        Multiline entries are escaped (\\n) so each history entry stays on one line.
        Silently ignores errors to avoid disrupting the chat experience.
        """
        try:
            ensure_zorac_dir()
            escaped = [entry.replace("\n", "\\n") for entry in self._history[-500:]]
            Path(HISTORY_FILE).write_text("\n".join(escaped) + "\n")
        except Exception:
            pass

    # -------------------------------------------------------------------
    # Input Handling
    # -------------------------------------------------------------------

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle user input when Enter is pressed.

        Textual calls this automatically when ChatInput posts a Submitted
        message. This replaces the old prompt_toolkit REPL loop — instead of
        looping with prompt_async(), Textual's event system dispatches input
        submissions as messages.

        The routing logic is the same as the old architecture:
          - Commands (starting with /) are dispatched via the command_handlers table
          - Normal text goes to handle_chat() for LLM interaction
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
                # Dispatch to the appropriate command handler.
                # We pass `parts` so handlers can access arguments
                # (e.g., /config set KEY VALUE → parts = ["/config", "set", "KEY", "VALUE"])
                await self.command_handlers[cmd](parts)
            else:
                self._log_system(
                    f"Unknown command: {cmd}. Type /help for available commands.",
                    style="red",
                )
        else:
            # Not a command — process as a regular chat message
            await self.handle_chat(user_input)

    async def on_key(self, event) -> None:
        """Handle Up/Down arrow keys for command history navigation.

        Implements readline-like history navigation:
          - Up arrow: Move backward through history (older entries)
          - Down arrow: Move forward through history (newer entries)

        For multiline content, Up/Down move the cursor within the text
        instead of navigating history. History navigation only activates
        when the cursor is on the first line (Up) or last line (Down).

        When the user first presses Up, we save their current input in
        _history_temp so they can return to it by pressing Down past the
        most recent history entry. This prevents losing partially-typed
        messages when browsing history.
        """
        input_widget = self.query_one("#user-input", ChatInput)
        if not input_widget.has_focus:
            return

        if event.key == "up":
            # Only navigate history when cursor is on the first line
            if not input_widget.cursor_at_first_line:
                return
            event.prevent_default()
            if not self._history:
                return
            if self._history_index == -1:
                # Starting to navigate — save current input
                self._history_temp = input_widget.text
                self._history_index = len(self._history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            else:
                return  # Already at the oldest entry
            input_widget.clear()
            input_widget.insert(self._history[self._history_index])
            input_widget._auto_resize()
            # Move cursor to end of text
            last_line = input_widget.document.line_count - 1
            last_col = len(input_widget.document.get_line(last_line))
            input_widget.move_cursor((last_line, last_col))

        elif event.key == "down":
            # Only navigate history when cursor is on the last line
            if not input_widget.cursor_at_last_line:
                return
            event.prevent_default()
            if self._history_index == -1:
                return  # Not currently navigating history
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                text = self._history[self._history_index]
            else:
                # Past the newest entry — restore the saved input
                self._history_index = -1
                text = self._history_temp
            input_widget.clear()
            input_widget.insert(text)
            input_widget._auto_resize()
            # Move cursor to end of text
            last_line = input_widget.document.line_count - 1
            last_col = len(input_widget.document.get_line(last_line))
            input_widget.move_cursor((last_line, last_col))

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

        The actual streaming happens in _stream_response(), which runs as a
        Textual worker thread to avoid blocking the UI event loop. The input
        widget is disabled during streaming to prevent overlapping requests.

        Args:
            user_input: The user's chat message (already confirmed to not be a command).
        """
        assert self.client is not None

        # Keep system message date current for long-running sessions (across midnight)
        self._refresh_system_date()

        # Lazy connection check: if we're offline, try to reconnect before chatting.
        # This handles the case where the server was started after Zorac.
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
        # This happens before the API call to ensure we don't send too many
        # tokens to the model (which would cause an error or truncation).
        current_tokens = count_tokens(self.messages)
        if current_tokens > MAX_INPUT_TOKENS:
            await self._summarize_messages()

        # Disable input during streaming to prevent overlapping requests
        input_widget = self.query_one("#user-input", ChatInput)
        input_widget.disabled = True
        self._streaming = True

        self._stream_response()

    @work(exclusive=True, group="stream")
    async def _stream_response(self) -> None:
        """Stream the LLM response with live Markdown rendering.

        This method runs as a Textual worker thread (via @work decorator) so
        streaming doesn't block the UI event loop. The exclusive=True parameter
        ensures only one stream runs at a time, and group="stream" allows
        action_cancel_stream() to cancel it by group name.

        The streaming implementation uses Textual's Markdown.get_stream() API,
        which provides a write-based interface for incrementally building
        Markdown content. As tokens arrive from the LLM, they're written to
        the stream and the Markdown widget re-renders automatically. This is
        simpler than the old Rich Live approach and handles scrolling natively.

        Performance stats (tokens/second, response time) are calculated locally
        using tiktoken. This gives approximate but useful metrics without
        needing the server to report them.
        """
        assert self.client is not None
        from textual.worker import get_current_worker

        worker = get_current_worker()
        chat_log = self.query_one("#chat-log", VerticalScroll)
        stats_bar = self.query_one("#stats-bar", Static)
        input_widget = self.query_one("#user-input", ChatInput)

        # Add assistant label to the chat log
        label = Static("\n[bold purple]Assistant:[/bold purple]")
        chat_log.mount(label)
        label.scroll_visible()

        # Show "Thinking..." while waiting for the first token.
        # This provides immediate visual feedback that the request was sent.
        stats_bar.update(" Thinking... ")

        full_content = ""
        start_time = time.time()

        try:
            if self.stream_enabled:
                # --- Streaming mode (default) ---
                # Streaming provides a responsive experience: tokens appear
                # as they're generated (60-65 tok/s on RTX 4090), rather than
                # waiting for the entire response to complete.
                stream_response = await self.client.chat.completions.create(
                    model=self.vllm_model,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    stream=True,
                )

                # Create a Markdown widget and get a streaming handle for it.
                # Markdown.get_stream() returns an async context that accepts
                # write() calls to incrementally build the content. The widget
                # re-renders as content arrives, handling code blocks, lists,
                # headings, etc. as they complete.
                md_widget = Markdown("")
                chat_log.mount(md_widget)
                md_widget.scroll_visible()

                stream = Markdown.get_stream(md_widget)
                stream_tokens = 0

                async for chunk in stream_response:
                    # Check for cancellation (Ctrl+C) on each chunk
                    if worker.is_cancelled:
                        break

                    if chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        full_content += content_chunk
                        stream_tokens += len(self.encoding.encode(content_chunk))
                        elapsed = time.time() - start_time
                        tps = stream_tokens / elapsed if elapsed > 0 else 0

                        # Write the chunk to the Markdown stream and keep
                        # the chat log scrolled to the bottom
                        await stream.write(content_chunk)
                        chat_log.scroll_end(animate=False)

                        # Update the stats bar with real-time performance metrics
                        stats_bar.update(
                            f" {stream_tokens} tokens | {elapsed:.1f}s | {tps:.1f} tok/s "
                        )

                # Finalize the stream so the Markdown widget renders completely
                await stream.stop()
                chat_log.scroll_end()

            else:
                # --- Non-streaming mode ---
                # Waits for the complete response before displaying.
                # Useful for debugging or when streaming causes issues.
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
            # Only show errors if the stream wasn't intentionally cancelled
            if not worker.is_cancelled:
                self._log_system(f"Error receiving response: {e}", style="red")
                full_content += f"\n[Error: {e}]"

        # --- Performance metrics ---
        # Calculate stats and store them for display in the stats bar.
        # The stats bar persists across interactions, providing seamless
        # stats visibility without cluttering the scrollback.
        end_time = time.time()
        duration = end_time - start_time
        tokens = len(self.encoding.encode(full_content)) if full_content else 0
        tps = tokens / duration if duration > 0 else 0

        # Persist the assistant's response and save the session to disk.
        # Auto-saving after every response ensures no conversation is lost,
        # even if the application crashes or the terminal is closed.
        if full_content and not worker.is_cancelled:
            self.messages.append({"role": "assistant", "content": full_content})
            save_session(self.messages)

        # Update stats dict — shown by _update_stats_bar()
        current_tokens = count_tokens(self.messages)
        self.stats = {
            "tokens": tokens,
            "duration": duration,
            "tps": tps,
            "total_msgs": len(self.messages),
            "current_tokens": current_tokens,
        }
        self._update_stats_bar()

        # Re-enable input so the user can type their next message
        self._streaming = False
        input_widget.disabled = False
        input_widget.focus()

    # -------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------

    def action_cancel_stream(self) -> None:
        """Handle Ctrl+C — cancel the current streaming response.

        If a response is being streamed, cancels the worker group ("stream")
        which stops the async iteration over chunks. The input is re-enabled
        so the user can continue chatting. The partial response is NOT saved
        to the session (to avoid incomplete messages in history).

        If not streaming, Ctrl+C is silently ignored — this prevents
        accidental exits and matches the behavior of most REPL-style apps
        where Ctrl+C interrupts the current operation but doesn't quit.
        """
        if self._streaming:
            self.workers.cancel_group(self, "stream")
            self._streaming = False
            input_widget = self.query_one("#user-input", ChatInput)
            input_widget.disabled = False
            input_widget.focus()
            self._log_system("Response interrupted.", style="yellow")
        # If not streaming, ignore Ctrl+C (don't quit)

    def action_quit_app(self) -> None:
        """Handle Ctrl+D — save the session and exit the application.

        Ctrl+D (EOF) is the standard Unix signal for "end of input". We save
        the session before exiting so the user can resume where they left off.
        """
        save_session(self.messages)
        self.exit()

    # -------------------------------------------------------------------
    # System message management
    # -------------------------------------------------------------------

    def _update_system_message(self):
        """Update the system message with today's date.

        The system message (messages[0]) includes the current date so the LLM
        can answer time-sensitive questions. This method refreshes it when:
          - Loading a session from a previous day
          - A session spans midnight (detected by _refresh_system_date)
        """
        self._current_date = datetime.date.today()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": get_initial_system_message()}

    def _refresh_system_date(self):
        """Check if the date has changed and update the system message if needed.

        Called before each chat interaction to handle long-running sessions
        that span midnight. Without this, a session started at 11pm would
        still tell the LLM it's "yesterday" at 1am.
        """
        today = datetime.date.today()
        if today != self._current_date:
            self._update_system_message()

    # -------------------------------------------------------------------
    # Summarization
    # -------------------------------------------------------------------

    async def _summarize_messages(self, auto: bool = True) -> None:
        """Summarize messages with TUI-appropriate feedback.

        This is a TUI wrapper around summarize_old_messages() from llm.py.
        It handles the UI aspects (status messages, stats bar updates) while
        delegating the actual summarization logic to the LLM module.

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

    # -------------------------------------------------------------------
    # Command Handlers
    # -------------------------------------------------------------------
    # Each handler is an async method that receives the parsed command parts
    # as a list of strings. For example, "/config set KEY VALUE" arrives as
    # ["/config", "set", "KEY", "VALUE"]. The _args parameter is prefixed
    # with _ for handlers that don't use arguments (convention for unused params).

    async def cmd_quit(self, _args: list[str]):
        """Handle /quit and /exit — save session and exit the application."""
        save_session(self.messages)
        self._log_system("Session saved. Goodbye!", style="green")
        self.exit()

    async def cmd_help(self, _args: list[str]):
        """Handle /help — display all available commands."""
        self._log_system(get_help_text())

    async def cmd_clear(self, _args: list[str]):
        """Handle /clear — reset conversation to a fresh state.

        Replaces the entire message history with just the system message,
        effectively starting a new conversation. The cleared state is
        immediately saved to disk so it persists across restarts. The chat
        log is also visually cleared and the header is re-displayed.
        """
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
        """Handle /save — manually save the current session to disk.

        Sessions are auto-saved after every assistant response, but this
        command allows explicit saves after typing a long message or before
        making changes you might want to revert with /load.
        """
        if save_session(self.messages):
            self._log_system(f"Session saved to {SESSION_FILE}", style="green")

    async def cmd_load(self, _args: list[str]):
        """Handle /load — reload the session from disk.

        Discards any in-memory changes and restores the last saved state.
        Useful for reverting to a previous point in the conversation.
        """
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
        """Handle /tokens — display current token usage statistics.

        Shows how much of the context window is being used, helping users
        understand when auto-summarization will trigger and how much room
        remains for new messages.
        """
        token_count = count_tokens(self.messages)
        self._log_system(
            f"Token usage:\n"
            f"   Current: ~{token_count} tokens\n"
            f"   Limit: {MAX_INPUT_TOKENS} tokens\n"
            f"   Remaining: ~{MAX_INPUT_TOKENS - token_count} tokens\n"
            f"   Messages: {len(self.messages)}"
        )

    async def cmd_summarize(self, _args: list[str]):
        """Handle /summarize — force conversation summarization.

        Unlike auto-summarization (triggered by token limit), this allows
        users to manually condense their conversation at any time. Useful
        for cleaning up long conversations even before hitting the limit.
        """
        assert self.client is not None
        if len(self.messages) <= KEEP_RECENT_MESSAGES + 1:
            self._log_system(
                f"Not enough messages to summarize. Need more than "
                f"{KEEP_RECENT_MESSAGES + 1} messages.",
                style="yellow",
            )
        else:
            # auto=False tells the summarizer to show "as requested"
            # instead of "token limit approaching"
            await self._summarize_messages(auto=False)
            save_session(self.messages)
            self._log_system("Session saved with summary", style="green")

    async def cmd_summary(self, _args: list[str]):
        """Handle /summary — display the current conversation summary if one exists.

        The summary message (if present) is always at index 1 in the messages
        list, right after the system message. It's identified by the
        "Previous conversation summary:" prefix that summarize_old_messages
        adds when creating it.
        """
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
        """Handle /reconnect — retry connection to the vLLM server.

        Useful when the server was started after Zorac, or after a network
        interruption. Simply re-runs the connection check.
        """
        assert self.client is not None
        self.is_connected = await self._check_connection()
        if not self.is_connected:
            self._log_system("Connection failed. Still offline.", style="bold yellow")

    async def cmd_config(self, args: list[str]):
        """Handle /config — manage configuration settings at runtime.

        Supports three subcommands:
          /config list           — Show all current settings
          /config set KEY VALUE  — Update a setting (persisted to config file)
          /config get KEY        — Show a specific setting's value

        When a setting is changed via /config set, we:
          1. Validate the value (type checking for numeric fields, encoding validation)
          2. Save to the config file (persists across restarts)
          3. Update the in-memory value on the ZoracApp instance (takes effect immediately)
          4. For some settings (like VLLM_BASE_URL), also update the client object

        Settings are stored as strings in the config file but converted to
        their appropriate types when loaded into the ZoracApp instance.
        """
        assert self.client is not None

        # --- /config list: Show all current configuration values ---
        if len(args) == 1 or args[1] == "list":
            # Mask the API key for security — show only first 4 chars
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
            # --- /config set KEY VALUE: Update a configuration setting ---
            key = args[2].upper()
            # Join remaining args as value to support values with spaces
            value = " ".join(args[3:])

            if key not in DEFAULT_CONFIG:
                self._log_system(
                    f"Unknown setting: {key}\nAvailable keys: {', '.join(DEFAULT_CONFIG.keys())}",
                    style="red",
                )
            else:
                # Validate the value before saving. Each setting type has its own
                # validation rules to prevent invalid configurations.
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

                # Save to config file and update in-memory state
                config = load_config()
                config[key] = value
                if save_config(config):
                    self._log_system(f"Updated {key} in {CONFIG_FILE}", style="green")

                    # Apply the change to the running instance immediately.
                    # Each setting has its own update logic because some require
                    # updating the client object, others need type conversion, etc.
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
            # --- /config get KEY: Show a specific setting's current value ---
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
            # --- Invalid /config usage: show help ---
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
    Textual's alternate screen. Once setup is complete (or skipped for
    returning users), we launch the Textual app.

    contextlib.suppress(KeyboardInterrupt) catches the final Ctrl+C that
    might occur during shutdown, preventing an ugly traceback when the
    user exits.

    This function is referenced in pyproject.toml as the console script
    entry point: [project.scripts] zorac = "zorac.main:main"
    """
    # First-time setup: guide new users through initial configuration.
    # This must happen before Textual takes over the terminal because
    # the setup wizard uses standard input() prompts.
    if is_first_run():
        from .utils import print_header

        print_header()
        run_first_time_setup()

    app = ZoracApp()
    with contextlib.suppress(KeyboardInterrupt):
        app.run()


if __name__ == "__main__":
    main()
