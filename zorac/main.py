"""
Main application module for Zorac — the interactive CLI chat client.

This is the heart of Zorac. It orchestrates all other modules into a cohesive
interactive experience. The main components are:

  1. **Zorac class**: The application controller that manages state, handles
     user input, processes commands, and coordinates chat interactions.

  2. **SlashCommandCompleter**: A prompt_toolkit completer that provides
     tab-completion for slash commands.

  3. **main()**: The entry point that creates and runs the Zorac instance.

Architecture: Async-first design
  The entire application is built on Python's asyncio. This is essential because:
    - The OpenAI client library (AsyncOpenAI) uses async HTTP calls
    - Streaming responses need non-blocking I/O to render in real-time
    - prompt_toolkit's prompt_async() prevents input from blocking the event loop
    - Status spinners and live updates require the event loop to keep running

  The async architecture follows this flow:
    main() → asyncio.run(app.run()) → event loop → prompt_async() / API calls

Command processing:
  Commands (starting with /) are handled by a dispatch table (self.command_handlers)
  that maps command strings to async handler methods. This pattern is cleaner than
  a long if/elif chain and makes it easy to add new commands.

  Normal chat messages (not starting with /) go through handle_chat(), which:
    1. Checks/manages the connection
    2. Appends the user message to history
    3. Checks token limits and triggers summarization if needed
    4. Streams the LLM response with live Markdown rendering
    5. Saves the session and displays performance stats
"""

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
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PtStyle
from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

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


class SlashCommandCompleter(Completer):
    """Tab-completion provider for slash commands.

    Integrates with prompt_toolkit's completion system to provide tab-completion
    for Zorac's interactive commands. Only activates when the user's input starts
    with "/" — typing regular chat text won't trigger command suggestions.

    prompt_toolkit calls get_completions() on every keystroke (when completion
    is active). We filter the command list against what the user has typed so far
    and yield matching commands.

    Example:
        User types "/su" → suggests "/summarize", "/summary"
        User types "hello" → no suggestions (not a command)
    """

    def __init__(self, commands: list[str]):
        # Sort commands alphabetically for consistent ordering in the completion menu
        self.commands = sorted(commands)

    def get_completions(self, document, _complete_event):
        """Yield matching command completions for the current input.

        Args:
            document: prompt_toolkit Document with cursor position and text.
            _complete_event: Event metadata (unused — we always complete synchronously).

        Yields:
            Completion objects for commands matching the current input.
        """
        text = document.text_before_cursor.lstrip()

        # Only suggest completions when the user is typing a command
        if not text.startswith("/"):
            return

        for command in self.commands:
            # Case-insensitive matching for user convenience
            if command.lower().startswith(text.lower()):
                # start_position=-len(text) tells prompt_toolkit how many characters
                # to replace. This ensures the completion replaces the partial command
                # the user typed (e.g., "/su" → "/summarize" replaces 3 chars).
                yield Completion(command, start_position=-len(text))


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


class Zorac:
    """Main application class for the Zorac CLI chat client.

    This class follows the "application controller" pattern — it owns all
    application state and coordinates between the various subsystems (config,
    session, LLM, UI). The lifecycle is:

        app = Zorac()       # Create instance with default state
        await app.setup()   # Initialize: load config, connect, restore session
        await app.run()     # Enter the interactive loop

    Why a class instead of module-level functions?
      - Encapsulates mutable state (messages, connection status, config values)
      - Makes testing easier (create isolated instances with different state)
      - Command handlers can access shared state via self
      - Clean lifecycle management (setup → run → quit)

    Key state:
      - self.messages: The conversation history (list of OpenAI message dicts)
      - self.client: The async OpenAI client connected to the vLLM server
      - self.is_connected: Whether we have a working server connection
      - self.running: Controls the main loop (set to False by /quit)
    """

    def __init__(self):
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
        self.client: AsyncOpenAI | None = None  # Set during setup()
        self.is_connected = False  # Updated by check_connection()
        self.messages: list[ChatCompletionMessageParam] = []  # Conversation history
        self.prompt_session: PromptSession | None = None  # prompt_toolkit session
        self.running = True  # Main loop control flag
        self.session_start_time: float = 0.0  # For calculating session duration
        self._current_date: datetime.date | None = None  # Tracks date for system message refresh

        # Stats from the most recent chat interaction, displayed in the bottom toolbar
        self.stats: dict[str, int | float] = {
            "tokens": 0,
            "duration": 0.0,
            "tps": 0.0,
            "total_msgs": 0,
            "current_tokens": 0,
        }

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

    async def setup(self):
        """Initialize the application: display header, load config, connect to server.

        This is separated from __init__ because it performs async operations
        (connection check) and I/O (first-time setup wizard, session loading).
        Python's __init__ can't be async, so we use a separate setup() method.

        Setup sequence:
          1. Display welcome header (ASCII art + connection info)
          2. Run first-time setup wizard if no config file exists
          3. Load configuration from all sources (env, file, defaults)
          4. Ensure data directory exists
          5. Set up prompt_toolkit for interactive input
          6. Create OpenAI client and verify connection
          7. Load previous session or create a fresh one
          8. Update system message with current date
        """
        print_header()

        # First-time setup: guide new users through initial configuration
        if is_first_run():
            run_first_time_setup()

        self.load_configuration()
        ensure_zorac_dir()
        self.setup_prompt_session()

        # Create the async OpenAI client. We use OpenAI's client library because
        # vLLM implements the OpenAI-compatible API, so we get a well-tested,
        # maintained client for free. The client handles HTTP connection pooling,
        # retries, streaming, and error handling.
        self.client = AsyncOpenAI(base_url=self.vllm_base_url, api_key=self.vllm_api_key)
        self.is_connected = await check_connection(self.client)

        if not self.is_connected:
            console.print(
                "\n[bold yellow]Proceeding offline (some features may not work)...[/bold yellow]"
            )

        self.session_start_time = time.time()

        # Try to restore the previous conversation session from disk.
        # If successful, the user can continue where they left off.
        loaded_messages = load_session()
        if loaded_messages:
            self.messages = loaded_messages
            token_count = count_tokens(self.messages)
            console.print(
                f"[green]✓ Loaded previous session ({len(self.messages)} messages, ~{token_count} tokens)[/green]"
            )
        else:
            # First run or no saved session — start with just the system message
            self.messages = [{"role": "system", "content": get_initial_system_message()}]

        # Update the system message's date in case the session was saved yesterday
        # (or earlier). This ensures the LLM always knows today's date.
        self._update_system_message()

    def load_configuration(self):
        """Load all settings from the configuration system.

        Reads each setting using the three-tier priority system
        (env var > config file > default). This method is called during
        setup() and could be called again to reload configuration.
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

    def setup_prompt_session(self):
        """Set up prompt_toolkit session with persistent history, tab-completion, and styling.

        prompt_toolkit provides a readline-like experience with extra features:
          - FileHistory: Saves command history to ~/.zorac/history, persisting
            across sessions. Users can press up/down to recall previous inputs.
          - SlashCommandCompleter: Tab-completion for /commands.
          - PromptSession: Manages the input state, including multi-line editing.
          - PtStyle: Dark-themed input bar with styled prompt and bottom toolbar.
          - placeholder: Hint text shown when input is empty.
        """
        # Collect all command triggers for the tab-completion list
        all_triggers = list(self.command_handlers.keys())

        pt_style = PtStyle.from_dict(
            {
                "prompt": "bold ansiblue",
                "": "bg:#1a1a2e",
                "placeholder": "italic #666666 bg:#1a1a2e",
                "bottom-toolbar": "#888888 bg:#0f0f1a",
                "bottom-toolbar.text": "#888888 bg:#0f0f1a",
            }
        )

        self.prompt_session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            completer=SlashCommandCompleter(all_triggers),
            style=pt_style,
            placeholder=HTML("<placeholder>Type your message or /&lt;command&gt;</placeholder>"),
        )

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

    def _get_stats_toolbar(self) -> list[tuple[str, str]]:
        """Build the bottom toolbar content showing stats or ready state.

        Returns prompt_toolkit style tuples for the bottom toolbar. Before any
        chat interaction, shows "Ready" or loaded session info. After a chat,
        shows response stats and conversation totals.
        """
        if self.stats["tokens"] > 0:
            return [
                (
                    "class:bottom-toolbar",
                    f" Stats: {int(self.stats['tokens'])} tokens in "
                    f"{self.stats['duration']:.1f}s ({self.stats['tps']:.1f} tok/s)"
                    f" | Total: {int(self.stats['total_msgs'])} msgs, "
                    f"~{int(self.stats['current_tokens'])}/{MAX_INPUT_TOKENS} tokens ",
                )
            ]

        # Before any chat — show session info if messages loaded, otherwise "Ready"
        if len(self.messages) > 1:
            token_count = count_tokens(self.messages)
            return [
                (
                    "class:bottom-toolbar",
                    f" Session: {len(self.messages)} msgs, ~{token_count} tokens ",
                )
            ]

        return [("class:bottom-toolbar", " Ready ")]

    async def get_multiline_input(self) -> str:
        """Get user input using prompt_toolkit's async prompt.

        Uses prompt_async() instead of the synchronous prompt() to avoid
        blocking the asyncio event loop. This is important because:
          - Rich's Live display and status spinners run on the event loop
          - A blocking prompt would freeze all async operations
          - Other async tasks (if any) can run while waiting for input

        The prompt is styled with blue bold "You: " to visually distinguish
        user input from assistant output (which uses purple).

        Returns:
            The user's input string, stripped of leading/trailing whitespace.
            Empty string if the user presses Ctrl+D (EOF) or Ctrl+C.
        """
        assert self.prompt_session is not None, "Prompt session not initialized."

        # FormattedText uses ANSI style tuples: (style_string, text)
        # This is prompt_toolkit's styling format (different from Rich's markup).
        formatted_prompt = FormattedText([("class:prompt", "> ")])
        try:
            user_input = await self.prompt_session.prompt_async(
                formatted_prompt,
                bottom_toolbar=self._get_stats_toolbar,
            )
            return str(user_input).strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    async def run(self):
        """Main interactive loop — the core of the application.

        This is a standard REPL (Read-Eval-Print Loop):
          1. Read: Get user input via prompt_toolkit
          2. Eval: Process as command or chat message
          3. Print: Display results via Rich
          4. Loop: Repeat until self.running is False

        Error handling:
          - EOFError (Ctrl+D): Triggers graceful quit (save and exit)
          - KeyboardInterrupt (Ctrl+C): Warns user but continues the session.
            This allows interrupting a long response without losing the session.
          - Other exceptions: Displayed as error messages without crashing.
            The session continues so the user doesn't lose their conversation.

        patch_stdout() is needed because prompt_toolkit and Rich both write to
        stdout. Without it, Rich output during prompt_toolkit's input phase
        can corrupt the terminal display.
        """
        await self.setup()

        while self.running:
            try:
                # patch_stdout() ensures Rich's output doesn't interfere with
                # prompt_toolkit's terminal control during input collection.
                with patch_stdout():
                    print()
                    user_input = await self.get_multiline_input()

                if not user_input:
                    if not self.running:
                        break
                    continue

                # Route input: commands start with "/", everything else is chat
                if user_input.startswith("/"):
                    parts = user_input.split()
                    cmd = parts[0].lower()
                    if cmd in self.command_handlers:
                        # Dispatch to the appropriate command handler.
                        # We pass `parts` so handlers can access arguments
                        # (e.g., /config set KEY VALUE → parts = ["/config", "set", "KEY", "VALUE"])
                        await self.command_handlers[cmd](parts)
                        continue
                    else:
                        console.print(
                            f"\n[red]Unknown command: {cmd}. Type /help for available commands.[/red]\n"
                        )
                        continue

                # Not a command — process as a regular chat message
                await self.handle_chat(user_input)

            except EOFError:
                # Ctrl+D: graceful exit
                await self.cmd_quit([])
            except KeyboardInterrupt:
                # Ctrl+C: interrupt current operation but keep the session alive.
                # This is especially useful for interrupting long streaming responses.
                console.print(
                    "\n\n[yellow]Interrupted. Type /quit to exit or continue chatting.[/yellow]\n"
                )
            except Exception as e:
                # Catch-all: display the error but keep running. Crashing would
                # lose the user's unsaved conversation, which is worse than
                # showing an error message.
                console.print(f"\n[red]Error: {e}[/red]\n")

    async def handle_chat(self, user_input: str):
        """Process a standard chat interaction with the LLM.

        This is the core chat flow:
          1. Check/refresh server connection
          2. Append user message to conversation history
          3. Check token count → trigger auto-summarization if over limit
          4. Send messages to LLM and stream the response
          5. Render response as live-updating Markdown
          6. Save session and display performance stats

        The streaming implementation uses Rich's Live context manager for
        real-time Markdown rendering. As tokens arrive from the LLM, the
        accumulated text is re-rendered as Markdown and the Live display
        updates in place (no flickering or scrolling artifacts).

        Performance stats (tokens/second, response time) are calculated locally
        using tiktoken. This gives approximate but useful performance metrics
        without needing the server to report them.

        Args:
            user_input: The user's chat message (already confirmed to not be a command).
        """
        assert self.client is not None, "Client not initialized. Call setup() first."

        # Keep system message date current for long-running sessions (across midnight)
        self._refresh_system_date()

        # Lazy connection check: if we're offline, try to reconnect before chatting.
        # This handles the case where the server was started after Zorac.
        if not self.is_connected:
            console.print()
            self.is_connected = await check_connection(self.client)
            if not self.is_connected:
                console.print(
                    "\n[bold yellow]Still offline. Use /reconnect to retry or check your server.[/bold yellow]\n"
                )
                return

        # Add the user's message to the conversation history
        self.messages.append({"role": "user", "content": user_input})

        # Check if we've exceeded the token limit and need to summarize.
        # This happens before the API call to ensure we don't send too many
        # tokens to the model (which would cause an error or truncation).
        current_tokens = count_tokens(self.messages)
        if current_tokens > MAX_INPUT_TOKENS:
            self.messages = await summarize_old_messages(self.client, self.messages)

        start_time = time.time()
        console.print("\n[bold purple]Assistant:[/bold purple]")

        full_content = ""
        first_chunk_received = False

        # Show a "Thinking..." spinner while waiting for the first token.
        # This provides immediate visual feedback that the request was sent.
        # The spinner is stopped once the first chunk arrives.
        with console.status(
            "[bold purple]Thinking...[/bold purple]", spinner="dots", spinner_style="purple"
        ) as status:
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

                    # Rich's Live context manager provides flicker-free updates.
                    # refresh_per_second=10 balances smooth display with CPU usage.
                    # Higher values (like 30) would waste CPU re-rendering unchanged
                    # content. Lower values (like 2) would make updates feel choppy.
                    stream_tokens = 0
                    with Live("", refresh_per_second=10) as live:
                        async for chunk in stream_response:
                            if chunk.choices[0].delta.content:
                                if not first_chunk_received:
                                    # Stop the "Thinking..." spinner now that content is arriving
                                    status.stop()
                                    first_chunk_received = True

                                # Accumulate content and re-render as Markdown.
                                # We re-render the entire accumulated text (not just the
                                # new chunk) because Markdown formatting is context-sensitive:
                                # a new line might complete a code block, list, or heading
                                # that changes how earlier text should be rendered.
                                content_chunk = chunk.choices[0].delta.content
                                full_content += content_chunk
                                stream_tokens += len(self.encoding.encode(content_chunk))
                                elapsed = time.time() - start_time
                                tps = stream_tokens / elapsed if elapsed > 0 else 0

                                markdown_content = Markdown(
                                    full_content, justify="left", code_theme=self.code_theme
                                )
                                stats_text = Text(
                                    f" {stream_tokens} tokens | {elapsed:.1f}s | {tps:.1f} tok/s",
                                    style="dim",
                                )
                                live.update(Group(markdown_content, stats_text))

                        # Final update with just the markdown so scrollback is clean
                        # (the transient stats line doesn't persist).
                        if full_content:
                            final_md = Markdown(
                                full_content, justify="left", code_theme=self.code_theme
                            )
                            live.update(final_md)
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
                    status.stop()
                    full_content = completion_response.choices[0].message.content or ""
                    markdown_content = Markdown(
                        full_content, justify="left", code_theme=self.code_theme
                    )
                    console.print(markdown_content)

            except Exception as e:
                console.print(f"[red]Error receiving response: {e}[/red]")
                full_content += f"\n[Error: {e}]"

        # --- Performance metrics ---
        # Calculate stats and store them for display in the bottom toolbar.
        # The toolbar persists across the next input prompt, providing
        # seamless stats visibility without cluttering scrollback.
        end_time = time.time()
        duration = end_time - start_time
        tokens = len(self.encoding.encode(full_content))
        tps = tokens / duration if duration > 0 else 0

        # Persist the assistant's response and save the session to disk.
        # Auto-saving after every response ensures no conversation is lost,
        # even if the application crashes or the terminal is closed.
        self.messages.append({"role": "assistant", "content": full_content})
        save_session(self.messages)
        current_tokens = count_tokens(self.messages)

        # Update stats dict — shown by _get_stats_toolbar() on next prompt
        self.stats = {
            "tokens": tokens,
            "duration": duration,
            "tps": tps,
            "total_msgs": len(self.messages),
            "current_tokens": current_tokens,
        }

    # -----------------------------------------------------------------------
    # Command Handlers
    # -----------------------------------------------------------------------
    # Each handler is an async method that receives the parsed command parts
    # as a list of strings. For example, "/config set KEY VALUE" arrives as
    # ["config", "set", "KEY", "VALUE"]. The _args parameter is prefixed
    # with _ for handlers that don't use arguments (convention for unused params).

    async def cmd_quit(self, _args: list[str]):
        """Handle /quit and /exit — save session and exit the application."""
        save_session(self.messages)
        console.print("\n[green]✓ Session saved. Goodbye![/green]\n")
        self.running = False

    async def cmd_help(self, _args: list[str]):
        """Handle /help — display all available commands with descriptions."""
        console.print(f"\n{get_help_text()}\n")

    async def cmd_clear(self, _args: list[str]):
        """Handle /clear — reset conversation to a fresh state.

        Replaces the entire message history with just the system message,
        effectively starting a new conversation. The cleared state is
        immediately saved to disk so it persists across restarts.
        """
        self.messages = [{"role": "system", "content": get_initial_system_message()}]
        self._current_date = datetime.date.today()
        save_session(self.messages)
        console.print("\n[green]✓ Conversation history cleared and saved![/green]\n")

    async def cmd_save(self, _args: list[str]):
        """Handle /save — manually save the current session to disk.

        Sessions are auto-saved after every assistant response, but this
        command allows explicit saves after typing a long message or before
        making changes you might want to revert with /load.
        """
        if save_session(self.messages):
            console.print(f"\n[green]✓ Session saved to {SESSION_FILE}[/green]\n")

    async def cmd_load(self, _args: list[str]):
        """Handle /load — reload the session from disk.

        Discards any in-memory changes and restores the last saved state.
        Useful for reverting to a previous point in the conversation.
        """
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
        """Handle /tokens — display current token usage statistics.

        Shows how much of the context window is being used, helping users
        understand when auto-summarization will trigger and how much room
        remains for new messages.
        """
        token_count = count_tokens(self.messages)
        console.print("\n[bold] Token usage:[/bold]")
        console.print(f"   Current: ~{token_count} tokens")
        console.print(f"   Limit: {MAX_INPUT_TOKENS} tokens")
        console.print(f"   Remaining: ~{MAX_INPUT_TOKENS - token_count} tokens")
        console.print(f"   Messages: {len(self.messages)}\n")

    async def cmd_summarize(self, _args: list[str]):
        """Handle /summarize — force conversation summarization.

        Unlike auto-summarization (triggered by token limit), this allows
        users to manually condense their conversation at any time. Useful
        for cleaning up long conversations even before hitting the limit.
        """
        assert self.client is not None, "Client not initialized."
        if len(self.messages) <= KEEP_RECENT_MESSAGES + 1:
            console.print(
                f"\n[yellow]⚠ Not enough messages to summarize. Need more than {KEEP_RECENT_MESSAGES + 1} messages.[/yellow]\n"
            )
        else:
            # auto=False tells summarize_old_messages to show "as requested"
            # instead of "token limit approaching"
            self.messages = await summarize_old_messages(self.client, self.messages, auto=False)
            save_session(self.messages)
            console.print("[green]✓ Session saved with summary[/green]\n")

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
                console.print("\n[bold] Current Conversation Summary:[/bold]\n")
                markdown_content = Markdown(
                    summary_text, justify="left", code_theme=self.code_theme
                )
                console.print(Panel(markdown_content, box=box.ROUNDED, expand=True))
                console.print()
                summary_found = True

        if not summary_found:
            console.print(
                "\n[yellow]ℹ No summary exists yet. Use /summarize to create one.[/yellow]\n"
            )

    async def cmd_reconnect(self, _args: list[str]):
        """Handle /reconnect — retry connection to the vLLM server.

        Useful when the server was started after Zorac, or after a network
        interruption. Simply re-runs the connection check.
        """
        assert self.client is not None, "Client not initialized."
        self.is_connected = await check_connection(self.client)
        if not self.is_connected:
            console.print("\n[bold yellow]Connection failed. Still offline.[/bold yellow]\n")
        else:
            console.print()

    async def cmd_config(self, args: list[str]):
        """Handle /config — manage configuration settings at runtime.

        Supports three subcommands:
          /config list           — Show all current settings
          /config set KEY VALUE  — Update a setting (persisted to config file)
          /config get KEY        — Show a specific setting's value

        When a setting is changed via /config set, we:
          1. Validate the value (type checking for numeric fields, encoding validation)
          2. Save to the config file (persists across restarts)
          3. Update the in-memory value on the Zorac instance (takes effect immediately)
          4. For some settings (like VLLM_BASE_URL), also update the client object

        Settings are stored as strings in the config file but converted to
        their appropriate types when loaded into the Zorac instance.
        """
        assert self.client is not None, "Client not initialized."

        if len(args) == 1 or args[1] == "list":
            # --- /config list: Show all current configuration values ---
            console.print("\n[bold]Configuration:[/bold]")
            console.print(f"  VLLM_BASE_URL:      [cyan]{self.vllm_base_url}[/cyan]")
            console.print(f"  VLLM_MODEL:         [cyan]{self.vllm_model}[/cyan]")
            # Mask the API key for security — show only first 4 chars
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
            # --- /config set KEY VALUE: Update a configuration setting ---
            key = args[2].upper()
            # Join remaining args as value to support values with spaces
            value = " ".join(args[3:])

            if key not in DEFAULT_CONFIG:
                console.print(f"\n[red]Unknown setting: {key}[/red]")
                console.print(f"Available keys: {', '.join(DEFAULT_CONFIG.keys())}\n")
            else:
                # Validate the value before saving. Each setting type has its own
                # validation rules to prevent invalid configurations.
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

                # Save to config file and update in-memory state
                config = load_config()
                config[key] = value
                if save_config(config):
                    console.print(f"\n[green]✓ Updated {key} in {CONFIG_FILE}[/green]")

                    # Apply the change to the running instance immediately.
                    # Each setting has its own update logic because some require
                    # updating the client object, others need type conversion, etc.
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
                console.print(f"\n{key} = {settings_map[key]}\n")
            else:
                console.print(f"\n[red]Unknown setting: {key}[/red]\n")
                console.print(f"Available settings: {', '.join(settings_map.keys())}\n")
        else:
            # --- Invalid /config usage: show help ---
            console.print("\n[bold]Usage:[/bold]")
            console.print("  /config list               - Show current configuration")
            console.print("  /config set <KEY> <VALUE>  - set a configuration value")
            console.print("  /config get <KEY>          - Get a specific configuration value\n")


def main():
    """Application entry point — creates and runs the Zorac instance.

    Uses asyncio.run() to start the async event loop that drives the entire
    application. contextlib.suppress(KeyboardInterrupt) catches the final
    Ctrl+C that might occur during shutdown, preventing an ugly traceback
    when the user exits.

    This function is referenced in pyproject.toml as the console script
    entry point: [project.scripts] zorac = "zorac.main:main"
    """
    app = Zorac()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(app.run())


if __name__ == "__main__":
    main()
