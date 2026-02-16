"""
Command handlers for the Zorac TUI.

Provides CommandHandlersMixin with all cmd_* methods for interactive commands,
mixed into ZoracApp.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import tiktoken
from openai import AsyncOpenAI
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

from .commands import get_help_text, get_initial_system_message
from .config import (
    CONFIG_FILE,
    DEFAULT_CONFIG,
    KEEP_RECENT_MESSAGES,
    MAX_INPUT_TOKENS,
    SESSION_FILE,
    load_config,
    save_config,
)
from .session import load_session, save_session
from .utils import count_tokens

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class CommandHandlersMixin:
    """Mixin providing all cmd_* command handler methods."""

    # Type stubs for attributes provided by ZoracApp
    messages: list[ChatCompletionMessageParam]
    client: AsyncOpenAI | None
    is_connected: bool
    _current_date: datetime.date | None
    vllm_base_url: str
    vllm_model: str
    vllm_api_key: str
    temperature: float
    max_output_tokens: int
    stream_enabled: bool
    tiktoken_encoding: str
    code_theme: str
    encoding: Any
    stats: dict[str, int | float]

    # Method stubs for ZoracApp / App methods — only present during type
    # checking so they don't shadow real methods inherited via MRO at runtime.
    if TYPE_CHECKING:

        def _log_system(self, text: str, style: str = "dim") -> None: ...
        def _update_stats_bar(self) -> None: ...
        def _write_header(self, show_commands: bool = True) -> None: ...
        async def _check_connection(self) -> bool: ...
        async def _summarize_messages(self, auto: bool = True) -> None: ...
        def query_one(self, selector: str, expect_type: type | None = None) -> Any: ...
        def exit(
            self, result: object = None, return_code: int = 0, message: str | None = None
        ) -> None: ...

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
        self._write_header(show_commands=False)
        self._log_system("")
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
            f"   Messages: {len(self.messages)}\n"
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
                f"  Config File:        {CONFIG_FILE}\n"
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
                        self.client = AsyncOpenAI(
                            base_url=self.vllm_base_url, api_key=self.vllm_api_key
                        )
                        self._log_system(
                            "Base URL updated. Connection will be verified on next message.",
                            style="yellow",
                        )
                    elif key == "VLLM_MODEL":
                        self.vllm_model = value
                    elif key == "VLLM_API_KEY":
                        self.vllm_api_key = value
                        self.client = AsyncOpenAI(
                            base_url=self.vllm_base_url, api_key=self.vllm_api_key
                        )
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
