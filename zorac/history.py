"""
Command history management for the Zorac TUI.

Provides HistoryMixin with history load/save and Up/Down arrow navigation,
mixed into ZoracApp.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import HISTORY_FILE, ensure_zorac_dir


class HistoryMixin:
    """Mixin providing command history load/save and keyboard navigation."""

    # Type stubs for attributes provided by ZoracApp
    _history: list[str]
    _history_index: int
    _history_temp: str

    # Method stubs for App methods — only present during type
    # checking so they don't shadow real methods inherited via MRO at runtime.
    if TYPE_CHECKING:

        def query_one(self, selector: str, expect_type: type | None = None) -> Any: ...

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
        from .widgets import ChatInput

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
