"""
Chat input widget for the Zorac TUI.

Provides a multiline text input with Enter to submit, Shift+Enter for newlines,
auto-resizing, and inline command suggestions accepted with Tab.
"""

from dataclasses import dataclass

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


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
