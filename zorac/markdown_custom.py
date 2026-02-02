"""Custom Markdown renderer with left-aligned headings via monkey-patching"""

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Heading, Markdown
from rich.text import Text

# Store original method
_original_heading_rich_console = Heading.__rich_console__


def _left_aligned_heading_rich_console(
    self, _console: Console, _options: ConsoleOptions
) -> RenderResult:
    """Custom heading renderer that left-aligns headings"""
    text = self.text
    text.justify = "left"  # Force left alignment instead of center

    if self.tag == "h1":
        # H1 - no panel, just bold white text with spacing
        yield Text("")
        yield Text(text.plain, style="bold #ffffff")
    elif self.tag == "h2":
        # H2 - bold light gray text with spacing
        yield Text("")
        yield Text(text.plain, style="bold #cccccc")
    else:
        # H3+ - bold gray text
        yield Text(text.plain, style="bold #aaaaaa")


# Monkey-patch the Heading class to use left-aligned rendering
Heading.__rich_console__ = _left_aligned_heading_rich_console


# Simple wrapper class
class LeftAlignedMarkdown(Markdown):
    """Markdown renderer that uses monkey-patched left-aligned headings"""

    def __init__(self, markup: str, *args, **kwargs):
        """Initialize with left justification"""
        if "justify" not in kwargs:
            kwargs["justify"] = "left"
        super().__init__(markup, *args, **kwargs)
