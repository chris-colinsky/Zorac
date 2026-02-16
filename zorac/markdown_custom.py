"""
Custom Markdown renderer with left-aligned headings via monkey-patching.

Problem:
  Rich's built-in Markdown renderer centers all headings inside bordered panels.
  While this looks nice for standalone documents, it's visually jarring in a chat
  interface where everything else (text, code blocks, lists) is left-aligned.
  Centered headings break the reading flow and feel out of place.

Solution:
  We monkey-patch Rich's Heading class to render headings as simple left-aligned
  bold text instead of centered panels. This gives us clean, readable headings
  that integrate naturally with the rest of the chat output.

What is monkey-patching?
  Monkey-patching means modifying a class or module at runtime by replacing its
  methods or attributes. Here, we replace Rich's `Heading.__rich_console__` method
  (which controls how headings are rendered) with our own version. This is a
  pragmatic solution when:
    - You need to customize behavior in a third-party library
    - The library doesn't provide a clean extension point for what you need
    - Subclassing alone isn't sufficient (Rich creates Heading instances internally
      during Markdown parsing, so we can't easily substitute our own subclass)

  Trade-offs of monkey-patching:
    - Pro: Simple, effective, no need to fork the library
    - Con: Fragile — if Rich changes Heading's internal API, this could break
    - Con: Affects all Heading instances globally (which is actually what we want here)

Heading style hierarchy:
  H1: Bold white text — most prominent, used for main section titles
  H2: Bold light gray — secondary sections
  H3+: Bold gray — subsections and lower

Why a wrapper class (LeftAlignedMarkdown)?
  Even though the monkey-patch affects all Heading instances globally, the wrapper
  class provides a clean API boundary. Consumers import LeftAlignedMarkdown instead
  of plain Markdown, making the customization explicit and self-documenting.
  It also sets justify="left" by default for consistency.
"""

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Heading, Markdown
from rich.text import Text

# Store reference to the original method before patching. This is good practice
# in case we ever need to restore the original behavior (e.g., for testing).
_original_heading_rich_console = Heading.__rich_console__


def _left_aligned_heading_rich_console(
    self, _console: Console, _options: ConsoleOptions
) -> RenderResult:
    """Custom heading renderer that replaces Rich's centered panel headings.

    Rich's `__rich_console__` protocol is how Rich renderables define their
    visual output. Any object with this method can be rendered by Rich's Console.
    The method is a generator that yields renderable objects (Text, Panel, etc.).

    Our implementation yields simple Text objects instead of Rich's default
    centered Panel, producing clean left-aligned headings.

    Args:
        self: The Heading instance being rendered (has .text and .tag attributes).
        _console: The Console doing the rendering (unused — prefixed with _ by convention).
        _options: Rendering options like width (unused).

    Yields:
        Text objects representing the heading with appropriate styling.
    """
    text = self.text
    text.justify = "left"  # Force left alignment instead of center

    if self.tag == "h1":
        # H1: Blank line for spacing, then bold white text.
        # White (#ffffff) provides maximum contrast for top-level headings.
        yield Text("")
        yield Text(text.plain, style="bold #ffffff")
    elif self.tag == "h2":
        # H2: Slightly subdued with light gray (#cccccc).
        # Still bold to stand out, but visually subordinate to H1.
        yield Text("")
        yield Text(text.plain, style="bold #cccccc")
    else:
        # H3 and below: Even more subdued gray (#aaaaaa).
        # No extra spacing — these are inline sub-headings.
        yield Text(text.plain, style="bold #aaaaaa")


# Apply the monkey-patch: replace Rich's heading renderer with our custom version.
# This takes effect globally — every Markdown instance created after this point
# will use left-aligned headings. The patch happens at import time, which is fine
# because this module is imported early in the application lifecycle (via main.py).
Heading.__rich_console__ = _left_aligned_heading_rich_console


class LeftAlignedMarkdown(Markdown):
    """Markdown renderer that uses monkey-patched left-aligned headings.

    This is a thin wrapper around Rich's Markdown class that sets left
    justification as the default. The actual heading customization is done
    by the monkey-patch above — this class exists primarily for:
      1. API clarity: `LeftAlignedMarkdown` makes the customization explicit
      2. Default justify="left": Ensures all content is left-aligned by default
      3. Future extensibility: A place to add more Markdown customizations

    Usage:
        from .markdown_custom import LeftAlignedMarkdown as Markdown
        md = Markdown("# Hello\\nSome text", code_theme="monokai")
        console.print(md)
    """

    def __init__(self, markup: str, *args, **kwargs):
        """Initialize with left justification as the default.

        Passes through all arguments to Rich's Markdown.__init__, which
        handles parsing and rendering. The only override is setting
        justify="left" if not explicitly provided by the caller.

        Args:
            markup: The markdown text to render.
            *args, **kwargs: Passed through to Rich's Markdown constructor.
                Key kwargs include:
                - code_theme (str): Pygments theme for syntax highlighting
                - justify (str): Text justification ("left", "center", "right")
        """
        if "justify" not in kwargs:
            kwargs["justify"] = "left"
        super().__init__(markup, *args, **kwargs)
