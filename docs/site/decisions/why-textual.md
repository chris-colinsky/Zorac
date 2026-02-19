# ADR: Why Textual

An Architecture Decision Record explaining why Zorac uses Textual as its TUI framework, what alternatives were considered, and the trade-offs involved.

---

## Context

### The Need for a Rich Terminal UI

Zorac is an interactive chat client for local LLMs. The UI needs to support:

- **Real-time streaming** — Tokens must appear as they're generated (60+ tok/s)
- **Markdown rendering** — Code blocks, headings, bold, lists, links
- **Persistent layout** — Input bar and stats bar stay pinned while chat scrolls
- **Keyboard handling** — Enter to submit, Shift+Enter for newlines, Ctrl+C to cancel, Up/Down for history
- **Async operations** — Network calls, streaming, and UI updates must not block each other

A basic `print()`/`input()` loop can't deliver this experience. We need a framework that manages the terminal, handles events, and provides rich rendering.

### Requirements: Streaming, Markdown, Widgets

The specific requirements that drove the framework choice:

| Requirement | Why It Matters |
|-------------|---------------|
| Streaming Markdown display | Core UX — tokens render as formatted Markdown in real-time |
| Fixed layout with docking | Input bar must stay visible while chat scrolls |
| Multiline input with key bindings | Enter submits, Shift+Enter newlines, Tab autocomplete |
| Background workers for async I/O | Streaming can't block the UI thread |
| CSS-like styling | Consistent look without manual escape codes |
| Cross-terminal compatibility | Must work in iTerm2, kitty, WezTerm, macOS Terminal |

---

## Alternatives Considered

### prompt_toolkit

[prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/) is a mature library for building interactive command-line applications. It's the foundation of IPython and many popular CLI tools.

**What it offers:**

- Excellent input handling (multiline, completion, history, key bindings)
- Full-screen applications with layout system
- Syntax highlighting
- Well-documented and battle-tested

**Why it wasn't chosen:**

- **No built-in Markdown rendering** — Would need to integrate Rich for output formatting, creating two separate rendering systems
- **No streaming Markdown API** — Real-time token streaming with incremental Markdown parsing would require custom implementation
- **No widget system** — Layout is functional but primitive compared to Textual's widget model
- **Event loop complexity** — Mixing prompt_toolkit's event loop with async OpenAI streaming would require careful orchestration

Zorac originally used prompt_toolkit (with Rich for output). The migration to Textual was driven by the need for streaming Markdown — prompt_toolkit couldn't provide the real-time rendering experience that Textual's `Markdown.get_stream()` API delivers natively.

### Rich Live Display

[Rich](https://rich.readthedocs.io/) (by the same author as Textual) provides beautiful terminal output with Markdown, syntax highlighting, tables, and progress bars. Its `Live` display allows updating a renderable in place.

**What it offers:**

- Excellent Markdown rendering
- `Live()` context manager for in-place updates
- Same aesthetic as Textual (they share the Rich rendering engine)

**Why it wasn't chosen:**

- **Flickering with complex Markdown** — Rich's `Live` display refreshes the entire output on each update. With long Markdown content (code blocks, nested lists), this causes visible flickering during streaming.
- **No widget layout** — No concept of docked widgets, scrollable regions, or fixed-position elements. The input bar and stats bar would need to be manually positioned.
- **No input handling** — Rich is output-only. Input would still require prompt_toolkit or raw terminal handling.
- **No streaming Markdown API** — Would need to replace the entire Markdown content on each token, rather than incrementally appending.

Rich is still used by Zorac — the `Console` singleton for pre-TUI output, and Rich's rendering engine powers Textual's widgets internally.

### Blessed / Curses

[Blessed](https://blessed.readthedocs.io/) (a modern wrapper around curses) and raw curses provide low-level terminal control.

**What they offer:**

- Complete terminal control (positioning, colors, input)
- Maximum flexibility
- No dependencies (curses is in the Python standard library)

**Why they weren't chosen:**

- **No Markdown rendering** — Would need to build or integrate a Markdown parser
- **No widget abstraction** — Everything is manual: drawing, scrolling, layout, event handling
- **Significant development effort** — Building a scrollable chat log with streaming Markdown would be a project unto itself
- **Cross-terminal quirks** — Curses applications often break in modern terminal emulators without careful testing

Curses-based approaches make sense for applications that need pixel-level control. For a chat application, the development cost vastly outweighs the benefits.

### Web-Based (Gradio, Streamlit)

Web-based frameworks provide the richest UI capabilities with the least effort.

**What they offer:**

- Full HTML/CSS/JS rendering
- Rich Markdown with code highlighting
- Built-in streaming support (Gradio)
- Accessible via any browser

**Why they weren't chosen:**

- **Separate server process** — Requires running a web server in addition to the vLLM server
- **Browser dependency** — Adds a heavyweight dependency for a terminal-focused tool
- **Mismatch with the project philosophy** — Zorac is about running AI locally and staying lightweight. A web UI contradicts the terminal-native approach.
- **Overhead** — Python web server + browser rendering is heavier than a TUI that runs directly in the terminal

Gradio is an excellent choice for sharing demos or building multi-user interfaces. For a personal tool that lives in the terminal, it's unnecessarily complex.

---

## Decision: Textual

### Widget System and Layout

Textual provides a proper widget system with composition, containment, and layout:

```python
def compose(self) -> ComposeResult:
    yield VerticalScroll(id="chat-log")     # Scrollable chat area
    yield Vertical(
        ChatInput(id="user-input"),          # Multiline input
        Static(" Ready ", id="stats-bar"),   # Performance metrics
        id="bottom-bar",
    )
```

Widgets are composed declaratively (via `yield` in `compose()`) and styled with CSS. The framework handles mounting, unmounting, layout calculation, and painting — the same concerns a web framework handles, but for the terminal.

### Built-in Markdown Support

Textual includes a `Markdown` widget that renders formatted Markdown with syntax-highlighted code blocks, headings, lists, tables, and links. This is powered by Rich's rendering engine internally, but wrapped in a proper widget that integrates with Textual's layout and scrolling.

### Streaming API (Markdown.get_stream)

The single most important feature for Zorac. `Markdown.get_stream()` provides a write-based interface for incrementally building Markdown content:

```python
md_widget = Markdown("")
stream = Markdown.get_stream(md_widget)

async for chunk in llm_response:
    await stream.write(chunk)  # Widget re-renders incrementally

await stream.stop()
```

This API handles incremental parsing, partial code blocks, and progressive rendering — all the hard problems of streaming Markdown. No other framework offered this capability out of the box.

### CSS-Based Styling

Textual CSS (TCSS) provides a familiar styling model:

```css
#stats-bar {
    background: #0f0f1a;
    color: #888888;
    padding: 1 1;
}
```

This replaces manual Rich markup strings with a centralized stylesheet, making the UI easier to maintain and modify.

### Worker Threads for Async Operations

The `@work` decorator provides managed background threads that can safely interact with the widget tree:

```python
@work(exclusive=True, group="stream")
async def _stream_response(self) -> None:
    stats_bar.update(...)  # Thread-safe widget update
    chat_log.mount(...)    # Thread-safe widget mounting
```

This solves the threading problem cleanly — no manual locks, no thread-safety bugs, no complex event loop coordination.

---

## Trade-offs

### Learning Curve

Textual is a full TUI framework with its own widget model, CSS subset, message system, and lifecycle. The learning curve is steeper than a simple print/input loop or even prompt_toolkit.

Mitigation: Textual has excellent documentation and the concepts map well to web development (widgets ≈ components, TCSS ≈ CSS, messages ≈ events). Developers with React or similar experience adapt quickly.

### Terminal Compatibility

Textual works well in modern terminal emulators but has known quirks:

- **macOS Terminal.app** — Limited support for Kitty keyboard protocol (Shift+Enter indistinguishable from Enter)
- **Older terminals** — May not render all Unicode characters correctly
- **SSH sessions** — Generally works, but some features (like clipboard) may not function

Zorac documents terminal compatibility in its README and uses fallbacks where possible (e.g., `ctrl+j` as an alternative Shift+Enter signal for iTerm2).

### Shift+Enter Behavior Across Terminals

The most notable terminal compatibility issue. Zorac needs Enter to submit and Shift+Enter to insert a newline. But different terminals report Shift+Enter differently:

| Terminal | Shift+Enter Reported As | Zorac Handling |
|----------|------------------------|----------------|
| kitty | `shift+enter` | Native support |
| WezTerm | `shift+enter` | Native support |
| iTerm2 | `ctrl+j` | Handled as newline |
| macOS Terminal.app | `enter` | No distinction — Enter always submits |

Zorac handles both `shift+enter` and `ctrl+j` as newline insertion. In terminals that can't distinguish (Terminal.app), multiline input is still possible via pasting — the bracket paste mode preserves newlines in pasted text.

---

## Outcome

### Migration from prompt_toolkit

Zorac was originally built with prompt_toolkit for input and Rich for output. The migration to Textual:

- **Unified the rendering stack** — One framework for input, output, layout, and styling
- **Enabled streaming Markdown** — The primary motivation for the migration
- **Simplified the architecture** — Replaced manual screen management, cursor positioning, and refresh logic with Textual's widget model
- **Improved the UX** — Persistent stats bar, smooth scrolling, and cleaner layout

### What Worked Well

- **`Markdown.get_stream()`** — Exactly the API needed for LLM streaming. The streaming experience is smooth and handles all Markdown edge cases.
- **Worker threads** — Clean separation between UI and I/O. No thread-safety issues.
- **CSS styling** — Easy to adjust the look without touching logic code.
- **Widget composition** — Adding new UI elements (like the stats bar) was straightforward.
- **Mixin compatibility** — Textual's `App` class works well with Python's MRO and mixin pattern.

### What Could Be Better

- **Startup time** — Textual adds ~200ms to application startup compared to a bare terminal. Acceptable but noticeable.
- **Terminal.app support** — The Shift+Enter limitation on macOS Terminal.app is a recurring support question. Users are advised to use iTerm2 or kitty.
- **Custom widget complexity** — Building `ChatInput` (extending `TextArea`) required understanding Textual's internals for key handling and auto-resize. The framework could benefit from a chat-oriented input widget in its standard library.
- **Debugging** — Textual's alternate screen makes print-debugging harder. The `textual console` development tool helps but adds process overhead.
