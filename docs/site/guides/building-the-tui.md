# Building the TUI

A guide to building a terminal user interface with Textual — covering the widget system, event handling, CSS styling, and the patterns used in Zorac.

---

## Why a TUI?

### Terminal vs Web vs Desktop

When building a chat application, you have several UI options:

| Approach | Pros | Cons |
|----------|------|------|
| **Web app** (Gradio, Streamlit) | Easy to build, browser-based | Requires a web server, heavier stack |
| **Desktop app** (Electron, Qt) | Rich UI, native feel | Large bundle size, complex build process |
| **CLI** (print/input) | Simple, universal | No formatting, no live updates |
| **TUI** (Textual, curses) | Rich UI in the terminal, lightweight | Terminal compatibility quirks |

Zorac uses a TUI because it fits the project's philosophy: lightweight, runs anywhere there's a terminal, zero browser or desktop framework overhead. For a tool that's already running on the command line next to vLLM, staying in the terminal feels natural.

### The Case for Rich Terminal Interfaces

Modern TUI frameworks like Textual have narrowed the gap with web UIs considerably. Zorac's TUI provides:

- **Real-time streaming** — Tokens appear as they're generated, with live Markdown rendering
- **Styled output** — Syntax-highlighted code blocks, formatted headings, colored text
- **Interactive widgets** — Multiline input with auto-complete, a persistent stats bar
- **Layout management** — A scrollable chat area with a docked input bar and status line

All of this runs in a standard terminal emulator with no external dependencies beyond Python.

---

## Textual Fundamentals

### App, Widgets, and Compose

Textual applications are built from three core concepts:

1. **App** — The application class that owns the event loop and manages global state
2. **Widgets** — UI components (text, buttons, inputs) that display content and handle interaction
3. **Compose** — A method that declares the widget layout using `yield` statements

Zorac's application class:

```python
class ZoracApp(CommandHandlersMixin, StreamingMixin, HistoryMixin, App):
    def compose(self) -> ComposeResult:
        all_triggers = sorted(self.command_handlers.keys())
        yield VerticalScroll(id="chat-log")
        yield Vertical(
            ChatInput(commands=all_triggers, id="user-input",
                      placeholder="Type your message or /command"),
            Static(" Ready ", id="stats-bar"),
            id="bottom-bar",
        )
```

The `compose()` method creates three key widgets:

- **`VerticalScroll#chat-log`** — A scrollable container that holds all chat messages
- **`ChatInput#user-input`** — The multiline input widget where you type
- **`Static#stats-bar`** — A text widget showing performance metrics

### The Event Loop

Textual manages an async event loop automatically. When you call `app.run()`, Textual:

1. Enters the terminal's **alternate screen** (separate from your normal terminal)
2. Calls `compose()` to build the widget tree
3. Fires `on_mount()` to initialize the application
4. Enters the event loop, dispatching keyboard events, messages, and timer callbacks
5. Exits the alternate screen when the app terminates

Zorac's lifecycle:

```python
def main():
    if is_first_run():
        print_header()       # Runs BEFORE Textual (normal terminal)
        run_first_time_setup()
    app = ZoracApp()
    app.run()                # Textual takes over the terminal
```

First-time setup runs before `app.run()` because it uses `input()` for prompting, which doesn't work inside Textual's alternate screen.

### CSS Styling (TCSS)

Textual uses a CSS subset called **TCSS** (Textual CSS) for layout and styling. It supports properties like `dock`, `padding`, `background`, `height`, and `color`.

Zorac's stylesheet is defined on the `ZoracApp` class:

```css
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
```

Key layout decisions:

- **`#bottom-bar` docks to the bottom** — The input and stats bar stay pinned, never scroll away
- **`#user-input` has min/max height** — Allows auto-resizing from 1 to 5 lines
- **`#stats-bar` uses a dark background** — Visually separates it from the chat content

### Message Passing

Widgets communicate through **messages** — typed objects that bubble up the DOM tree until a handler catches them. This decouples widgets from application logic.

Zorac's `ChatInput` widget posts a `Submitted` message when Enter is pressed:

```python
# In ChatInput (widget)
self.post_message(self.Submitted(input=self, value=text))

# In ZoracApp (application) — automatically matched by name
async def on_chat_input_submitted(self, event: ChatInput.Submitted):
    user_input = event.value.strip()
    # ... handle the input
```

Textual auto-routes messages to handlers named `on_<widget_class>_<message_name>`. The widget doesn't need to know about `ZoracApp`, and `ZoracApp` doesn't need to know about `ChatInput`'s internals.

---

## Zorac's UI Architecture

### Layout: Chat Log, Input, Stats Bar

```
┌─────────────────────────────────────────────┐
│                                             │
│  VerticalScroll#chat-log                    │
│  ┌─────────────────────────────────────┐    │
│  │ [System messages, user messages,    │    │
│  │  assistant responses with Markdown] │    │
│  │                                     │    │
│  │  (scrollable)                       │    │
│  └─────────────────────────────────────┘    │
│                                             │
├─────────────────────────────────────────────┤
│  ChatInput#user-input                       │
│  [Type your message or /command          ]  │
├─────────────────────────────────────────────┤
│  Static#stats-bar                           │
│  Stats: 245 tokens in 3.8s (64.5 tok/s)    │
└─────────────────────────────────────────────┘
```

The chat log takes up all available space and scrolls. The input bar and stats bar are docked to the bottom via the `#bottom-bar` container.

Messages are added to the chat log by **mounting** new widgets:

```python
def _log_user(self, text: str) -> None:
    chat_log = self.query_one("#chat-log", VerticalScroll)
    widget = Static(f"\n[bold blue]You:[/bold blue] {text}")
    chat_log.mount(widget)
    widget.scroll_visible()
```

Each message becomes a `Static` widget (for user/system messages) or a `Markdown` widget (for assistant responses). This widget-per-message approach is simpler than maintaining a single text buffer and gives Textual control over layout and scrolling.

### Custom Widgets: ChatInput

`ChatInput` extends Textual's `TextArea` to create a chat-oriented input:

```python
class ChatInput(TextArea):
    async def _on_key(self, event: events.Key) -> None:
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
```

Key behaviors:

- **Enter submits** — Overrides TextArea's default (insert newline)
- **Shift+Enter inserts newline** — Detected as `"shift+enter"` (Kitty protocol) or `"ctrl+j"` (iTerm2)
- **Auto-resize** — Height grows from 1 to 5 lines based on content
- **Command suggestions** — Typing `/` shows inline suggestions, accepted with Tab

### Markdown Rendering

Assistant responses are rendered as formatted Markdown using Textual's built-in `Markdown` widget. Zorac extends this with `LeftAlignedMarkdown` (in `zorac/markdown_custom.py`) to force left-aligned headings instead of Textual's default centered headings.

During streaming, Markdown is built incrementally using `Markdown.get_stream()`:

```python
md_widget = Markdown("")
chat_log.mount(md_widget)
stream = Markdown.get_stream(md_widget)

async for chunk in stream_response:
    await stream.write(content_chunk)  # Widget re-renders automatically

await stream.stop()  # Finalize the render
```

This API handles the complexity of incremental Markdown parsing — partial code blocks, incomplete lists, and in-progress formatting are all rendered correctly as tokens arrive.

### The Mixin Pattern

Zorac uses Python mixins to organize the application into focused modules:

```python
class ZoracApp(CommandHandlersMixin, StreamingMixin, HistoryMixin, App):
    ...
```

| Mixin | File | Responsibility |
|-------|------|---------------|
| `CommandHandlersMixin` | `handlers.py` | All `/command` handlers (`cmd_help`, `cmd_clear`, etc.) |
| `StreamingMixin` | `streaming.py` | The `_stream_response()` worker method |
| `HistoryMixin` | `history.py` | Command history load/save and Up/Down navigation |

Each mixin declares type stubs for the attributes it expects from `ZoracApp` (like `self.client`, `self.messages`). This provides type safety while keeping the code modular. The actual attributes are defined in `ZoracApp.__init__()`.

Python's MRO (Method Resolution Order) handles the composition: `ZoracApp` inherits from all three mixins and `App`, with methods resolved left-to-right.

---

## Streaming in the TUI

### Workers and Background Threads

LLM streaming can't run on the main thread — it would block the UI, making the application unresponsive. Textual provides **workers** for this:

```python
@work(exclusive=True, group="stream")
async def _stream_response(self) -> None:
    ...
```

The `@work` decorator:

- **Runs the method in a worker thread** — The UI event loop continues processing events (scrolling, Ctrl+C) while tokens stream in
- **`exclusive=True`** — Only one streaming worker can run at a time. Starting a new one cancels the old one.
- **`group="stream"`** — Allows cancellation by group name: `self.workers.cancel_group(self, "stream")`

### Markdown.get_stream() API

Textual's `Markdown.get_stream()` is purpose-built for streaming content into a Markdown widget:

```python
md_widget = Markdown("")
stream = Markdown.get_stream(md_widget)

# Write content incrementally
await stream.write("# Hello\n\nThis is ")
await stream.write("streaming ")
await stream.write("**markdown**.")

# Finalize
await stream.stop()
```

The stream handles:

- **Incremental parsing** — Content is re-parsed as new text arrives
- **Partial elements** — A half-written code block renders as plain text until the closing \`\`\` arrives
- **Widget updates** — The `Markdown` widget's DOM is updated automatically, triggering Textual's layout/paint cycle

### Keeping the UI Responsive

Several techniques keep the UI smooth during streaming:

1. **`scroll_end(animate=False)`** — Auto-scrolls the chat log without animation (animation would lag behind fast token output)
2. **Stats bar updates via `stats_bar.update()`** — Uses Textual's reactive update mechanism rather than direct terminal writes
3. **Cancellation check on each chunk** — `if worker.is_cancelled: break` ensures Ctrl+C stops streaming immediately rather than waiting for the current chunk
4. **Input disabled during streaming** — Prevents overlapping requests

---

## Patterns You Can Reuse

### Command Dispatch Tables

Instead of a long if/elif chain for command routing, Zorac uses a dictionary mapping commands to handler methods:

```python
self.command_handlers = {
    "/quit": self.cmd_quit,
    "/exit": self.cmd_quit,  # Alias
    "/help": self.cmd_help,
    "/clear": self.cmd_clear,
    "/save": self.cmd_save,
    ...
}

# Routing is a simple lookup
if cmd in self.command_handlers:
    await self.command_handlers[cmd](parts)
```

This pattern makes it easy to add new commands — define a `cmd_*` method and add an entry to the dictionary. The command list is also used to generate help text and autocomplete suggestions.

### Auto-Resizing Input

The `ChatInput` widget auto-resizes between 1 and 5 lines based on content:

```python
def _auto_resize(self) -> None:
    line_count = self.document.line_count
    content_height = max(1, min(5, line_count))
    self.styles.height = content_height + self._BORDER_OVERHEAD
```

This technique — setting `styles.height` dynamically based on content — works for any Textual widget. The `_BORDER_OVERHEAD` constant (2) accounts for the border's top and bottom rows.

### Persistent Stats Bar

The stats bar pattern — a docked `Static` widget that displays contextual information — is reusable in any Textual application:

```python
# In compose()
yield Static(" Ready ", id="stats-bar")

# Update from anywhere
stats_bar = self.query_one("#stats-bar", Static)
stats_bar.update(f" {tokens} tokens | {elapsed:.1f}s | {tps:.1f} tok/s ")
```

The stats bar shows different content depending on the application state:

- **Before any chat:** "Ready" or session info
- **During streaming:** Real-time token count, elapsed time, tokens/second
- **After streaming:** Final response stats plus conversation totals

This provides always-available status information without cluttering the chat log.
