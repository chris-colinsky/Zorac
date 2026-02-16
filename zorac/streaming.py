"""
LLM response streaming for the Zorac TUI.

Provides StreamingMixin with the _stream_response worker method,
mixed into ZoracApp.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from textual import work
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

from .session import save_session
from .utils import count_tokens
from .widgets import ChatInput

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam


class StreamingMixin:
    """Mixin providing the _stream_response worker for LLM streaming."""

    # Type stubs for attributes provided by ZoracApp
    client: AsyncOpenAI | None
    messages: list[ChatCompletionMessageParam]
    stream_enabled: bool
    vllm_model: str
    temperature: float
    max_output_tokens: int
    encoding: Any
    _streaming: bool
    stats: dict[str, int | float]

    # Method stubs for ZoracApp / App methods — only present during type
    # checking so they don't shadow real methods inherited via MRO at runtime.
    if TYPE_CHECKING:

        def _log_system(self, text: str, style: str = "dim") -> None: ...
        def _update_stats_bar(self) -> None: ...
        def query_one(self, selector: str, expect_type: type | None = None) -> Any: ...

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
