"""
Utility functions for Zorac.

This module contains shared helper functions used across the application:
  - Token counting (for context window management)
  - Welcome header display (ASCII art and connection info)
  - Server connection verification

These are pure utility functions with no application state — they take inputs
and produce outputs without side effects beyond console output. This makes
them easy to test and reuse.
"""

from importlib.metadata import PackageNotFoundError, version

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich import box
from rich.panel import Panel

from .config import TIKTOKEN_ENCODING, VLLM_BASE_URL, VLLM_MODEL
from .console import console


def get_version() -> str:
    """Get the installed package version from Python package metadata.

    Uses importlib.metadata to read the version from the installed package.
    This approach reads from the package's dist-info directory (created by pip/uv
    during installation), so the version is always accurate without needing to
    parse pyproject.toml at runtime.

    Returns "dev" when running from source without installing, which is a
    conventional fallback for development environments.
    """
    try:
        return version("zorac")
    except PackageNotFoundError:
        return "dev"


def count_tokens(
    messages: list[ChatCompletionMessageParam],
    encoding_name: str | None = None,
):
    """Count tokens in a list of chat messages using tiktoken.

    This function is critical for context window management. LLMs have a fixed
    context window (input + output tokens). We need to know how many tokens
    our conversation history uses so we can:
      1. Trigger auto-summarization before hitting the limit
      2. Display usage stats to the user via /tokens
      3. Ensure we leave enough room for the model's response

    Token counting methodology:
      We follow OpenAI's token counting approach for chat models:
      - Each message has 4 tokens of overhead for the message envelope:
        <im_start>{role/name}\\n{content}<im_end>\\n
      - Each string value in the message is encoded and counted
      - 2 tokens are added at the end for the reply priming: <im_start>assistant

    Why tiktoken?
      tiktoken is OpenAI's fast BPE tokenizer library. It gives us an accurate
      token count that matches what the model actually sees. The default encoding
      (cl100k_base) is used by GPT-4 and most modern models. Users running
      different model families can configure TIKTOKEN_ENCODING to match their
      model's tokenizer.

    Note: The token count is an approximation. Different models may tokenize
    slightly differently, but this is close enough for context management.

    Args:
        messages: List of chat messages in OpenAI format.
        encoding_name: Tiktoken encoding name (e.g., "cl100k_base", "o200k_base").
                       Defaults to the configured TIKTOKEN_ENCODING.

    Returns:
        Estimated token count for the entire message list.
    """
    # Resolve the encoding, falling back to the configured default.
    # If an invalid encoding is specified, fall back to cl100k_base as a safe default.
    resolved_encoding = encoding_name or TIKTOKEN_ENCODING
    try:
        encoding = tiktoken.get_encoding(resolved_encoding)
    except ValueError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        # Every message has a fixed overhead of 4 tokens for the envelope format:
        # <im_start>{role}\n{content}<im_end>\n
        num_tokens += 4
        for _, value in message.items():
            if isinstance(value, str):
                # Standard string content — encode and count
                num_tokens += len(encoding.encode(value))
            elif isinstance(value, list):
                # Handle multipart content (e.g., messages with both text and images).
                # We only count the text parts; image tokens are handled differently
                # by the model and aren't relevant for our context management.
                for part in value:
                    if isinstance(part, dict) and "text" in part:
                        num_tokens += len(encoding.encode(str(part["text"])))

    # Every conversation is primed with <im_start>assistant, which adds 2 tokens.
    # This accounts for the model's reply priming overhead.
    num_tokens += 2
    return num_tokens


def print_header():
    """Print the welcome header with ASCII art logo and connection info.

    This is displayed once at startup. The header serves dual purposes:
      1. Brand identity — the ASCII art logo makes Zorac visually distinctive
      2. Quick reference — shows connection info and available commands

    The ASCII art uses Unicode box-drawing characters (█, ╔, ║, etc.) for the
    logo, styled with Rich's markup for purple coloring.

    Rich's Panel widget creates a bordered box around the connection info,
    with box.ROUNDED for aesthetically pleasing rounded corners. The panel
    uses expand=False so it only takes up as much width as needed.
    """
    # ASCII art logo using Unicode block characters.
    # The [bold purple]...[/bold purple] is Rich markup for styling.
    logo = """[bold purple]
     ███████╗ ██████╗ ██████╗  █████╗  ██████╗`
     ╚══███╔╝██╔═══██╗██╔══██╗██╔══██╗██╔════╝
       ███╔╝ ██║   ██║██████╔╝███████║██║
      ███╔╝  ██║   ██║██╔══██╗██╔══██║██║
     ███████╗╚██████╔╝██║  ██║██║  ██║╚██████╗
     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
[/bold purple]
        [dim italic]intelligence running on localhost[/dim italic]
"""

    # Connection info and command quick-reference displayed in a bordered panel.
    # [dim] tags make less important info visually recede, while [green] and [bold]
    # draw attention to interactive elements and key information.
    header_text = f"""[dim]Zorac v{get_version()}[/dim]
[dim]Connected to: {VLLM_BASE_URL}[/dim]
[dim]Model: {VLLM_MODEL}[/dim]

[bold]Commands:[/bold]
  [green]/clear[/green]     - Clear conversation history
  [green]/save[/green]      - Manually save session
  [green]/load[/green]      - Reload session from disk
  [green]/tokens[/green]    - Show current token usage
  [green]/summarize[/green] - Force summarization of conversation
  [green]/summary[/green]   - Show current conversation summary
  [green]/reconnect[/green] - Retry server connection
  [green]/config[/green]    - Manage configuration
  [green]/quit[/green]      - Exit the chat"""

    console.print(logo)
    console.print(Panel(header_text, box=box.ROUNDED, expand=False))


async def check_connection(client: AsyncOpenAI) -> bool:
    """Verify connection to the vLLM server by listing available models.

    Uses the OpenAI-compatible /v1/models endpoint to verify the server is
    reachable and responding. This is a lightweight health check — the models
    list endpoint is fast and doesn't require loading the model into GPU memory.

    This function is async because the OpenAI client's HTTP calls are async.
    Using async here prevents the connection check from blocking the event loop,
    which matters because vLLM servers can sometimes be slow to respond
    (especially during model loading).

    The Rich status spinner ("Verifying connection...") provides visual feedback
    during the check, which can take a few seconds on slow networks.

    Args:
        client: The AsyncOpenAI client instance configured with the server URL.

    Returns:
        True if the server is reachable and responding, False otherwise.
    """
    try:
        with console.status("[bold green]Verifying connection...[/bold green]"):
            # client.models.list() calls GET /v1/models on the vLLM server.
            # If this succeeds, the server is running and accepting requests.
            await client.models.list()
        console.print("[green]✓ Connection verified[/green]")
        return True
    except Exception as e:
        # Broad exception handling because connection failures can manifest as
        # many different exception types: ConnectionError, TimeoutError,
        # httpx.ConnectError, etc. We display the specific error for debugging.
        console.print("[bold red]✗ Connection failed![/bold red]")
        console.print(f"[red]Error: {e}[/red]")
        console.print(f"[yellow]Target URL: {client.base_url}[/yellow]")
        console.print(
            "[yellow]Review your .env file and ensure the VLLM_BASE_URL is correct and the server is running.[/yellow]"
        )
        return False
