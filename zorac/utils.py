from importlib.metadata import PackageNotFoundError, version

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from rich import box
from rich.panel import Panel

from .config import TIKTOKEN_ENCODING, VLLM_BASE_URL, VLLM_MODEL
from .console import console


def get_version() -> str:
    """Get the installed package version."""
    try:
        return version("zorac")
    except PackageNotFoundError:
        return "dev"


def count_tokens(
    messages: list[ChatCompletionMessageParam],
    encoding_name: str | None = None,
):
    """Count tokens in messages using tiktoken.

    Args:
        messages: List of chat messages to count tokens for.
        encoding_name: Tiktoken encoding name (e.g. "cl100k_base", "o200k_base").
                       Defaults to the configured TIKTOKEN_ENCODING.
    """
    resolved_encoding = encoding_name or TIKTOKEN_ENCODING
    try:
        encoding = tiktoken.get_encoding(resolved_encoding)
    except ValueError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for _, value in message.items():
            if isinstance(value, str):
                num_tokens += len(encoding.encode(value))
            elif isinstance(value, list):
                # Handle list of content parts if necessary, for now just stringify
                for part in value:
                    if isinstance(part, dict) and "text" in part:
                        num_tokens += len(encoding.encode(str(part["text"])))
    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens


def print_header():
    """Print welcome header with ASCII art logo"""
    # ASCII art logo for Zorac (purple color matching terminal prompt)
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
    """Verify connection to the vLLM server"""
    try:
        with console.status("[bold green]Verifying connection...[/bold green]"):
            await client.models.list()
        console.print("[green]✓ Connection verified[/green]")
        return True
    except Exception as e:
        console.print("[bold red]✗ Connection failed![/bold red]")
        console.print(f"[red]Error: {e}[/red]")
        console.print(f"[yellow]Target URL: {client.base_url}[/yellow]")
        console.print(
            "[yellow]Review your .env file and ensure the VLLM_BASE_URL is correct and the server is running.[/yellow]"
        )
        return False
