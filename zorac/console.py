"""
Shared Rich Console singleton for terminal output.

This module creates a single, shared Rich Console instance used across the entire
application. This is a common pattern called the "Singleton pattern" — instead of
each module creating its own Console, they all import and share this one.

Why a singleton?
  - Rich's Console manages terminal state (width, color support, cursor position).
    Having multiple Console instances can cause conflicts and inconsistent output.
  - A single instance ensures all output goes through one coordinated pipeline,
    which matters when using Rich's Live display and Status spinners.
  - It also makes testing easier: tests can mock `console` in one place and
    all modules pick up the mock automatically.

Usage:
    from .console import console
    console.print("[green]Success![/green]")
"""

from rich.console import Console

# The shared console instance. All terminal output in Zorac flows through this
# object, ensuring consistent formatting and preventing output conflicts between
# Rich's advanced features like Live updates and Status spinners.
console = Console()
