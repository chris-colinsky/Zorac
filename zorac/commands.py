"""
Command registry for Zorac interactive commands.

This module provides a centralized registry of all available interactive commands
in the Zorac chat client. The registry is used for both the /help display and
to inform the LLM about available functionality through the system prompt.
"""

from typing import TypedDict


class CommandInfo(TypedDict):
    """Type definition for command information."""

    command: str  # Command name (e.g., "/help", "/quit or /exit")
    description: str  # Short one-line description for /help display
    detailed: str  # Detailed explanation for system prompt


# Centralized registry of all interactive commands
COMMANDS: list[CommandInfo] = [
    {
        "command": "/help",
        "description": "Show all available commands",
        "detailed": "Display a list of all available interactive commands with descriptions. "
        "This helps users discover and understand the functionality available in Zorac.",
    },
    {
        "command": "/quit or /exit",
        "description": "Save and exit the application",
        "detailed": "Save the current conversation session to disk and exit Zorac. "
        "The session will be automatically restored on the next run.",
    },
    {
        "command": "/clear",
        "description": "Reset conversation to initial system message",
        "detailed": "Clear the entire conversation history and reset to a fresh session "
        "with only the initial system message. The cleared session is automatically saved to disk.",
    },
    {
        "command": "/save",
        "description": "Manually save session to disk",
        "detailed": "Manually save the current conversation session to ~/.zorac/session.json. "
        "Sessions are auto-saved after each assistant response, but this command allows "
        "explicit saves at any time.",
    },
    {
        "command": "/load",
        "description": "Reload session from disk",
        "detailed": "Reload the conversation session from ~/.zorac/session.json, discarding "
        "any unsaved changes in the current session. Useful for reverting to the last saved state.",
    },
    {
        "command": "/tokens",
        "description": "Display current token usage statistics",
        "detailed": "Show detailed token usage information including current token count, "
        "token limit, remaining capacity, and message count. Helps users monitor "
        "conversation size and predict when auto-summarization will occur.",
    },
    {
        "command": "/summarize",
        "description": "Force conversation summarization",
        "detailed": "Force summarization of the conversation history, even if the token limit "
        "hasn't been reached. The LLM creates a concise summary of older messages while "
        "preserving the most recent messages. Useful for condensing long conversations manually.",
    },
    {
        "command": "/summary",
        "description": "Display the current conversation summary",
        "detailed": "Display the current conversation summary if one exists. The summary is "
        "created automatically when the conversation exceeds the token limit, or manually "
        "via the /summarize command. If no summary exists, a message is displayed.",
    },
    {
        "command": "/config",
        "description": "Manage configuration settings",
        "detailed": "Manage Zorac configuration settings. Supports three subcommands: "
        "'/config list' shows all current configuration values, "
        "'/config set <KEY> <VALUE>' updates a configuration setting, and "
        "'/config get <KEY>' displays a specific configuration value. "
        "Settings include server URL, model name, token limits, temperature, and streaming mode.",
    },
]


def get_help_text() -> str:
    """
    Generate formatted help text for /help command display.

    Returns:
        Formatted string with all commands and descriptions.
    """
    lines = ["[bold]Available Commands:[/bold]"]

    for cmd in COMMANDS:
        # Special handling for /config to show subcommands
        if cmd["command"] == "/config":
            lines.append(f"  [cyan]{cmd['command']}[/cyan]            - {cmd['description']}")
            lines.append("    [cyan]/config list[/cyan]     - Show current configuration")
            lines.append("    [cyan]/config set[/cyan]      - Set a configuration value")
            lines.append("    [cyan]/config get[/cyan]      - Get a specific configuration value")
        else:
            # Calculate padding for alignment
            padding = " " * (18 - len(cmd["command"]))
            lines.append(f"  [cyan]{cmd['command']}[/cyan]{padding}- {cmd['description']}")

    return "\n".join(lines)


def get_system_prompt_commands() -> str:
    """
    Generate command information for system prompt.

    Returns:
        Formatted string describing all commands for LLM context.
    """
    lines = [
        "\nThe user is interacting with you through Zorac, a terminal-based chat client for local LLMs.",
        "",
        "Available Commands:",
        "The following commands are available to the user:",
        "",
    ]

    for cmd in COMMANDS:
        lines.append(f"{cmd['command']} - {cmd['detailed']}")
        lines.append("")

    lines.append(
        "When users ask about functionality, help them understand these commands naturally. "
        "Suggest relevant commands when appropriate and provide usage examples when helpful."
    )

    return "\n".join(lines)
