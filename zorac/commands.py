"""
Command registry for Zorac interactive commands.

This module is the single source of truth for all interactive commands available
in the Zorac chat client. Rather than scattering command definitions across
multiple files, we centralize them here. This pattern has several benefits:

  1. Adding a new command only requires updating one place (this list).
  2. The /help display and system prompt stay automatically in sync.
  3. It's easy to see all available commands at a glance.

The registry serves two audiences:
  - **Users**: via `get_help_text()` which formats commands for terminal display
    with Rich markup for colors and alignment.
  - **The LLM**: via `get_system_prompt_commands()` which formats commands as
    plain text for inclusion in the system prompt, enabling the AI to understand
    and suggest commands naturally during conversation.

Architecture note:
  The actual command *handlers* (the code that runs when you type /help, /quit, etc.)
  live in main.py's Zorac class. This module only defines the *metadata* — names,
  descriptions, and detailed explanations. This separation keeps the registry
  lightweight and free of runtime dependencies.
"""

import datetime
from typing import TypedDict


class CommandInfo(TypedDict):
    """Type definition for command information.

    Using TypedDict (instead of a plain dict) gives us type safety — the type checker
    will catch typos like 'trigger' instead of 'triggers', and IDEs will offer
    autocomplete for the keys. This is especially valuable in a list of dicts where
    a typo could silently break the /help display.
    """

    triggers: list[str]  # Command triggers (e.g., ["/help"] or ["/quit", "/exit"])
    description: str  # Short one-line description for /help display
    detailed: str  # Detailed explanation for system prompt (LLM context)


# Centralized registry of all interactive commands.
# Each entry defines:
#   - triggers: The slash-command(s) that invoke this action. Some commands have
#     aliases (e.g., /quit and /exit do the same thing) for user convenience.
#   - description: A brief label shown in the /help output. Keep these short and
#     consistent in style (imperative verb + object).
#   - detailed: A thorough explanation included in the system prompt so the LLM
#     can understand what each command does and suggest them to users naturally.
COMMANDS: list[CommandInfo] = [
    {
        "triggers": ["/help"],
        "description": "Show all available commands",
        "detailed": "Display a list of all available interactive commands with descriptions. "
        "This helps users discover and understand the functionality available in Zorac.",
    },
    {
        "triggers": ["/quit", "/exit"],
        "description": "Save and exit the application",
        "detailed": "Save the current conversation session to disk and exit Zorac. "
        "The session will be automatically restored on the next run.",
    },
    {
        "triggers": ["/clear"],
        "description": "Reset conversation to initial system message",
        "detailed": "Clear the entire conversation history and reset to a fresh session "
        "with only the initial system message. The cleared session is automatically saved to disk.",
    },
    {
        "triggers": ["/save"],
        "description": "Manually save session to disk",
        "detailed": "Manually save the current conversation session to ~/.zorac/session.json. "
        "Sessions are auto-saved after each assistant response, but this command allows "
        "explicit saves at any time.",
    },
    {
        "triggers": ["/load"],
        "description": "Reload session from disk",
        "detailed": "Reload the conversation session from ~/.zorac/session.json, discarding "
        "any unsaved changes in the current session. Useful for reverting to the last saved state.",
    },
    {
        "triggers": ["/tokens"],
        "description": "Display current token usage statistics",
        "detailed": "Show detailed token usage information including current token count, "
        "token limit, remaining capacity, and message count. Helps users monitor "
        "conversation size and predict when auto-summarization will occur.",
    },
    {
        "triggers": ["/summarize"],
        "description": "Force conversation summarization",
        "detailed": "Force summarization of the conversation history, even if the token limit "
        "hasn't been reached. The LLM creates a concise summary of older messages while "
        "preserving the most recent messages. Useful for condensing long conversations manually.",
    },
    {
        "triggers": ["/summary"],
        "description": "Display the current conversation summary",
        "detailed": "Display the current conversation summary if one exists. The summary is "
        "created automatically when the conversation exceeds the token limit, or manually "
        "via the /summarize command. If no summary exists, a message is displayed.",
    },
    {
        "triggers": ["/reconnect"],
        "description": "Retry connection to the vLLM server",
        "detailed": "Attempt to reconnect to the vLLM server. Useful when the server was "
        "unavailable at startup or the connection was lost. Shows the same connection "
        "status as the initial startup check.",
    },
    {
        "triggers": ["/config"],
        "description": "Manage configuration settings",
        "detailed": "Manage Zorac configuration settings. Supports three subcommands: "
        "'/config list' shows all current configuration values, "
        "'/config set <KEY> <VALUE>' updates a configuration setting, and "
        "'/config get <KEY>' displays a specific configuration value. "
        "Settings include server URL, model name, token limits, temperature, and streaming mode.",
    },
]


def get_help_text() -> str:
    """Generate formatted help text for the /help command display.

    This produces Rich-markup-formatted text for beautiful terminal output.
    Rich markup uses bracket syntax like [cyan]text[/cyan] for colors and
    [bold]text[/bold] for styles.

    The formatting includes:
      - Color-coded command names (cyan) for visual distinction
      - Manual padding for column alignment (since Rich's console.print
        handles the markup stripping for width calculation)
      - Special handling for /config to show its subcommands indented
      - Keyboard shortcuts section for Ctrl+C and Ctrl+D

    Returns:
        Formatted string with Rich markup, ready for console.print().
    """
    lines = ["[bold]Available Commands:[/bold]"]

    for cmd in COMMANDS:
        # Join triggers for display (e.g., "/quit, /exit")
        trigger_str = ", ".join(cmd["triggers"])

        # Special handling for /config to show subcommands inline.
        # This gives users a quick reference without needing to type /config alone.
        if "/config" in cmd["triggers"]:
            lines.append(f"  [cyan]{trigger_str}[/cyan]            - {cmd['description']}")
            lines.append("    [cyan]/config list[/cyan]     - Show current configuration")
            lines.append("    [cyan]/config set[/cyan]      - Set a configuration value")
            lines.append("    [cyan]/config get[/cyan]      - Get a specific configuration value")
        else:
            # Pad trigger strings to align descriptions in a clean column.
            # 18 chars accommodates the longest trigger string ("/quit, /exit" = 12 chars).
            padding = " " * (18 - len(trigger_str))
            lines.append(f"  [cyan]{trigger_str}[/cyan]{padding}- {cmd['description']}")

    # Keyboard shortcuts section
    lines.append("")
    lines.append("[bold]Keyboard Shortcuts:[/bold]")
    lines.append("  [cyan]Ctrl+C[/cyan]            - Interrupt streaming response")
    lines.append("  [cyan]Ctrl+D[/cyan]            - Save and exit")
    lines.append("  [cyan]Shift+Enter[/cyan]       - Insert a newline")
    lines.append("  [cyan]Tab[/cyan]               - Accept command suggestion")

    return "\n".join(lines)


def get_initial_system_message() -> str:
    """Build the initial system message that defines the LLM's behavior.

    The system message serves two purposes:
      1. Identity & behavior: "You are Zorac, a helpful AI assistant"
      2. Command awareness: Includes all available commands so the LLM can
         suggest them naturally when users ask about functionality

    Including the current date allows the LLM to answer time-sensitive questions
    and provides temporal context for the conversation.

    Returns:
        The complete system message string to use as messages[0].
    """
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    base_message = f"You are Zorac, a helpful AI assistant. Today's date is {today}."
    command_info = get_system_prompt_commands()
    return f"{base_message}{command_info}"


def get_system_prompt_commands() -> str:
    """Generate command information for inclusion in the LLM's system prompt.

    Unlike get_help_text(), this produces plain text (no Rich markup) because
    it will be sent to the LLM as part of the system message. Rich markup like
    [cyan] would confuse the model and waste tokens.

    Including command information in the system prompt enables the LLM to:
      - Answer questions like "how do I save my session?" naturally
      - Proactively suggest relevant commands (e.g., "you can use /tokens to check usage")
      - Understand the application context it's operating within

    Returns:
        Plain-text formatted string describing all commands for LLM context.
    """
    lines = [
        "\nThe user is interacting with you through Zorac, a terminal-based chat client for local LLMs.",
        "",
        "Available Commands:",
        "The following commands are available to the user:",
        "",
    ]

    for cmd in COMMANDS:
        # Use "or" instead of "," for the LLM — reads more naturally in prose.
        trigger_str = " or ".join(cmd["triggers"])
        lines.append(f"{trigger_str} - {cmd['detailed']}")
        lines.append("")

    lines.append("Keyboard Shortcuts:")
    lines.append(
        "Ctrl+C - Interrupt a streaming response. The partial response is discarded and the user can continue chatting."
    )
    lines.append("Ctrl+D - Save the session and exit the application.")
    lines.append("Shift+Enter - Insert a newline in the input (Enter submits).")
    lines.append("Tab - Accept an inline command suggestion.")
    lines.append("")
    lines.append(
        "When users ask about functionality, help them understand these commands naturally. "
        "Suggest relevant commands when appropriate and provide usage examples when helpful."
    )

    return "\n".join(lines)
