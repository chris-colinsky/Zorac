"""
Session persistence for Zorac conversations.

This module handles saving and loading conversation history to/from disk,
enabling conversations to survive across application restarts.

Storage format:
  Sessions are stored as JSON files (default: ~/.zorac/session.json). JSON was
  chosen over alternatives like pickle or SQLite because:
    - It's human-readable — users can inspect or edit their session files
    - It's portable across Python versions (pickle is version-sensitive)
    - The data structure is simple (a list of message dicts) — no need for
      a database
    - It integrates naturally with the OpenAI message format, which is already
      a list of dicts with "role" and "content" keys

Error handling strategy:
  Both save and load operations use broad exception handling rather than
  catching specific exceptions. This is intentional for a CLI tool where:
    - File system errors are varied (permissions, disk full, network drives)
    - We'd rather show a warning and continue than crash the application
    - The user can always start a fresh session if loading fails

Session lifecycle:
  1. On startup: load_session() restores the previous conversation
  2. After each assistant response: save_session() persists automatically
  3. On /quit or /exit: save_session() ensures nothing is lost
  4. On /clear: save_session() persists the reset state
"""

import json

from openai.types.chat import ChatCompletionMessageParam

from .config import SESSION_FILE, ensure_zorac_dir
from .console import console


def save_session(messages: list[ChatCompletionMessageParam], filepath=SESSION_FILE):
    """Save conversation history to a JSON file.

    The messages list follows the OpenAI chat completion format — each message
    is a dict with "role" (system/user/assistant) and "content" keys. This
    format is used throughout the application, so we can serialize it directly
    without any transformation.

    Args:
        messages: The full conversation history to persist.
        filepath: Where to save (defaults to ~/.zorac/session.json).
                  The parameter is exposed primarily for testing, allowing
                  tests to use temporary files without touching the user's
                  real session.

    Returns:
        True if save succeeded, False otherwise.
    """
    try:
        # Ensure the parent directory exists before writing.
        # This handles the first-run case where ~/.zorac/ doesn't exist yet.
        ensure_zorac_dir()
        with open(filepath, "w") as f:
            # indent=2 makes the file human-readable if users want to inspect it
            json.dump(messages, f, indent=2)
        return True
    except Exception as e:
        console.print(f"[red]Error saving session: {e}[/red]")
        return False


def load_session(filepath=SESSION_FILE):
    """Load conversation history from a JSON file.

    Returns None in two cases:
      1. The file doesn't exist (first run, or session was deleted)
      2. The file contains invalid JSON (corrupted file)

    In both cases, the caller (main.py) will start a fresh conversation
    with just the system message. This makes the function safe to call
    unconditionally on startup.

    Args:
        filepath: Where to load from (defaults to ~/.zorac/session.json).

    Returns:
        List of message dicts if successful, None otherwise.
    """
    try:
        if filepath.exists():
            with open(filepath) as f:
                messages = json.load(f)
            return messages
        return None
    except Exception as e:
        console.print(f"[red]Error loading session: {e}[/red]")
        return None
