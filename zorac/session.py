import json

from openai.types.chat import ChatCompletionMessageParam

from .config import SESSION_FILE, ensure_zorac_dir
from .console import console


def save_session(messages: list[ChatCompletionMessageParam], filepath=SESSION_FILE):
    """Save conversation history to a JSON file"""
    try:
        ensure_zorac_dir()
        with open(filepath, "w") as f:
            json.dump(messages, f, indent=2)
        return True
    except Exception as e:
        console.print(f"[red]Error saving session: {e}[/red]")
        return False


def load_session(filepath=SESSION_FILE):
    """Load conversation history from a JSON file"""
    try:
        if filepath.exists():
            with open(filepath) as f:
                messages = json.load(f)
            return messages
        return None
    except Exception as e:
        console.print(f"[red]Error loading session: {e}[/red]")
        return None
