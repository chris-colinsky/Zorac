"""
Configuration management for Zorac.

This module handles all configuration: defaults, file-based config, environment
variable overrides, and first-time setup. It implements a three-tier priority system:

    Environment Variable > Config File (~/.zorac/config.json) > Default Value

This priority order is a common pattern in CLI tools (used by Docker, kubectl, etc.)
because it provides flexibility at every level:
  - Defaults: Sensible out-of-the-box experience, no setup required.
  - Config file: Persistent user preferences that survive restarts.
  - Environment variables: Temporary overrides for testing or one-off runs,
    e.g., `VLLM_BASE_URL=http://other-server:8000/v1 zorac`

Module-level constants:
  Configuration values like VLLM_BASE_URL and MAX_INPUT_TOKENS are resolved at
  import time (bottom of this file) and exposed as module-level constants. This
  means they're computed once when the application starts. The /config command
  in main.py handles runtime changes by updating both the config file and the
  in-memory Zorac instance attributes directly.

Why python-dotenv?
  The `load_dotenv()` call at module import reads a `.env` file from the project
  root and injects its values into `os.environ`. This is convenient for development
  (keeps secrets out of shell history) and follows the twelve-factor app methodology
  for configuration (https://12factor.net/config).
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .console import console

# Load environment variables from .env file in the project root.
# This call modifies os.environ, making .env values available via os.getenv().
# If no .env file exists, this is a no-op (no error raised).
load_dotenv()

# Default configuration values.
# These are the fallback values used when neither an environment variable nor
# a config file entry exists for a given setting. They represent the "zero config"
# experience — Zorac should work with these defaults if you have a standard
# vLLM server running on localhost.
DEFAULT_CONFIG = {
    "VLLM_BASE_URL": "http://localhost:8000/v1",  # vLLM's default OpenAI-compatible endpoint
    "VLLM_API_KEY": "EMPTY",  # vLLM doesn't require auth by default; "EMPTY" is conventional
    "VLLM_MODEL": "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ",  # 24B AWQ-quantized model
    "MAX_INPUT_TOKENS": "12000",  # Max tokens for system prompt + conversation history
    "MAX_OUTPUT_TOKENS": "4000",  # Max tokens the model can generate per response
    "KEEP_RECENT_MESSAGES": "6",  # Messages preserved during auto-summarization
    "TEMPERATURE": "0.1",  # Low temperature = more deterministic/focused responses
    "STREAM": "true",  # Stream tokens in real-time for responsive feel
    "TIKTOKEN_ENCODING": "cl100k_base",  # Token counting encoding (matches GPT-4/ChatGPT)
    "CODE_THEME": "monokai",  # Pygments syntax highlighting theme for code blocks
}

# File paths for Zorac's data directory.
# All user data lives under ~/.zorac/ by default, keeping the home directory clean.
# Each path can be overridden via environment variables for custom setups
# (e.g., storing data on a different drive or in a dotfiles-managed location).
ZORAC_DIR = Path(os.getenv("ZORAC_DIR", str(Path.home() / ".zorac")))
SESSION_FILE = Path(os.getenv("ZORAC_SESSION_FILE", str(ZORAC_DIR / "session.json")))
HISTORY_FILE = Path(os.getenv("ZORAC_HISTORY_FILE", str(ZORAC_DIR / "history")))
CONFIG_FILE = Path(os.getenv("ZORAC_CONFIG_FILE", str(ZORAC_DIR / "config.json")))


def ensure_zorac_dir():
    """Ensure the ~/.zorac/ storage directory exists, creating it if needed.

    Called before any file write operation (saving sessions, config, etc.).
    Uses `parents=True` to create intermediate directories and `exist_ok=True`
    to avoid errors if the directory already exists (race condition safe).

    We catch all exceptions here because directory creation can fail for many
    reasons (permissions, disk full, read-only filesystem) and we'd rather
    warn the user than crash the application.
    """
    if not ZORAC_DIR.exists():
        try:
            ZORAC_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not create directory {ZORAC_DIR}: {e}[/yellow]")


def load_config() -> dict[str, Any]:
    """Load configuration from the JSON config file (~/.zorac/config.json).

    Returns an empty dict if the file doesn't exist (first run) or can't be
    parsed (corrupted). This makes it safe to call unconditionally — the caller
    can always fall back to defaults.

    Returns:
        Dictionary of configuration key-value pairs, or empty dict on failure.
    """
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config: dict[str, Any] = json.load(f)
                return config
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load config file: {e}[/yellow]")
    return {}


def save_config(config: dict[str, Any]) -> bool:
    """Save configuration to the JSON config file.

    Ensures the parent directory exists before writing. Uses indent=2 for
    human-readable formatting so users can edit the config file manually
    if they prefer.

    Args:
        config: Dictionary of configuration key-value pairs to persist.

    Returns:
        True if save succeeded, False otherwise.
    """
    try:
        ensure_zorac_dir()
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        console.print(f"[red]Error saving config file: {e}[/red]")
        return False


def get_setting(key: str, default: str) -> str:
    """Get a configuration value using the three-tier priority system.

    Priority: Environment Variable > Config File > Default

    This is the core configuration lookup function. The typed variants
    (get_int_setting, get_float_setting, get_bool_setting) all delegate to
    this function and add type conversion on top.

    Args:
        key: The configuration key to look up (e.g., "VLLM_BASE_URL").
        default: Fallback value if not found in env or config file.

    Returns:
        The resolved configuration value as a string.
    """
    # 1. Check environment variable first (highest priority).
    #    This allows temporary overrides: VLLM_BASE_URL=http://... zorac
    env_val = os.getenv(key)
    if env_val:
        return env_val

    # 2. Check config file (medium priority).
    #    These are the user's persistent preferences from ~/.zorac/config.json.
    config = load_config()
    if key in config:
        return str(config[key])

    # 3. Fall back to the provided default (lowest priority).
    return default


def get_int_setting(key: str, default: int) -> int:
    """Get an integer configuration value with type validation.

    Wraps get_setting() and converts the string result to int. If conversion
    fails (e.g., user set MAX_INPUT_TOKENS="abc"), logs a warning and returns
    the default. This prevents the application from crashing due to invalid
    configuration values.

    Args:
        key: The configuration key to look up.
        default: Fallback integer value.

    Returns:
        The resolved configuration value as an integer.
    """
    value = get_setting(key, str(default))
    try:
        return int(value)
    except ValueError:
        console.print(
            f"[yellow]Warning: Invalid integer value for {key}: {value}, using default {default}[/yellow]"
        )
        return default


def get_float_setting(key: str, default: float) -> float:
    """Get a float configuration value with type validation.

    Used primarily for TEMPERATURE, which controls LLM response randomness.
    Range is 0.0 (deterministic) to 2.0 (very random), with 0.1 as default
    for focused, consistent responses.

    Args:
        key: The configuration key to look up.
        default: Fallback float value.

    Returns:
        The resolved configuration value as a float.
    """
    value = get_setting(key, str(default))
    try:
        return float(value)
    except ValueError:
        console.print(
            f"[yellow]Warning: Invalid float value for {key}: {value}, using default {default}[/yellow]"
        )
        return default


def get_bool_setting(key: str, default: bool) -> bool:
    """Get a boolean configuration value with flexible parsing.

    Accepts multiple truthy strings ("true", "1", "yes", "on") for user
    convenience. Anything else is treated as False. This is more forgiving
    than Python's strict bool() which treats any non-empty string as True.

    Args:
        key: The configuration key to look up.
        default: Fallback boolean value.

    Returns:
        The resolved configuration value as a boolean.
    """
    value = get_setting(key, str(default).lower())
    return value.lower() in ("true", "1", "yes", "on")


def is_first_run() -> bool:
    """Check if this is the first time Zorac is being run.

    We detect first run by checking for the config file's existence. If it
    doesn't exist, the user has never completed setup. This is simpler and
    more reliable than using a separate "first run" flag file.
    """
    return not CONFIG_FILE.exists()


def run_first_time_setup() -> None:
    """Interactive first-time setup wizard.

    Guides new users through basic configuration (server URL and model name)
    with sensible defaults. The wizard:
      1. Shows a welcome message explaining what's happening
      2. Prompts for the vLLM server URL (Enter to accept default)
      3. Prompts for the model name (Enter to accept default)
      4. Saves all settings (including defaults for other options) to config file

    Using input() instead of prompt_toolkit here because:
      - prompt_toolkit hasn't been initialized yet at this point
      - The setup wizard is simple enough that basic input() works fine
      - We want to keep the first-run experience lightweight

    The lazy import of Rich components (Columns, Rule) avoids circular imports
    and keeps the module's top-level imports minimal.
    """
    from rich.columns import Columns
    from rich.rule import Rule

    console.print()

    # Create a styled horizontal rule as a visual separator.
    # Wrapped in Columns to constrain its width to match the header panel (~56 chars).
    rule = Rule("[bold cyan]Welcome to Zorac![/bold cyan]", style="cyan")
    console.print(Columns([rule], width=56, expand=False))

    console.print()
    console.print("This appears to be your first time running Zorac.")
    console.print("Let's configure your vLLM server connection.")
    console.print()

    # Prompt for server URL — most users will either accept the default
    # (local vLLM server) or enter their homelab server's address.
    console.print("[bold]Server Configuration:[/bold]")
    console.print(f"  Default: {DEFAULT_CONFIG['VLLM_BASE_URL']}")
    base_url = input("  vLLM Server URL (or press Enter for default): ").strip()
    if not base_url:
        base_url = DEFAULT_CONFIG["VLLM_BASE_URL"]

    # Prompt for model name — different users may run different models
    # depending on their GPU and VRAM capacity.
    console.print(f"\n  Default: {DEFAULT_CONFIG['VLLM_MODEL']}")
    model = input("  Model name (or press Enter for default): ").strip()
    if not model:
        model = DEFAULT_CONFIG["VLLM_MODEL"]

    # Build complete config including all default values. We save everything
    # (not just what the user changed) so the config file serves as a complete
    # reference of all available settings.
    config = {
        "VLLM_BASE_URL": base_url.rstrip("/"),  # Normalize: remove trailing slash
        "VLLM_MODEL": model,
        "VLLM_API_KEY": DEFAULT_CONFIG["VLLM_API_KEY"],
        "MAX_INPUT_TOKENS": DEFAULT_CONFIG["MAX_INPUT_TOKENS"],
        "MAX_OUTPUT_TOKENS": DEFAULT_CONFIG["MAX_OUTPUT_TOKENS"],
        "KEEP_RECENT_MESSAGES": DEFAULT_CONFIG["KEEP_RECENT_MESSAGES"],
        "TEMPERATURE": DEFAULT_CONFIG["TEMPERATURE"],
        "STREAM": DEFAULT_CONFIG["STREAM"],
        "TIKTOKEN_ENCODING": DEFAULT_CONFIG["TIKTOKEN_ENCODING"],
        "CODE_THEME": DEFAULT_CONFIG["CODE_THEME"],
    }

    if save_config(config):
        console.print()
        console.print(f"[green]✓ Configuration saved to {CONFIG_FILE}[/green]")
        console.print("[dim]You can change these settings anytime with /config[/dim]")
        console.print()
    else:
        console.print("[red]✗ Failed to save configuration[/red]")
        console.print()


# ---------------------------------------------------------------------------
# Module-level configuration constants
# ---------------------------------------------------------------------------
# These are resolved once at import time using the three-tier priority system.
# They're used as defaults throughout the application. Note that some of these
# (like TEMPERATURE, MAX_OUTPUT_TOKENS) can be changed at runtime via /config,
# which updates both the config file and the Zorac instance's attributes.
# The module-level constants here represent the *initial* values.

# Server connection settings
VLLM_BASE_URL = get_setting("VLLM_BASE_URL", DEFAULT_CONFIG["VLLM_BASE_URL"]).strip().rstrip("/")
VLLM_API_KEY = get_setting("VLLM_API_KEY", DEFAULT_CONFIG["VLLM_API_KEY"])
VLLM_MODEL = get_setting("VLLM_MODEL", DEFAULT_CONFIG["VLLM_MODEL"])

# Display and encoding settings
TIKTOKEN_ENCODING = get_setting("TIKTOKEN_ENCODING", DEFAULT_CONFIG["TIKTOKEN_ENCODING"])
CODE_THEME = get_setting("CODE_THEME", DEFAULT_CONFIG["CODE_THEME"])

# Token limits — these control the context window management.
# MAX_INPUT_TOKENS triggers auto-summarization when exceeded.
# MAX_OUTPUT_TOKENS caps how much the model can generate per response.
# KEEP_RECENT_MESSAGES determines how many recent messages are preserved
# when older messages are summarized.
MAX_INPUT_TOKENS = get_int_setting("MAX_INPUT_TOKENS", 12000)
MAX_OUTPUT_TOKENS = get_int_setting("MAX_OUTPUT_TOKENS", 4000)
KEEP_RECENT_MESSAGES = get_int_setting("KEEP_RECENT_MESSAGES", 6)

# Model parameters
TEMPERATURE = get_float_setting("TEMPERATURE", 0.1)
STREAM = get_bool_setting("STREAM", True)
