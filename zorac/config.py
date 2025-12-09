import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .console import console

# Load environment variables from .env file
load_dotenv()

# Configuration Defaults
DEFAULT_CONFIG = {
    "VLLM_BASE_URL": "http://localhost:8000/v1",
    "VLLM_API_KEY": "EMPTY",
    "VLLM_MODEL": "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ",
    "MAX_INPUT_TOKENS": "12000",
    "MAX_OUTPUT_TOKENS": "4000",
    "KEEP_RECENT_MESSAGES": "6",
    "TEMPERATURE": "0.1",
    "STREAM": "true",
}

# File Paths
ZORAC_DIR = Path(os.getenv("ZORAC_DIR", str(Path.home() / ".zorac")))
SESSION_FILE = Path(os.getenv("ZORAC_SESSION_FILE", str(ZORAC_DIR / "session.json")))
HISTORY_FILE = Path(os.getenv("ZORAC_HISTORY_FILE", str(ZORAC_DIR / "history")))
CONFIG_FILE = Path(os.getenv("ZORAC_CONFIG_FILE", str(ZORAC_DIR / "config.json")))


def ensure_zorac_dir():
    """Ensure the zorac storage directory exists"""
    if not ZORAC_DIR.exists():
        try:
            ZORAC_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not create directory {ZORAC_DIR}: {e}[/yellow]")


def load_config() -> dict[str, Any]:
    """Load configuration from file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                config: dict[str, Any] = json.load(f)
                return config
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load config file: {e}[/yellow]")
    return {}


def save_config(config: dict[str, Any]) -> bool:
    """Save configuration to file"""
    try:
        ensure_zorac_dir()
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        console.print(f"[red]Error saving config file: {e}[/red]")
        return False


def get_setting(key: str, default: str) -> str:
    """Get setting with priority: Env Var > Config File > Default"""
    # 1. Environment Variable
    env_val = os.getenv(key)
    if env_val:
        return env_val

    # 2. Config File
    config = load_config()
    if key in config:
        return str(config[key])

    # 3. Default
    return default


def get_int_setting(key: str, default: int) -> int:
    """Get integer setting with priority: Env Var > Config File > Default"""
    value = get_setting(key, str(default))
    try:
        return int(value)
    except ValueError:
        console.print(
            f"[yellow]Warning: Invalid integer value for {key}: {value}, using default {default}[/yellow]"
        )
        return default


def get_float_setting(key: str, default: float) -> float:
    """Get float setting with priority: Env Var > Config File > Default"""
    value = get_setting(key, str(default))
    try:
        return float(value)
    except ValueError:
        console.print(
            f"[yellow]Warning: Invalid float value for {key}: {value}, using default {default}[/yellow]"
        )
        return default


def get_bool_setting(key: str, default: bool) -> bool:
    """Get boolean setting with priority: Env Var > Config File > Default"""
    value = get_setting(key, str(default).lower())
    return value.lower() in ("true", "1", "yes", "on")


# Initialize Configuration
VLLM_BASE_URL = get_setting("VLLM_BASE_URL", DEFAULT_CONFIG["VLLM_BASE_URL"]).strip().rstrip("/")
VLLM_API_KEY = get_setting("VLLM_API_KEY", DEFAULT_CONFIG["VLLM_API_KEY"])
VLLM_MODEL = get_setting("VLLM_MODEL", DEFAULT_CONFIG["VLLM_MODEL"])

# Token limits (configurable)
MAX_INPUT_TOKENS = get_int_setting("MAX_INPUT_TOKENS", 12000)
MAX_OUTPUT_TOKENS = get_int_setting("MAX_OUTPUT_TOKENS", 4000)
KEEP_RECENT_MESSAGES = get_int_setting("KEEP_RECENT_MESSAGES", 6)

# Model parameters (configurable)
TEMPERATURE = get_float_setting("TEMPERATURE", 0.1)
STREAM = get_bool_setting("STREAM", True)
