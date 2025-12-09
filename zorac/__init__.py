"""Zorac - Interactive CLI chat client for vLLM inference servers"""

from .config import (
    CONFIG_FILE,
    DEFAULT_CONFIG,
    HISTORY_FILE,
    KEEP_RECENT_MESSAGES,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    SESSION_FILE,
    STREAM,
    TEMPERATURE,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
    ZORAC_DIR,
    ensure_zorac_dir,
    get_bool_setting,
    get_float_setting,
    get_int_setting,
    get_setting,
    load_config,
    save_config,
)
from .console import console
from .llm import summarize_old_messages
from .session import load_session, save_session
from .utils import check_connection, count_tokens, print_header

__all__ = [
    # Config
    "CONFIG_FILE",
    "DEFAULT_CONFIG",
    "HISTORY_FILE",
    "KEEP_RECENT_MESSAGES",
    "MAX_INPUT_TOKENS",
    "MAX_OUTPUT_TOKENS",
    "SESSION_FILE",
    "STREAM",
    "TEMPERATURE",
    "VLLM_API_KEY",
    "VLLM_BASE_URL",
    "VLLM_MODEL",
    "ZORAC_DIR",
    "ensure_zorac_dir",
    "get_bool_setting",
    "get_float_setting",
    "get_int_setting",
    "get_setting",
    "load_config",
    "save_config",
    # Console
    "console",
    # LLM
    "summarize_old_messages",
    # Session
    "load_session",
    "save_session",
    # Utils
    "check_connection",
    "count_tokens",
    "print_header",
]
