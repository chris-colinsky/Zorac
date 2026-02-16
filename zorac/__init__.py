"""
Zorac - Interactive CLI chat client for vLLM inference servers.

This __init__.py file defines the public API of the zorac package. It controls
what symbols are available when users write `from zorac import ...`.

Design choices:
  - We re-export key symbols from submodules so consumers (including tests)
    can import directly from `zorac` without knowing the internal module structure.
    For example: `from zorac import count_tokens` instead of `from zorac.utils import count_tokens`.
  - The `__all__` list explicitly declares every public symbol. This serves two purposes:
    1. It documents the public API — anything not in __all__ is considered internal.
    2. It controls what `from zorac import *` exports (though star imports are discouraged).
  - Symbols are grouped by their source module (Config, Console, LLM, Session, Utils)
    to make it easy to find where each symbol is defined.
"""

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
    TIKTOKEN_ENCODING,
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
    # Config — server settings, token limits, and configuration management
    "CONFIG_FILE",
    "DEFAULT_CONFIG",
    "HISTORY_FILE",
    "KEEP_RECENT_MESSAGES",
    "MAX_INPUT_TOKENS",
    "MAX_OUTPUT_TOKENS",
    "SESSION_FILE",
    "STREAM",
    "TEMPERATURE",
    "TIKTOKEN_ENCODING",
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
    # Console — shared Rich terminal output instance
    "console",
    # LLM — language model interaction and context management
    "summarize_old_messages",
    # Session — conversation persistence (save/load to disk)
    "load_session",
    "save_session",
    # Utils — token counting, UI header, connection verification
    "check_connection",
    "count_tokens",
    "print_header",
]
