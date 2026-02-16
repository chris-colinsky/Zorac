"""
Test suite for Zorac — comprehensive unit and integration tests.

This test suite uses pytest with several plugins:
  - pytest-asyncio: Enables testing async functions with `async def test_*()`.
    Configured with `asyncio_mode = "auto"` in pyproject.toml, which means
    pytest automatically handles async test functions without needing the
    @pytest.mark.asyncio decorator.
  - pytest-mock: Provides the `mocker` fixture for creating mocks (though
    we mostly use unittest.mock directly here for explicit control).
  - pytest-cov: Generates code coverage reports showing which lines are tested.

Testing philosophy:
  - Each test class groups related tests around a single feature or module
  - Tests are independent — they don't depend on each other's execution order
  - External dependencies (file system, network, API calls) are mocked to
    ensure tests are fast, deterministic, and don't require a running server
  - We test both success paths and error/edge cases

Mocking strategy:
  - MagicMock: Used for objects with complex interfaces (like the OpenAI client)
    where we only need to stub specific methods
  - @patch: Used to temporarily replace module-level objects (like `console`)
    to prevent test output from polluting the terminal and to verify what
    would have been printed
  - tempfile: Used for file-based tests to avoid touching the user's real
    session/config files

Common patterns you'll see:
  - `@pytest.fixture(autouse=True)`: Setup/teardown that runs automatically
    for every test in the class
  - `ChatCompletionMessageParam`: OpenAI's type for chat messages, ensuring
    our test data matches the real API format
  - Async mock functions: `async def mock_create(...)` to simulate async
    API calls that the real code awaits
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from openai.types.chat import ChatCompletionMessageParam

from zorac import (
    KEEP_RECENT_MESSAGES,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    SESSION_FILE,
    TIKTOKEN_ENCODING,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
    count_tokens,
    ensure_zorac_dir,
    get_setting,
    load_session,
    print_header,
    save_config,
    save_session,
    summarize_old_messages,
)
from zorac.commands import get_help_text, get_system_prompt_commands
from zorac.main import get_initial_system_message


class TestCountTokens:
    """Test token counting functionality.

    Token counting is critical for context window management — if it's wrong,
    auto-summarization might trigger too early (losing context) or too late
    (causing API errors). These tests verify accuracy across different message
    types and edge cases.
    """

    def test_count_tokens_simple_message(self):
        """Test token counting with a simple single-word message.

        A message with "hello" should produce at least 6 tokens:
          - 4 tokens for message envelope overhead (<im_start>{role}\n{content}<im_end>\n)
          - 1 token for "hello"
          - 2 tokens for reply priming (<im_start>assistant)
        Total minimum: 7 tokens
        """
        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hello"}]
        count = count_tokens(messages)
        assert count > 0
        assert count > 5

    def test_count_tokens_empty_list(self):
        """Test token counting with no messages at all.

        Even with zero messages, there's a base overhead of 2 tokens for the
        reply priming (<im_start>assistant). This is a fixed cost that every
        conversation pays.
        """
        messages: list[ChatCompletionMessageParam] = []
        assert count_tokens(messages) == 2

    def test_count_tokens_multiple_messages(self):
        """Test token counting with a typical conversation (system + user + assistant).

        Verifies that multi-message conversations accumulate tokens correctly.
        Each message adds 4 overhead tokens plus the encoded content length.
        """
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = count_tokens(messages)
        # 3 messages × 4 overhead + content tokens + 2 base = well over 10
        assert count > 10

    def test_count_tokens_long_content(self):
        """Test token counting with substantial content.

        100 repetitions of "word" should produce significantly more tokens,
        verifying that the function scales correctly with content length.
        """
        long_content = " ".join(["word"] * 100)
        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": long_content}]
        count = count_tokens(messages)
        assert count > 100

    def test_count_tokens_with_list_content(self):
        """Test token counting with multipart messages (list-based content).

        The OpenAI API supports messages with multiple content parts (e.g.,
        text + images). Our token counter handles the text parts in these
        multipart messages. This test verifies that list-type content doesn't
        crash the counter and produces a valid token count.
        """
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            }
        ]
        count = count_tokens(messages)
        assert count > 0


class TestSessionManagement:
    """Test session save/load functionality.

    Sessions are stored as JSON files. These tests verify that conversations
    can be saved to disk and loaded back without data loss or corruption.
    All tests use temporary files to avoid touching the user's real session.
    """

    @pytest.fixture(autouse=True)
    def setup_temp_file(self):
        """Create a temporary file for testing, cleaned up after each test.

        autouse=True means this fixture runs automatically for every test in
        this class — no need to pass it as a parameter.

        The yield-based pattern is pytest's way of doing setup/teardown:
          - Code before yield = setup
          - Code after yield = teardown (runs even if the test fails)
        """
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()
        yield
        # Cleanup: remove the temp file if it still exists
        if self.temp_path.exists():
            self.temp_path.unlink()

    def test_save_session_success(self):
        """Test that save_session creates a file and returns True on success."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        result = save_session(messages, self.temp_path)
        assert result is True
        assert self.temp_path.exists()

    def test_save_session_content_verification(self):
        """Test that the saved JSON content matches the original messages.

        This is a critical test — it verifies data integrity by reading
        back the saved file and comparing the contents field by field.
        """
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Test message"},
        ]
        save_session(messages, self.temp_path)

        # Read the file directly (bypassing load_session) to verify raw content
        with open(self.temp_path) as f:
            saved_data = json.load(f)

        assert len(saved_data) == 1
        assert saved_data[0]["role"] == "user"
        assert saved_data[0]["content"] == "Test message"

    def test_load_session_success(self):
        """Test that load_session correctly restores a previously saved session."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
        ]
        save_session(messages, self.temp_path)
        loaded = load_session(self.temp_path)

        assert loaded is not None
        if loaded:
            assert len(loaded) == 2
            assert loaded[0]["role"] == "system"
            assert loaded[1].get("content") == "User message"

    def test_load_session_nonexistent_file(self):
        """Test that loading from a non-existent file returns None (not an error).

        This is the expected behavior on first run when no session file exists yet.
        """
        non_existent = Path("/tmp/this_file_does_not_exist_zorac_test.json")
        loaded = load_session(non_existent)
        assert loaded is None

    def test_load_session_invalid_json(self):
        """Test that loading corrupted JSON returns None instead of crashing.

        This handles edge cases like partially-written files (e.g., if the
        app was killed during a save) or manually edited files with syntax errors.
        """
        with open(self.temp_path, "w") as f:
            f.write("invalid json content {{{")

        loaded = load_session(self.temp_path)
        assert loaded is None

    def test_save_session_invalid_path(self):
        """Test that saving to an invalid path returns False instead of crashing.

        Verifies graceful error handling when the target directory doesn't exist
        (e.g., misconfigured ZORAC_SESSION_FILE environment variable).
        """
        invalid_path = Path("/invalid/directory/that/does/not/exist/session.json")
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Test"},
        ]
        result = save_session(messages, invalid_path)
        assert result is False


class TestPrintHeader:
    """Test the welcome header display.

    These tests mock the console to verify what would be printed without
    actually printing to the terminal. This keeps test output clean and
    allows us to inspect the rendered content programmatically.
    """

    @patch("zorac.utils.console")
    def test_print_header_calls_console(self, mock_console):
        """Test that print_header makes exactly 2 console.print calls.

        The header consists of two parts:
          1. The ASCII art logo
          2. The info panel (version, URL, model, commands)
        """
        print_header()
        assert mock_console.print.call_count == 2

    @patch("zorac.utils.console")
    def test_print_header_contains_url(self, mock_console):
        """Test that the header panel displays the configured server URL.

        Extracts the Panel renderable from the second print call and checks
        that the VLLM_BASE_URL appears in its text content.
        """
        print_header()
        # The second print call receives the Panel object
        panel = mock_console.print.call_args_list[1][0][0]
        panel_text = str(panel.renderable)
        assert VLLM_BASE_URL in panel_text

    @patch("zorac.utils.console")
    def test_print_header_contains_model(self, mock_console):
        """Test that the header panel displays the configured model name."""
        print_header()
        panel = mock_console.print.call_args_list[1][0][0]
        panel_text = str(panel.renderable)
        assert VLLM_MODEL in panel_text


class TestSummarizeOldMessages:
    """Test the conversation summarization feature.

    Summarization is the most complex feature — it involves async API calls,
    message list manipulation, and error handling. These tests verify the
    function's behavior in three scenarios:
      1. Too few messages (should skip summarization)
      2. Enough messages (should summarize and restructure)
      3. API error (should degrade gracefully)
    """

    async def test_summarize_skips_when_few_messages(self):
        """Test that summarization is a no-op when there aren't enough messages.

        When message count is <= KEEP_RECENT_MESSAGES + 1 (system message),
        there's nothing to summarize — all messages are "recent". The function
        should return the original list unchanged and make zero API calls.
        """
        mock_client = MagicMock()
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User 1"},
            {"role": "assistant", "content": "Assistant 1"},
        ]
        result = await summarize_old_messages(mock_client, messages)
        assert result == messages  # Unchanged
        assert mock_client.chat.completions.create.call_count == 0  # No API call made

    async def test_summarize_keeps_system_and_recent(self):
        """Test that summarization preserves the system message and recent messages.

        After summarization, the message list should have:
          - messages[0]: Original system message
          - messages[1]: New summary message (role=system, prefixed with "Previous conversation summary:")
          - messages[2:]: The KEEP_RECENT_MESSAGES most recent messages
        Total: KEEP_RECENT_MESSAGES + 2
        """
        # Create a mock client that returns a canned summary
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary of conversation"

        # The real client.chat.completions.create is async, so our mock must be too
        async def mock_create(*args, **kwargs):
            return mock_response

        mock_client.chat.completions.create = mock_create

        # Build a message list with enough messages to trigger summarization
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System"},
        ]
        for i in range(KEEP_RECENT_MESSAGES + 5):
            messages.append({"role": "user", "content": f"Message {i}"})

        result = await summarize_old_messages(mock_client, messages)

        # Verify structure: system + summary + KEEP_RECENT_MESSAGES recent
        assert len(result) == KEEP_RECENT_MESSAGES + 2
        assert result[0]["role"] == "system"  # Original system message
        assert result[1]["role"] == "system"  # Summary message
        content = result[1].get("content")
        assert isinstance(content, str)
        if isinstance(content, str):
            assert "Previous conversation summary" in content

    @patch("zorac.llm.console")
    async def test_summarize_handles_api_error(self, mock_console):
        """Test graceful degradation when the summarization API call fails.

        When the LLM API call fails (network error, server down, etc.), the
        function should fall back to keeping just the system message + recent
        messages. This loses the old context but keeps the app functional.
        """
        mock_client = MagicMock()

        # Simulate an API failure
        async def mock_create_fail(*args, **kwargs):
            raise Exception("API Error")

        mock_client.chat.completions.create = mock_create_fail

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System"},
        ]
        for i in range(KEEP_RECENT_MESSAGES + 5):
            messages.append({"role": "user", "content": f"Message {i}"})

        result = await summarize_old_messages(mock_client, messages)

        # Should have system message + recent messages (no summary since API failed)
        assert len(result) == KEEP_RECENT_MESSAGES + 1
        assert result[0]["role"] == "system"


class TestConfiguration:
    """Test configuration loading and default values.

    These tests verify that the configuration system produces valid values
    and that the three-tier priority system (env > config file > default)
    works correctly.
    """

    def test_default_values_exist(self):
        """Test that all essential configuration constants are initialized.

        These module-level constants are resolved at import time. If any
        configuration source is broken, these might be None or the wrong type.
        """
        assert isinstance(VLLM_BASE_URL, str)
        assert isinstance(VLLM_API_KEY, str)
        assert isinstance(VLLM_MODEL, str)
        assert isinstance(SESSION_FILE, Path)
        assert isinstance(TIKTOKEN_ENCODING, str)

    def test_token_limits_are_positive(self):
        """Test that token limits are positive integers.

        Zero or negative token limits would break the context management logic
        (e.g., infinite summarization loop or division by zero).
        """
        assert MAX_INPUT_TOKENS > 0
        assert MAX_OUTPUT_TOKENS > 0
        assert KEEP_RECENT_MESSAGES > 0

    def test_get_setting_priority(self):
        """Test the three-tier priority system: Env Var > Config File > Default.

        Verifies each priority level by progressively adding higher-priority
        sources and checking that they override lower-priority ones:
          1. No env var, no config file → returns default
          2. Config file exists → config file wins over default
          3. Env var set + config file exists → env var wins over both
        """
        key = "TEST_SETTING"
        default = "default_value"

        # Priority 3: Default (no env var, no config entry)
        assert get_setting(key, default) == default

        # Priority 2: Config file overrides default
        with patch("zorac.config.load_config", return_value={key: "config_value"}):
            assert get_setting(key, default) == "config_value"

        # Priority 1: Environment variable overrides both config file and default
        with (
            patch.dict(os.environ, {key: "env_value"}),
            patch("zorac.config.load_config", return_value={key: "config_value"}),
        ):
            assert get_setting(key, default) == "env_value"


class TestIntegration:
    """Integration tests that verify multiple modules working together.

    Unlike unit tests (which test individual functions in isolation), these
    tests exercise the interaction between modules — specifically, the
    save/load roundtrip that involves both session.py and the file system.
    """

    @pytest.fixture(autouse=True)
    def setup_temp_file(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()
        yield
        if self.temp_path.exists():
            self.temp_path.unlink()

    def test_save_and_load_roundtrip(self):
        """Test that data survives a complete save → load cycle without corruption.

        This is the most important integration test: it verifies that
        save_session() and load_session() are compatible with each other,
        and that no data is lost or modified during serialization/deserialization.
        """
        original_messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        save_session(original_messages, self.temp_path)
        loaded_messages = load_session(self.temp_path)

        assert loaded_messages is not None
        if loaded_messages is not None:
            assert len(loaded_messages) == len(original_messages)
            for i, msg in enumerate(loaded_messages):
                assert msg["role"] == original_messages[i]["role"]
                assert msg.get("content") == original_messages[i].get("content")


class TestHelpFeatureIntegration:
    """Integration tests for the command help system.

    These tests verify that the command registry (commands.py) integrates
    correctly with the system message builder (main.py) and that the output
    meets formatting and content requirements.
    """

    def test_get_initial_system_message_includes_commands(self):
        """Test that the system message contains both identity and command info."""
        system_message = get_initial_system_message()
        assert "You are Zorac, a helpful AI assistant." in system_message
        assert "Today's date is" in system_message
        assert "Available Commands:" in system_message

    def test_help_text_formatted_correctly(self):
        """Test that help text uses Rich markup for terminal formatting.

        The help text should contain Rich markup tags ([cyan], [bold]) that
        will be rendered as colors and styles in the terminal.
        """
        help_text = get_help_text()
        assert "[cyan]" in help_text
        assert "[bold]" in help_text
        assert "Available Commands:" in help_text

    def test_get_initial_system_message_includes_all_commands(self):
        """Test that the system message mentions every command by name.

        If a command is missing from the system prompt, the LLM won't know
        about it and can't suggest it to users.
        """
        system_message = get_initial_system_message()
        assert "/help" in system_message
        assert "/quit" in system_message
        assert "/clear" in system_message
        assert "/save" in system_message
        assert "/config" in system_message

    def test_system_prompt_commands_no_rich_formatting(self):
        """Test that the system prompt text is plain (no Rich markup).

        Rich markup like [cyan] would confuse the LLM and waste tokens.
        The system prompt should use plain text only.
        """
        prompt_commands = get_system_prompt_commands()
        assert "[cyan]" not in prompt_commands
        assert "[bold]" not in prompt_commands
        assert "/help" in prompt_commands

    def test_help_command_displays_output(self):
        """Test that get_help_text returns substantial, complete output.

        Verifies that the help text is not empty/truncated and contains
        key commands that should always be present.
        """
        help_output = get_help_text()
        assert isinstance(help_output, str)
        assert len(help_output) > 100  # Not trivially small
        assert "/help" in help_output
        assert "/quit, /exit" in help_output
        assert "Available Commands:" in help_output

    def test_system_message_token_overhead_reasonable(self):
        """Test that the system message doesn't consume too many tokens.

        The system message is always present and counts against the token
        limit. If it grows too large (>600 tokens), it would significantly
        reduce the space available for actual conversation. This test ensures
        the command information doesn't bloat the system message.
        """
        system_message = get_initial_system_message()
        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_message}]
        token_count = count_tokens(messages)
        assert token_count < 600, "System message with commands uses too many tokens"
        assert token_count > 50  # Should have meaningful content


class TestSummarizeFormat:
    """Test the format of summarization output messages.

    These tests verify the specific format of summary messages, which is
    important because other parts of the code (like /summary command) rely
    on the "Previous conversation summary:" prefix to identify summaries.
    """

    async def test_summarize_creates_proper_summary_format(self):
        """Test that the summary message uses the expected prefix format.

        The /summary command looks for messages starting with
        "Previous conversation summary:" — if summarization produces a
        different format, the /summary command would fail to find it.
        """
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary of conversation"

        async def mock_create(*args, **kwargs):
            return mock_response

        mock_client.chat.completions.create = mock_create

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System"},
        ]
        for i in range(KEEP_RECENT_MESSAGES + 5):
            messages.append({"role": "user", "content": f"Message {i}"})

        result = await summarize_old_messages(mock_client, messages)

        # Verify the summary message format
        assert result[1]["role"] == "system"
        content = result[1].get("content")
        assert isinstance(content, str)
        if isinstance(content, str):
            assert content.startswith("Previous conversation summary:")
            assert "Summary of conversation" in content

    def test_summary_extraction_from_messages(self):
        """Test that summary text can be correctly identified and extracted.

        This simulates what the /summary command does: it looks at messages[1],
        checks if it's a system message with the summary prefix, and extracts
        the actual summary text by removing the prefix.
        """
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "system",
                "content": "Previous conversation summary: This is the summary text.",
            },
            {"role": "user", "content": "Recent message"},
        ]

        # Verify the summary detection logic (matches cmd_summary implementation)
        content = messages[1].get("content", "")
        has_summary = (
            len(messages) > 1
            and messages[1].get("role") == "system"
            and isinstance(content, str)
            and content.startswith("Previous conversation summary:")
        )
        assert has_summary

        # Verify summary text extraction
        summary_content = messages[1].get("content")
        assert isinstance(summary_content, str)
        if isinstance(summary_content, str):
            summary_text = summary_content.replace("Previous conversation summary:", "").strip()
            assert summary_text == "This is the summary text."


class TestDirectoryManagement:
    """Test the ~/.zorac/ directory creation logic.

    Uses @patch to mock the ZORAC_DIR Path object, avoiding actual filesystem
    operations during testing. This ensures tests are fast and don't create
    real directories.
    """

    @patch("zorac.config.ZORAC_DIR")
    def test_ensure_zorac_dir_creates_if_missing(self, mock_dir):
        """Test that ensure_zorac_dir creates the directory when it doesn't exist."""
        mock_dir.exists.return_value = False
        ensure_zorac_dir()
        mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("zorac.config.ZORAC_DIR")
    def test_ensure_zorac_dir_does_nothing_if_exists(self, mock_dir):
        """Test that ensure_zorac_dir is a no-op when the directory already exists.

        This is important for performance — ensure_zorac_dir is called before
        every save operation, so it must be cheap when the directory exists.
        """
        mock_dir.exists.return_value = True
        ensure_zorac_dir()
        mock_dir.mkdir.assert_not_called()


class TestConfigurationExtended:
    """Extended configuration tests for boundary conditions and persistence."""

    def test_keep_recent_messages_reasonable(self):
        """Test that KEEP_RECENT_MESSAGES is within a sensible range.

        Too low (< 3): Would discard nearly all context during summarization
        Too high (> 20): Would keep too many messages, making summarization
        less effective at reducing token count
        """
        assert KEEP_RECENT_MESSAGES > 2
        assert KEEP_RECENT_MESSAGES < 20

    @patch("zorac.config.CONFIG_FILE", new_callable=MagicMock)
    @patch("zorac.config.ensure_zorac_dir")
    def test_save_config_creates_dir(self, mock_ensure_dir, _mock_config_path):
        """Test that save_config calls ensure_zorac_dir before writing.

        This verifies that the config file can be saved even on first run
        when ~/.zorac/ doesn't exist yet.
        """
        m = mock_open()
        with patch("builtins.open", m):
            save_config({"key": "value"})
            mock_ensure_dir.assert_called_once()


class TestTokenCountAfterSaveLoad:
    """Test that token counts are consistent across save/load cycles.

    If serialization changes the message format (e.g., adding/removing fields),
    token counts might differ before and after save/load. This would cause
    inconsistent behavior in auto-summarization triggers.
    """

    @pytest.fixture(autouse=True)
    def setup_temp_file(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()
        yield
        if self.temp_path.exists():
            self.temp_path.unlink()

    def test_token_count_after_save_load(self):
        """Test that token count is identical before and after a save/load cycle.

        This is a critical invariant: count_tokens(messages) should return
        the same value before serialization and after deserialization.
        """
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
        ]

        original_count = count_tokens(messages)
        save_session(messages, self.temp_path)
        loaded_messages = load_session(self.temp_path)

        if loaded_messages is not None:
            loaded_count = count_tokens(loaded_messages)
            assert original_count == loaded_count


class TestStatsBar:
    """Test the stats bar display logic.

    The stats bar shows contextual stats: "Ready" before any chat,
    session info when messages are loaded, and full performance stats
    after a chat interaction. Tests use Textual's async test framework.
    """

    async def test_stats_bar_initial_state(self):
        """Test that the stats bar shows 'Ready' before any chat interaction."""
        from textual.widgets import Static

        from zorac.main import ZoracApp

        app = ZoracApp()
        async with app.run_test(size=(80, 24)):
            stats_bar = app.query_one("#stats-bar", Static)
            text = stats_bar.content
            assert "Ready" in text

    async def test_stats_bar_after_stats_update(self):
        """Test that the stats bar shows full stats after updating."""
        from textual.widgets import Static

        from zorac.main import ZoracApp

        app = ZoracApp()
        async with app.run_test(size=(80, 24)):
            app.stats = {
                "tokens": 42,
                "duration": 1.5,
                "tps": 28.0,
                "total_msgs": 5,
                "current_tokens": 800,
            }
            app._update_stats_bar()
            stats_bar = app.query_one("#stats-bar", Static)
            text = stats_bar.content
            assert "42 tokens" in text
            assert "1.5s" in text
            assert "28.0 tok/s" in text
            assert "5 msgs" in text
            assert "800/12000" in text


class TestCheckConnectionAsync:
    """Test the async server connection verification.

    These tests verify both the success and failure paths of the connection
    check, using async mock functions to simulate the OpenAI client's behavior.
    """

    @patch("zorac.utils.console")
    async def test_check_connection_success(self, mock_console):
        """Test that a successful connection check returns True.

        Simulates a healthy vLLM server by mocking client.models.list()
        to return successfully (the actual return value doesn't matter —
        we just need it to not throw an exception).
        """
        from zorac.utils import check_connection

        mock_client = MagicMock()

        # Mock the async models.list() call
        async def mock_list():
            return []

        mock_client.models.list = mock_list

        result = await check_connection(mock_client)
        assert result is True

    @patch("zorac.utils.console")
    async def test_check_connection_failure(self, mock_console):
        """Test that a failed connection check returns False (not an exception).

        Simulates a down or unreachable server by making client.models.list()
        raise a ConnectionError. The function should catch the error, display
        a helpful message, and return False.
        """
        from zorac.utils import check_connection

        mock_client = MagicMock()
        mock_client.base_url = "http://localhost:8000/v1"

        # Mock the async models.list() call to simulate connection failure
        async def mock_list():
            raise ConnectionError("Connection refused")

        mock_client.models.list = mock_list

        result = await check_connection(mock_client)
        assert result is False
