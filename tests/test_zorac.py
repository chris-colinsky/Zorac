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
    """Test token counting functionality"""

    def test_count_tokens_simple_message(self):
        """Test token counting with a simple message"""
        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hello"}]
        count = count_tokens(messages)
        assert count > 0
        # Should be at least 6 tokens (4 overhead + 1 for 'hello' + 2 base)
        assert count > 5

    def test_count_tokens_empty_list(self):
        """Test token counting with empty message list"""
        messages: list[ChatCompletionMessageParam] = []
        # Base overhead for reply prime
        assert count_tokens(messages) == 2

    def test_count_tokens_multiple_messages(self):
        """Test token counting with multiple messages"""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = count_tokens(messages)
        assert count > 10

    def test_count_tokens_long_content(self):
        """Test token counting with long content"""
        long_content = " ".join(["word"] * 100)
        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": long_content}]
        count = count_tokens(messages)
        # Should be significantly more tokens
        assert count > 100

    def test_count_tokens_with_list_content(self):
        """Test token counting with list-based content (multipart messages)"""
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
    """Test session save/load functionality"""

    @pytest.fixture(autouse=True)
    def setup_temp_file(self):
        """Create a temporary file for testing"""
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()
        yield
        if self.temp_path.exists():
            self.temp_path.unlink()

    def test_save_session_success(self):
        """Test successful session save"""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        result = save_session(messages, self.temp_path)
        assert result is True
        assert self.temp_path.exists()

    def test_save_session_content_verification(self):
        """Test that saved session content is correct"""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Test message"},
        ]
        save_session(messages, self.temp_path)

        with open(self.temp_path) as f:
            saved_data = json.load(f)

        assert len(saved_data) == 1
        assert saved_data[0]["role"] == "user"
        assert saved_data[0]["content"] == "Test message"

    def test_load_session_success(self):
        """Test successful session load"""
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
        """Test loading from non-existent file"""
        non_existent = Path("/tmp/this_file_does_not_exist_zorac_test.json")
        loaded = load_session(non_existent)
        assert loaded is None

    def test_load_session_invalid_json(self):
        """Test loading from file with invalid JSON"""
        with open(self.temp_path, "w") as f:
            f.write("invalid json content {{{")

        loaded = load_session(self.temp_path)
        assert loaded is None

    def test_save_session_invalid_path(self):
        """Test saving to invalid path"""
        invalid_path = Path("/invalid/directory/that/does/not/exist/session.json")
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Test"},
        ]
        result = save_session(messages, invalid_path)
        assert result is False


class TestPrintHeader:
    """Test header printing functionality"""

    @patch("zorac.utils.console")
    def test_print_header_calls_console(self, mock_console):
        """Test that print_header calls console.print twice (logo + panel)"""
        print_header()
        # Should be called twice: once for logo, once for panel
        assert mock_console.print.call_count == 2

    @patch("zorac.utils.console")
    def test_print_header_contains_url(self, mock_console):
        """Test that header contains the vLLM URL"""
        print_header()
        panel = mock_console.print.call_args_list[1][0][0]
        panel_text = str(panel.renderable)
        assert VLLM_BASE_URL in panel_text

    @patch("zorac.utils.console")
    def test_print_header_contains_model(self, mock_console):
        """Test that header contains the model name"""
        print_header()
        panel = mock_console.print.call_args_list[1][0][0]
        panel_text = str(panel.renderable)
        assert VLLM_MODEL in panel_text


class TestSummarizeOldMessages:
    """Test message summarization functionality"""

    async def test_summarize_skips_when_few_messages(self):
        """Test that summarization is skipped when message count is low"""
        mock_client = MagicMock()
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User 1"},
            {"role": "assistant", "content": "Assistant 1"},
        ]
        result = await summarize_old_messages(mock_client, messages)
        assert result == messages
        assert mock_client.chat.completions.create.call_count == 0

    async def test_summarize_keeps_system_and_recent(self):
        """Test that summarization keeps system message and recent messages"""
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

        assert len(result) == KEEP_RECENT_MESSAGES + 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "system"  # Summary
        content = result[1].get("content")
        assert isinstance(content, str)
        if isinstance(content, str):
            assert "Previous conversation summary" in content

    @patch("zorac.llm.console")
    async def test_summarize_handles_api_error(self, mock_console):
        """Test that summarization handles API errors gracefully"""
        mock_client = MagicMock()

        async def mock_create_fail(*args, **kwargs):
            raise Exception("API Error")

        mock_client.chat.completions.create = mock_create_fail

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System"},
        ]
        for i in range(KEEP_RECENT_MESSAGES + 5):
            messages.append({"role": "user", "content": f"Message {i}"})

        result = await summarize_old_messages(mock_client, messages)

        assert len(result) == KEEP_RECENT_MESSAGES + 1
        assert result[0]["role"] == "system"


class TestConfiguration:
    """Test configuration loading and defaults"""

    def test_default_values_exist(self):
        """Test that default configuration values are set"""
        assert isinstance(VLLM_BASE_URL, str)
        assert isinstance(VLLM_API_KEY, str)
        assert isinstance(VLLM_MODEL, str)
        assert isinstance(SESSION_FILE, Path)
        assert isinstance(TIKTOKEN_ENCODING, str)

    def test_token_limits_are_positive(self):
        """Test that token limits are positive integers"""
        assert MAX_INPUT_TOKENS > 0
        assert MAX_OUTPUT_TOKENS > 0
        assert KEEP_RECENT_MESSAGES > 0

    def test_get_setting_priority(self):
        """Test that get_setting prioritizes Env > Config > Default"""
        key = "TEST_SETTING"
        default = "default_value"

        assert get_setting(key, default) == default

        with patch("zorac.config.load_config", return_value={key: "config_value"}):
            assert get_setting(key, default) == "config_value"

        with (
            patch.dict(os.environ, {key: "env_value"}),
            patch("zorac.config.load_config", return_value={key: "config_value"}),
        ):
            assert get_setting(key, default) == "env_value"


class TestIntegration:
    """Integration tests combining multiple functions"""

    @pytest.fixture(autouse=True)
    def setup_temp_file(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()
        yield
        if self.temp_path.exists():
            self.temp_path.unlink()

    def test_save_and_load_roundtrip(self):
        """Test that save and load work together correctly"""
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
    """Integration tests for the help feature"""

    def test_get_initial_system_message_includes_commands(self):
        """Test that initial system message includes command information"""
        system_message = get_initial_system_message()
        assert "You are Zorac, a helpful AI assistant." in system_message
        assert "Today's date is" in system_message
        assert "Available Commands:" in system_message

    def test_help_text_formatted_correctly(self):
        """Test that help text is formatted for console display"""
        help_text = get_help_text()
        assert "[cyan]" in help_text
        assert "[bold]" in help_text
        assert "Available Commands:" in help_text

    def test_get_initial_system_message_includes_all_commands(self):
        """Test that initial system message includes all command names"""
        system_message = get_initial_system_message()
        assert "/help" in system_message
        assert "/quit" in system_message
        assert "/clear" in system_message
        assert "/save" in system_message
        assert "/config" in system_message

    def test_system_prompt_commands_no_rich_formatting(self):
        """Test that system prompt doesn't include Rich formatting"""
        prompt_commands = get_system_prompt_commands()
        assert "[cyan]" not in prompt_commands
        assert "[bold]" not in prompt_commands
        assert "/help" in prompt_commands

    def test_help_command_displays_output(self):
        """Test that get_help_text returns valid, complete output"""
        help_output = get_help_text()
        assert isinstance(help_output, str)
        assert len(help_output) > 100
        assert "/help" in help_output
        assert "/quit, /exit" in help_output
        assert "Available Commands:" in help_output

    def test_system_message_token_overhead_reasonable(self):
        """Test that enhanced system message doesn't add excessive tokens"""
        system_message = get_initial_system_message()
        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_message}]
        token_count = count_tokens(messages)
        assert token_count < 500, "System message with commands uses too many tokens"
        assert token_count > 50


class TestSummarizeFormat:
    """Test summarization message format"""

    async def test_summarize_creates_proper_summary_format(self):
        """Test that summarization creates a summary message with the expected prefix"""
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

        assert result[1]["role"] == "system"
        content = result[1].get("content")
        assert isinstance(content, str)
        if isinstance(content, str):
            assert content.startswith("Previous conversation summary:")
            assert "Summary of conversation" in content

    def test_summary_extraction_from_messages(self):
        """Test that we can identify and extract summary from message list"""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "system",
                "content": "Previous conversation summary: This is the summary text.",
            },
            {"role": "user", "content": "Recent message"},
        ]

        content = messages[1].get("content", "")
        has_summary = (
            len(messages) > 1
            and messages[1].get("role") == "system"
            and isinstance(content, str)
            and content.startswith("Previous conversation summary:")
        )
        assert has_summary

        summary_content = messages[1].get("content")
        assert isinstance(summary_content, str)
        if isinstance(summary_content, str):
            summary_text = summary_content.replace("Previous conversation summary:", "").strip()
            assert summary_text == "This is the summary text."


class TestDirectoryManagement:
    """Test directory creation logic"""

    @patch("zorac.config.ZORAC_DIR")
    def test_ensure_zorac_dir_creates_if_missing(self, mock_dir):
        """Test that ensure_zorac_dir creates directory if it doesn't exist"""
        mock_dir.exists.return_value = False
        ensure_zorac_dir()
        mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("zorac.config.ZORAC_DIR")
    def test_ensure_zorac_dir_does_nothing_if_exists(self, mock_dir):
        """Test that ensure_zorac_dir does nothing if directory exists"""
        mock_dir.exists.return_value = True
        ensure_zorac_dir()
        mock_dir.mkdir.assert_not_called()


class TestConfigurationExtended:
    """Extended configuration tests"""

    def test_keep_recent_messages_reasonable(self):
        """Test that KEEP_RECENT_MESSAGES is a reasonable value"""
        assert KEEP_RECENT_MESSAGES > 2
        assert KEEP_RECENT_MESSAGES < 20

    @patch("zorac.config.CONFIG_FILE", new_callable=MagicMock)
    @patch("zorac.config.ensure_zorac_dir")
    def test_save_config_creates_dir(self, mock_ensure_dir, _mock_config_path):
        """Test that save_config calls ensure_zorac_dir"""
        m = mock_open()
        with patch("builtins.open", m):
            save_config({"key": "value"})
            mock_ensure_dir.assert_called_once()


class TestTokenCountAfterSaveLoad:
    """Test token count consistency across save/load"""

    @pytest.fixture(autouse=True)
    def setup_temp_file(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()
        yield
        if self.temp_path.exists():
            self.temp_path.unlink()

    def test_token_count_after_save_load(self):
        """Test that token count is consistent after save/load"""
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


class TestCheckConnectionAsync:
    """Test async check_connection function"""

    @patch("zorac.utils.console")
    async def test_check_connection_success(self, mock_console):
        """Test successful connection check"""
        from zorac.utils import check_connection

        mock_client = MagicMock()

        async def mock_list():
            return []

        mock_client.models.list = mock_list

        result = await check_connection(mock_client)
        assert result is True

    @patch("zorac.utils.console")
    async def test_check_connection_failure(self, mock_console):
        """Test failed connection check"""
        from zorac.utils import check_connection

        mock_client = MagicMock()
        mock_client.base_url = "http://localhost:8000/v1"

        async def mock_list():
            raise ConnectionError("Connection refused")

        mock_client.models.list = mock_list

        result = await check_connection(mock_client)
        assert result is False
