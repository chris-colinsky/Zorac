import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletionMessageParam

from zorac import (
    KEEP_RECENT_MESSAGES,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    SESSION_FILE,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
    count_tokens,
    load_session,
    print_header,
    save_session,
    summarize_old_messages,
)


class TestCountTokens(unittest.TestCase):
    """Test token counting functionality"""

    def test_count_tokens_simple_message(self):
        """Test token counting with a simple message"""
        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "hello"}]
        count = count_tokens(messages)
        self.assertGreater(count, 0)
        # Should be at least 6 tokens (4 overhead + 1 for 'hello' + 2 base)
        self.assertGreater(count, 5)

    def test_count_tokens_empty_list(self):
        """Test token counting with empty message list"""
        messages: list[ChatCompletionMessageParam] = []
        # Base overhead for reply prime
        self.assertEqual(count_tokens(messages), 2)

    def test_count_tokens_multiple_messages(self):
        """Test token counting with multiple messages"""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = count_tokens(messages)
        self.assertGreater(count, 10)

    def test_count_tokens_long_content(self):
        """Test token counting with long content"""
        long_content = " ".join(["word"] * 100)
        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": long_content}]
        count = count_tokens(messages)
        # Should be significantly more tokens
        self.assertGreater(count, 100)

    def test_count_tokens_with_list_content(self):
        """Test token counting with list-based content (multipart messages)"""
        # Create explicit objects or cast to satisfy strict typing
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
        self.assertGreater(count, 0)


class TestSessionManagement(unittest.TestCase):
    """Test session save/load functionality"""

    def setUp(self):
        """Create a temporary file for testing"""
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")  # noqa: SIM115
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary file"""
        if self.temp_path.exists():
            self.temp_path.unlink()

    def test_save_session_success(self):
        """Test successful session save"""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        result = save_session(messages, self.temp_path)
        self.assertTrue(result)
        self.assertTrue(self.temp_path.exists())

    def test_save_session_content_verification(self):
        """Test that saved session content is correct"""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Test message"},
        ]
        save_session(messages, self.temp_path)

        with open(self.temp_path) as f:
            saved_data = json.load(f)

        self.assertEqual(len(saved_data), 1)
        self.assertEqual(saved_data[0]["role"], "user")
        self.assertEqual(saved_data[0]["content"], "Test message")

    def test_load_session_success(self):
        """Test successful session load"""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
        ]
        save_session(messages, self.temp_path)
        loaded = load_session(self.temp_path)

        self.assertIsNotNone(loaded)
        if loaded:  # Type guard
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["role"], "system")
            self.assertEqual(loaded[1].get("content"), "User message")

    def test_load_session_nonexistent_file(self):
        """Test loading from non-existent file"""
        non_existent = Path("/tmp/this_file_does_not_exist_zorac_test.json")
        loaded = load_session(non_existent)
        self.assertIsNone(loaded)

    def test_load_session_invalid_json(self):
        """Test loading from file with invalid JSON"""
        with open(self.temp_path, "w") as f:
            f.write("invalid json content {{{")

        loaded = load_session(self.temp_path)
        self.assertIsNone(loaded)

    def test_save_session_invalid_path(self):
        """Test saving to invalid path"""
        invalid_path = Path("/invalid/directory/that/does/not/exist/session.json")
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Test"},
        ]
        result = save_session(messages, invalid_path)
        self.assertFalse(result)


class TestPrintHeader(unittest.TestCase):
    """Test header printing functionality"""

    @patch("zorac.console")
    def test_print_header_calls_console(self, mock_console):
        """Test that print_header calls console.print twice (logo + panel)"""
        print_header()
        # Should be called twice: once for logo, once for panel
        self.assertEqual(mock_console.print.call_count, 2)

    @patch("zorac.console")
    def test_print_header_contains_url(self, mock_console):
        """Test that header contains the vLLM URL"""
        print_header()
        # Get the Panel object from the second call (first is logo, second is panel)
        panel = mock_console.print.call_args_list[1][0][0]
        # Extract the text content from the panel
        panel_text = str(panel.renderable)
        self.assertIn(VLLM_BASE_URL, panel_text)

    @patch("zorac.console")
    def test_print_header_contains_model(self, mock_console):
        """Test that header contains the model name"""
        print_header()
        # Get the Panel object from the second call (first is logo, second is panel)
        panel = mock_console.print.call_args_list[1][0][0]
        # Extract the text content from the panel
        panel_text = str(panel.renderable)
        self.assertIn(VLLM_MODEL, panel_text)


class TestSummarizeOldMessages(unittest.TestCase):
    """Test message summarization functionality"""

    def setUp(self):
        """Set up mock OpenAI client"""
        self.mock_client = MagicMock()
        self.mock_response = MagicMock()
        self.mock_response.choices = [MagicMock()]
        self.mock_response.choices[0].message.content = "Summary of conversation"
        self.mock_client.chat.completions.create.return_value = self.mock_response

    def test_summarize_skips_when_few_messages(self):
        """Test that summarization is skipped when message count is low"""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User 1"},
            {"role": "assistant", "content": "Assistant 1"},
        ]
        result = summarize_old_messages(self.mock_client, messages)
        self.assertEqual(result, messages)
        self.mock_client.chat.completions.create.assert_not_called()

    def test_summarize_keeps_system_and_recent(self):
        """Test that summarization keeps system message and recent messages"""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System"},
        ]
        # Add more than KEEP_RECENT_MESSAGES + 1 messages
        for i in range(KEEP_RECENT_MESSAGES + 5):
            messages.append({"role": "user", "content": f"Message {i}"})

        result = summarize_old_messages(self.mock_client, messages)

        # Should have: system message + summary + KEEP_RECENT_MESSAGES
        self.assertEqual(len(result), KEEP_RECENT_MESSAGES + 2)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1]["role"], "system")  # Summary
        content = result[1].get("content")
        self.assertIsInstance(content, str)
        if isinstance(content, str):
            self.assertIn("Previous conversation summary", content)

    @patch("zorac.console")
    def test_summarize_handles_api_error(self, mock_console):
        """Test that summarization handles API errors gracefully"""
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System"},
        ]
        for i in range(KEEP_RECENT_MESSAGES + 5):
            messages.append({"role": "user", "content": f"Message {i}"})

        result = summarize_old_messages(self.mock_client, messages)

        # Should return system + recent messages on error
        self.assertEqual(len(result), KEEP_RECENT_MESSAGES + 1)
        self.assertEqual(result[0]["role"], "system")

    def test_summarize_creates_proper_summary_format(self):
        """Test that summarization creates a summary message with the expected prefix"""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "System"},
        ]
        # Add more than KEEP_RECENT_MESSAGES + 1 messages
        for i in range(KEEP_RECENT_MESSAGES + 5):
            messages.append({"role": "user", "content": f"Message {i}"})

        result = summarize_old_messages(self.mock_client, messages)

        # Verify summary message format (used by /summary command)
        self.assertEqual(result[1]["role"], "system")
        content = result[1].get("content")
        self.assertIsInstance(content, str)
        if isinstance(content, str):
            self.assertTrue(content.startswith("Previous conversation summary:"))
            self.assertIn("Summary of conversation", content)

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

        # Verify we can identify the summary (simulating /summary command logic)
        content = messages[1].get("content", "")
        # Cast to str or check instance to satisfy type checker for startswith
        has_summary = (
            len(messages) > 1
            and messages[1].get("role") == "system"
            and isinstance(content, str)
            and content.startswith("Previous conversation summary:")
        )
        self.assertTrue(has_summary)

        # Extract summary text
        summary_content = messages[1].get("content")
        self.assertIsInstance(summary_content, str)
        # Type checker knows summary_content is str here due to assertion above
        if isinstance(summary_content, str):
            summary_text = summary_content.replace("Previous conversation summary:", "").strip()
            self.assertEqual(summary_text, "This is the summary text.")


class TestConfiguration(unittest.TestCase):
    """Test configuration loading and defaults"""

    def test_default_values_exist(self):
        """Test that default configuration values are set"""
        self.assertIsInstance(VLLM_BASE_URL, str)
        self.assertIsInstance(VLLM_API_KEY, str)
        self.assertIsInstance(VLLM_MODEL, str)
        self.assertIsInstance(SESSION_FILE, Path)

    def test_token_limits_are_positive(self):
        """Test that token limits are positive integers"""
        self.assertGreater(MAX_INPUT_TOKENS, 0)
        self.assertGreater(MAX_OUTPUT_TOKENS, 0)
        self.assertGreater(KEEP_RECENT_MESSAGES, 0)

    def test_keep_recent_messages_reasonable(self):
        """Test that KEEP_RECENT_MESSAGES is a reasonable value"""
        self.assertGreater(KEEP_RECENT_MESSAGES, 2)
        self.assertLess(KEEP_RECENT_MESSAGES, 20)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple functions"""

    def setUp(self):
        """Set up temporary session file"""
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")  # noqa: SIM115
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary file"""
        if self.temp_path.exists():
            self.temp_path.unlink()

    def test_save_and_load_roundtrip(self):
        """Test that save and load work together correctly"""
        original_messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        # Save
        save_result = save_session(original_messages, self.temp_path)
        self.assertTrue(save_result)

        # Load
        loaded_messages = load_session(self.temp_path)
        self.assertIsNotNone(loaded_messages)

        # Verify
        if loaded_messages is not None:
            self.assertEqual(len(loaded_messages), len(original_messages))
            for i, msg in enumerate(loaded_messages):
                self.assertEqual(msg["role"], original_messages[i]["role"])
                self.assertEqual(msg.get("content"), original_messages[i].get("content"))

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
            self.assertEqual(original_count, loaded_count)


if __name__ == "__main__":
    unittest.main()
