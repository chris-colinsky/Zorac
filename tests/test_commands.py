"""
Tests for the command registry module.
"""

from zorac.commands import COMMANDS, get_help_text, get_system_prompt_commands


class TestCommandRegistry:
    """Tests for the COMMANDS registry structure."""

    def test_commands_is_list(self):
        """Verify COMMANDS is a list."""
        assert isinstance(COMMANDS, list)
        assert len(COMMANDS) > 0

    def test_all_commands_have_required_fields(self):
        """Verify all commands have required fields."""
        for cmd in COMMANDS:
            assert "command" in cmd, f"Command missing 'command' field: {cmd}"
            assert "description" in cmd, f"Command missing 'description' field: {cmd}"
            assert "detailed" in cmd, f"Command missing 'detailed' field: {cmd}"

    def test_all_command_fields_are_strings(self):
        """Verify all command fields are non-empty strings."""
        for cmd in COMMANDS:
            assert isinstance(cmd["command"], str)
            assert isinstance(cmd["description"], str)
            assert isinstance(cmd["detailed"], str)
            assert len(cmd["command"]) > 0
            assert len(cmd["description"]) > 0
            assert len(cmd["detailed"]) > 0

    def test_all_commands_start_with_slash(self):
        """Verify all commands start with '/'."""
        for cmd in COMMANDS:
            # Handle commands with "or" (e.g., "/quit or /exit")
            command_parts = cmd["command"].split(" or ")
            for part in command_parts:
                assert part.strip().startswith("/"), f"Command doesn't start with '/': {part}"

    def test_no_duplicate_commands(self):
        """Verify there are no duplicate command names."""
        command_names = [cmd["command"] for cmd in COMMANDS]
        assert len(command_names) == len(set(command_names)), "Duplicate commands found"

    def test_expected_commands_present(self):
        """Verify all expected commands are present."""
        expected_commands = [
            "/help",
            "/quit or /exit",
            "/clear",
            "/save",
            "/load",
            "/tokens",
            "/summarize",
            "/summary",
            "/reconnect",
            "/config",
        ]
        command_names = [cmd["command"] for cmd in COMMANDS]
        for expected in expected_commands:
            assert expected in command_names, f"Expected command not found: {expected}"


class TestGetHelpText:
    """Tests for get_help_text() function."""

    def test_returns_string(self):
        """Verify get_help_text returns a string."""
        result = get_help_text()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_header(self):
        """Verify help text contains header."""
        result = get_help_text()
        assert "Available Commands:" in result

    def test_contains_all_commands(self):
        """Verify help text contains all commands."""
        result = get_help_text()
        for cmd in COMMANDS:
            # Check that the command appears in the output
            assert cmd["command"] in result, f"Command not found in help text: {cmd['command']}"

    def test_contains_all_descriptions(self):
        """Verify help text contains all descriptions."""
        result = get_help_text()
        for cmd in COMMANDS:
            assert cmd["description"] in result, (
                f"Description not found in help text: {cmd['description']}"
            )

    def test_uses_rich_formatting(self):
        """Verify help text uses Rich color codes."""
        result = get_help_text()
        assert "[cyan]" in result, "Help text should use cyan color for commands"
        assert "[bold]" in result, "Help text should use bold for header"

    def test_includes_config_subcommands(self):
        """Verify help text includes /config subcommands."""
        result = get_help_text()
        assert "/config list" in result
        assert "/config set" in result
        assert "/config get" in result


class TestGetSystemPromptCommands:
    """Tests for get_system_prompt_commands() function."""

    def test_returns_string(self):
        """Verify get_system_prompt_commands returns a string."""
        result = get_system_prompt_commands()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_zorac_context(self):
        """Verify prompt contains Zorac context."""
        result = get_system_prompt_commands()
        assert "Zorac" in result
        assert "terminal-based chat client" in result

    def test_contains_all_commands(self):
        """Verify system prompt contains all commands."""
        result = get_system_prompt_commands()
        for cmd in COMMANDS:
            assert cmd["command"] in result, f"Command not found in system prompt: {cmd['command']}"

    def test_contains_detailed_descriptions(self):
        """Verify system prompt contains detailed descriptions."""
        result = get_system_prompt_commands()
        for cmd in COMMANDS:
            assert cmd["detailed"] in result, (
                f"Detailed description not found in system prompt: {cmd['detailed']}"
            )

    def test_contains_guidance_for_llm(self):
        """Verify system prompt contains guidance for LLM."""
        result = get_system_prompt_commands()
        assert "When users ask about functionality" in result
        assert "help them understand these commands" in result

    def test_no_rich_formatting_in_system_prompt(self):
        """Verify system prompt doesn't contain Rich formatting codes."""
        result = get_system_prompt_commands()
        assert "[cyan]" not in result, "System prompt should not contain Rich color codes"
        assert "[bold]" not in result, "System prompt should not contain Rich formatting codes"

    def test_system_prompt_is_reasonably_sized(self):
        """Verify system prompt isn't excessively long."""
        result = get_system_prompt_commands()
        # Should be informative but not wasteful of tokens
        # The actual size is ~2200 characters which is reasonable
        # This translates to roughly 400-450 tokens which is acceptable
        assert len(result) < 2600, "System prompt may be too long and waste tokens"
