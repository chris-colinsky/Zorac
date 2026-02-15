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
            assert "triggers" in cmd, f"Command missing 'triggers' field: {cmd}"
            assert "description" in cmd, f"Command missing 'description' field: {cmd}"
            assert "detailed" in cmd, f"Command missing 'detailed' field: {cmd}"

    def test_all_command_fields_are_correct_types(self):
        """Verify all command fields are non-empty strings or lists."""
        for cmd in COMMANDS:
            assert isinstance(cmd["triggers"], list)
            assert isinstance(cmd["description"], str)
            assert isinstance(cmd["detailed"], str)
            assert len(cmd["triggers"]) > 0
            assert len(cmd["description"]) > 0
            assert len(cmd["detailed"]) > 0
            for trigger in cmd["triggers"]:
                assert isinstance(trigger, str)
                assert len(trigger) > 0

    def test_all_commands_start_with_slash(self):
        """Verify all commands start with '/'."""
        for cmd in COMMANDS:
            for trigger in cmd["triggers"]:
                assert trigger.startswith("/"), f"Command doesn't start with '/': {trigger}"

    def test_no_duplicate_triggers(self):
        """Verify there are no duplicate command triggers."""
        all_triggers = []
        for cmd in COMMANDS:
            all_triggers.extend(cmd["triggers"])
        assert len(all_triggers) == len(set(all_triggers)), (
            f"Duplicate triggers found: {all_triggers}"
        )

    def test_expected_commands_present(self):
        """Verify all expected commands are present."""
        expected_triggers = [
            "/help",
            "/quit",
            "/exit",
            "/clear",
            "/save",
            "/load",
            "/tokens",
            "/summarize",
            "/summary",
            "/reconnect",
            "/config",
        ]
        all_triggers = []
        for cmd in COMMANDS:
            all_triggers.extend(cmd["triggers"])

        for expected in expected_triggers:
            assert expected in all_triggers, f"Expected trigger not found: {expected}"


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
            # Check that at least one trigger from each command appears
            found = False
            for trigger in cmd["triggers"]:
                if trigger in result:
                    found = True
                    break
            assert found, f"No trigger for command found in help text: {cmd['triggers']}"

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
            found = False
            for trigger in cmd["triggers"]:
                if trigger in result:
                    found = True
                    break
            assert found, f"No trigger for command found in system prompt: {cmd['triggers']}"

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
        assert len(result) < 3000, "System prompt may be too long and waste tokens"
