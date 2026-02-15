"""
Tests for the command registry module (zorac/commands.py).

This test file validates the command registry's data integrity and the output
of its two formatting functions. It's structured around three concerns:

  1. **TestCommandRegistry**: Validates the COMMANDS data structure itself —
     ensuring every command has the required fields, correct types, and follows
     naming conventions. These are "schema tests" that catch mistakes when
     adding or modifying commands.

  2. **TestGetHelpText**: Validates the Rich-formatted output shown to users
     via the /help command. Ensures all commands are listed with descriptions
     and that proper Rich markup is applied for terminal styling.

  3. **TestGetSystemPromptCommands**: Validates the plain-text output sent to
     the LLM in the system prompt. Ensures it contains all command information
     without Rich markup (which would confuse the model) and stays within
     reasonable token bounds.

Why test data structure integrity?
  The COMMANDS list is a hand-maintained data structure. Without these tests,
  it's easy to introduce subtle bugs: misspelled keys, missing fields, duplicate
  triggers, or commands that don't start with "/". These tests catch those issues
  immediately in CI rather than discovering them at runtime.
"""

from zorac.commands import COMMANDS, get_help_text, get_system_prompt_commands


class TestCommandRegistry:
    """Tests for the COMMANDS registry data structure.

    These tests validate the shape and content of the COMMANDS list.
    Think of them as "contract tests" — they ensure the data structure
    conforms to the expected schema that other parts of the code depend on.
    """

    def test_commands_is_list(self):
        """Verify COMMANDS is a non-empty list.

        An empty command list would mean /help shows nothing and the system
        prompt has no command information for the LLM.
        """
        assert isinstance(COMMANDS, list)
        assert len(COMMANDS) > 0

    def test_all_commands_have_required_fields(self):
        """Verify every command entry has all three required fields.

        The CommandInfo TypedDict requires triggers, description, and detailed.
        TypedDict provides static type checking, but this runtime test catches
        issues in dynamic scenarios (e.g., building commands programmatically).
        """
        for cmd in COMMANDS:
            assert "triggers" in cmd, f"Command missing 'triggers' field: {cmd}"
            assert "description" in cmd, f"Command missing 'description' field: {cmd}"
            assert "detailed" in cmd, f"Command missing 'detailed' field: {cmd}"

    def test_all_command_fields_are_correct_types(self):
        """Verify all fields have the correct types and are non-empty.

        Catches issues like an empty triggers list (command can't be invoked)
        or empty description/detailed strings (help output would be useless).
        """
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
        """Verify all command triggers start with '/'.

        The command dispatcher in main.py checks for leading '/' to distinguish
        commands from chat messages. A trigger without '/' would never be matched.
        """
        for cmd in COMMANDS:
            for trigger in cmd["triggers"]:
                assert trigger.startswith("/"), f"Command doesn't start with '/': {trigger}"

    def test_no_duplicate_triggers(self):
        """Verify there are no duplicate command triggers.

        Duplicate triggers would cause ambiguity in the command dispatcher.
        The first match would always win, making the second unreachable.
        """
        all_triggers = []
        for cmd in COMMANDS:
            all_triggers.extend(cmd["triggers"])
        assert len(all_triggers) == len(set(all_triggers)), (
            f"Duplicate triggers found: {all_triggers}"
        )

    def test_expected_commands_present(self):
        """Verify all expected commands are registered.

        This is a completeness check — if someone adds a command handler in
        main.py but forgets to register it in COMMANDS, this test will catch it.
        """
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
    """Tests for get_help_text() — the /help command output.

    get_help_text() produces Rich-formatted text for terminal display.
    These tests verify the output contains all necessary information and
    uses proper Rich markup for styling.
    """

    def test_returns_string(self):
        """Verify get_help_text returns a non-empty string."""
        result = get_help_text()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_header(self):
        """Verify help text includes the "Available Commands:" header."""
        result = get_help_text()
        assert "Available Commands:" in result

    def test_contains_all_commands(self):
        """Verify help text lists at least one trigger from every command.

        Some commands have multiple triggers (e.g., /quit and /exit), so we
        check that at least one trigger from each command appears.
        """
        result = get_help_text()
        for cmd in COMMANDS:
            found = False
            for trigger in cmd["triggers"]:
                if trigger in result:
                    found = True
                    break
            assert found, f"No trigger for command found in help text: {cmd['triggers']}"

    def test_contains_all_descriptions(self):
        """Verify help text includes every command's description string."""
        result = get_help_text()
        for cmd in COMMANDS:
            assert cmd["description"] in result, (
                f"Description not found in help text: {cmd['description']}"
            )

    def test_uses_rich_formatting(self):
        """Verify help text uses Rich markup for terminal styling.

        The help text should use [cyan] for command names and [bold] for the
        header, providing visual distinction in the terminal.
        """
        result = get_help_text()
        assert "[cyan]" in result, "Help text should use cyan color for commands"
        assert "[bold]" in result, "Help text should use bold for header"

    def test_includes_config_subcommands(self):
        """Verify help text shows /config subcommands (list, set, get).

        The /config command has subcommands that are shown inline in the help
        text for discoverability, since /config alone just shows the list.
        """
        result = get_help_text()
        assert "/config list" in result
        assert "/config set" in result
        assert "/config get" in result


class TestGetSystemPromptCommands:
    """Tests for get_system_prompt_commands() — the LLM system prompt content.

    This function generates plain text for the LLM's system prompt, enabling
    it to understand and suggest commands. These tests verify content completeness
    and ensure no Rich markup leaks into the prompt (which would waste tokens
    and confuse the model).
    """

    def test_returns_string(self):
        """Verify get_system_prompt_commands returns a non-empty string."""
        result = get_system_prompt_commands()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_zorac_context(self):
        """Verify the prompt sets context about what Zorac is.

        The LLM needs to know it's operating within a terminal chat client
        so it can provide contextually appropriate suggestions.
        """
        result = get_system_prompt_commands()
        assert "Zorac" in result
        assert "terminal-based chat client" in result

    def test_contains_all_commands(self):
        """Verify the system prompt mentions all registered commands."""
        result = get_system_prompt_commands()
        for cmd in COMMANDS:
            found = False
            for trigger in cmd["triggers"]:
                if trigger in result:
                    found = True
                    break
            assert found, f"No trigger for command found in system prompt: {cmd['triggers']}"

    def test_contains_detailed_descriptions(self):
        """Verify the system prompt includes the detailed descriptions.

        The system prompt uses the 'detailed' field (not 'description') because
        the LLM benefits from thorough explanations to give accurate suggestions.
        """
        result = get_system_prompt_commands()
        for cmd in COMMANDS:
            assert cmd["detailed"] in result, (
                f"Detailed description not found in system prompt: {cmd['detailed']}"
            )

    def test_contains_guidance_for_llm(self):
        """Verify the prompt includes behavioral guidance for the LLM.

        The system prompt should instruct the LLM on how to use command
        information — naturally suggesting commands when relevant rather
        than listing them unprompted.
        """
        result = get_system_prompt_commands()
        assert "When users ask about functionality" in result
        assert "help them understand these commands" in result

    def test_no_rich_formatting_in_system_prompt(self):
        """Verify the system prompt contains NO Rich markup.

        Rich formatting tags like [cyan] and [bold] are meaningless to the LLM.
        They would waste prompt tokens and might confuse the model into
        including them in its responses.
        """
        result = get_system_prompt_commands()
        assert "[cyan]" not in result, "System prompt should not contain Rich color codes"
        assert "[bold]" not in result, "System prompt should not contain Rich formatting codes"

    def test_system_prompt_is_reasonably_sized(self):
        """Verify the system prompt doesn't consume excessive tokens.

        The system prompt competes with conversation history for the context
        window. A prompt over 3000 chars would be wasteful.
        """
        result = get_system_prompt_commands()
        assert len(result) < 3000, "System prompt may be too long and waste tokens"
