# Help Feature Requirements

**Version:** 1.0
**Date:** 2026-02-01
**Status:** Approved

## Overview

Add comprehensive help functionality to Zorac chat application with two complementary features:
1. A `/help` command that displays all available commands
2. LLM awareness of commands through system prompt enhancement

## User Stories

### US-1: Display Available Commands
**As a** Zorac user
**I want** to see a list of all available commands
**So that** I can discover and understand the available functionality

**Acceptance Criteria:**
- User can type `/help` to display all commands
- Output format matches existing `/config list` style (simple formatted text)
- Each command shows a one-line description
- Display uses consistent color scheme (cyan for commands, white for descriptions)

### US-2: LLM Command Awareness
**As a** Zorac user
**I want** to ask the LLM natural language questions about commands
**So that** I can learn how to use the application conversationally

**Acceptance Criteria:**
- LLM can answer questions like "how do I save my session?"
- LLM can suggest relevant commands based on user needs
- LLM can provide usage examples for commands
- Command information is always available in system prompt
- LLM responses are contextually aware and helpful

## Functional Requirements

### FR-1: Command Registry
Create a centralized command registry (`zorac/commands.py`) that defines all available commands.

**Data Structure:**
```python
COMMANDS = [
    {
        "command": str,      # e.g., "/help", "/quit or /exit"
        "description": str,  # Short one-line description for /help display
        "detailed": str      # Detailed explanation for system prompt
    },
    # ...
]
```

**Commands to Include:**
1. `/help` - Show all available commands
2. `/quit` or `/exit` - Save and exit the application
3. `/clear` - Reset conversation to initial system message
4. `/save` - Manually save session to disk
5. `/load` - Reload session from disk
6. `/tokens` - Display current token usage statistics
7. `/summarize` - Force conversation summarization
8. `/summary` - Display the current conversation summary
9. `/config` - Manage configuration settings
   - `/config list` - Show current configuration
   - `/config set <KEY> <VALUE>` - Set a configuration value
   - `/config get <KEY>` - Get a specific configuration value

### FR-2: /help Command Implementation
Implement `/help` command in `zorac/main.py` main loop.

**Display Format:**
```
Available Commands:
  /help              - Show all available commands
  /quit or /exit     - Save and exit the application
  /clear             - Reset conversation to initial system message
  /save              - Manually save session to disk
  /load              - Reload session from disk
  /tokens            - Display current token usage statistics
  /summarize         - Force conversation summarization
  /summary           - Display the current conversation summary
  /config            - Manage configuration settings
    /config list     - Show current configuration
    /config set      - Set a configuration value
    /config get      - Get a specific configuration value
```

**Visual Style:**
- Use `console.print()` with Rich formatting
- Match `/config list` style (simple formatted text, not panels)
- Left-align command names with consistent spacing
- Use `[cyan]` for command names
- Use normal color for descriptions
- Add bold header "Available Commands:"

### FR-3: System Prompt Enhancement
Modify system prompt in `zorac/main.py` to include command awareness.

**Implementation:**
- Import command registry from `zorac/commands.py`
- Generate command help text from registry `detailed` field
- Append to system message in format:

```
You are a helpful assistant.

The user is interacting with you through Zorac, a terminal-based chat client for local LLMs.

Available Commands:
The following commands are available to the user:

/help - <detailed description>
/quit or /exit - <detailed description>
...

When users ask about functionality, help them understand these commands naturally.
```

### FR-4: LLM Interaction Capabilities
The LLM should be able to:
1. Answer questions about specific commands
2. Suggest relevant commands when users describe needs
3. Provide usage examples with proper syntax
4. Explain command parameters and options
5. Handle variations like "how do I...", "can I...", "what's the command for..."

## Non-Functional Requirements

### NFR-1: Performance
- Command registry loading should add minimal startup time (<10ms)
- `/help` command should display instantly (<100ms)
- System prompt expansion should not significantly increase token usage

### NFR-2: Maintainability
- Single source of truth for all command definitions
- Adding new commands requires updating only `COMMANDS` registry
- Command descriptions should be easy to update
- Code should follow existing project patterns

### NFR-3: User Experience
- `/help` output should be visually consistent with existing commands
- LLM responses about commands should feel natural
- Help information should be accurate and up-to-date

## Technical Design

### Architecture Overview
```
zorac/
├── commands.py          [NEW] - Command registry
├── main.py             [MODIFIED] - Add /help handler, enhance system prompt
└── ...
```

### Implementation Plan

#### Phase 1: Command Registry (Priority: High)
1. Create `zorac/commands.py`
2. Define `COMMANDS` list with all current commands
3. Export public API: `COMMANDS`, helper functions if needed

#### Phase 2: /help Command (Priority: High)
1. Add `/help` handler in `main.py` interactive loop
2. Import `COMMANDS` from `zorac.commands`
3. Format and display command list using Rich console
4. Match existing `/config list` visual style

#### Phase 3: System Prompt Enhancement (Priority: High)
1. Create function to generate command help text from registry
2. Modify system message initialization in `main()`
3. Append command information to system prompt
4. Test that LLM receives and uses command information

#### Phase 4: Testing (Priority: High)
1. Add unit tests for command registry
2. Add integration test for `/help` display
3. Add test for system prompt generation
4. Add test for LLM command awareness (mock-based)
5. Ensure 80%+ code coverage for new code

#### Phase 5: Documentation (Priority: High)
1. Update `README.md` - Add `/help` to command list
2. Update `CLAUDE.md` - Document command registry architecture
3. Update `docs/USAGE.md` - Add `/help` command documentation
4. Add docstrings to new code

## Testing Strategy

### Unit Tests
**Test File:** `tests/test_commands.py`

1. `test_command_registry_structure()`
   - Verify all commands have required fields
   - Validate data types
   - Check for duplicate commands

2. `test_help_command_display()`
   - Mock console output
   - Verify formatting and content
   - Check color codes

3. `test_system_prompt_generation()`
   - Verify command info is included
   - Check formatting
   - Validate structure

### Integration Tests
**Test File:** `tests/test_integration.py`

1. `test_help_command_in_main_loop()`
   - Simulate user typing `/help`
   - Verify output matches expected format

2. `test_llm_receives_command_info()`
   - Mock OpenAI client
   - Verify system message contains command information
   - Check LLM receives proper context

### Manual Testing Checklist
- [ ] `/help` displays all commands correctly
- [ ] Output matches `/config list` visual style
- [ ] Ask LLM "how do I save my session?" - receives helpful answer
- [ ] Ask LLM "what commands are available?" - lists commands
- [ ] Ask LLM "show me how to use /config" - provides examples
- [ ] New session includes command info in system prompt
- [ ] All existing commands still work correctly

## Documentation Updates

### README.md
Add `/help` command to the "Interactive Commands" section:
```markdown
- `/help` - Show all available commands
```

### CLAUDE.md
Update "Interactive Commands" section with `/help` and document new architecture:
```markdown
### Command Registry (zorac/commands.py)
- Centralized registry of all interactive commands
- Single source of truth for command definitions
- Used by both `/help` display and system prompt generation
```

### docs/USAGE.md
Add `/help` command documentation:
```markdown
## Getting Help

### /help Command
Display a list of all available commands:
```
zorac> /help
```

This shows all interactive commands with descriptions.
```

### Code Documentation
Add comprehensive docstrings to:
- `zorac/commands.py` - Module, COMMANDS constant, any helper functions
- `/help` handler in `main.py` - Inline comments
- System prompt generation - Inline comments

## Success Metrics

1. **Functionality:** `/help` command displays all commands correctly
2. **LLM Awareness:** LLM can answer at least 5 different command-related questions accurately
3. **Code Quality:** All new code has 80%+ test coverage
4. **Documentation:** All docs updated and accurate
5. **User Experience:** Help information is clear and helpful

## Dependencies

- No new external dependencies required
- Uses existing Rich library for formatting
- Uses existing test framework (pytest)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| System prompt becomes too long | Token usage increase | Keep `detailed` descriptions concise; monitor token count |
| LLM gives incorrect command info | User confusion | Test thoroughly; keep descriptions accurate |
| Command registry gets out of sync | Stale help info | Add pre-commit hook or test to verify completeness |
| Performance impact from larger system prompt | Slower responses | Measure token overhead; optimize if needed |

## Future Enhancements (Out of Scope)

- `/help <command>` - Show detailed help for specific command
- Interactive help mode
- Command autocomplete
- Search functionality in help
- Categorized command groups
- Alias support in command registry

## Acceptance Criteria Summary

The feature is complete when:
- [x] Requirements document written and reviewed
- [ ] `zorac/commands.py` created with complete command registry
- [ ] `/help` command implemented and working
- [ ] System prompt enhanced with command information
- [ ] LLM can answer natural language questions about commands
- [ ] All tests passing with 80%+ coverage
- [ ] All documentation updated (README, CLAUDE, USAGE)
- [ ] Manual testing completed
- [ ] Code review completed
- [ ] Feature merged to main branch

## Approval

This requirements document has been reviewed and approved for implementation.
