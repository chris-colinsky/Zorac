# Development Guide

This guide covers setting up Zorac for development, testing, and contributing.

## Development Setup

### Prerequisites

- Python 3.13+
- `uv` package manager
- Git

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/chris-colinsky/zorac.git
cd zorac

# Complete development setup (installs deps + pre-commit hooks)
make dev-setup

# Or manually:
uv sync --extra dev
cp .env.example .env
# Edit .env to set your vLLM server URL
```

## Project Structure

```
zorac/
├── zorac/                  # Main package
│   ├── __init__.py        # Package exports
│   ├── __main__.py        # Module entry point
│   ├── commands.py        # Command registry
│   ├── config.py          # Configuration management
│   ├── console.py         # Rich console singleton
│   ├── llm.py             # LLM operations
│   ├── main.py            # Main event loop
│   ├── markdown_custom.py # Custom markdown renderer
│   ├── session.py         # Session persistence
│   └── utils.py           # Utility functions
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_commands.py   # Command registry tests (NEW)
│   └── test_zorac.py      # Comprehensive tests
├── requirements/           # Requirements documents (NEW)
│   └── help_feature.md    # Help feature requirements
├── docs/                   # Documentation
│   ├── INSTALLATION.md
│   ├── CONFIGURATION.md
│   ├── USAGE.md
│   ├── DEVELOPMENT.md     # This file
│   ├── SERVER_SETUP.md    # vLLM server guide
│   └── TEST_TRAINING.md   # Multi-GPU training guide (NEW)
├── .github/workflows/      # CI/CD
│   └── release.yml        # Release automation
├── pyproject.toml          # Dependencies and metadata
├── uv.lock                 # Dependency lock file
├── .env.example            # Configuration template
├── README.md               # Project overview
├── CLAUDE.md               # AI assistant guide
├── CHANGELOG.md            # Version history
├── CONTRIBUTING.md         # Contribution guidelines
├── Makefile                # Development commands
├── run_zorac.py            # Binary entry point
├── zorac.spec              # PyInstaller spec
└── LICENSE                 # MIT License
```

## Running Tests

### Basic Testing

```bash
# Run all tests
make test

# Run with coverage report (generates HTML report)
make test-coverage

# View coverage report
open htmlcov/index.html
```

### Test Coverage

Current coverage: **42%** with **34 passing tests**

The test suite covers:
- **Token counting** with various message types
- **Session management** (save/load, error handling)
- **UI header rendering**
- **Message summarization logic**
- **Configuration validation**
- **Command registry** structure and validation
- **Help system** (display and LLM awareness)
- **Integration workflows** (save/load roundtrips, help feature)

**Module coverage:**
- `zorac/commands.py`: 100% coverage
- `zorac/session.py`: 100% coverage
- `zorac/console.py`: 100% coverage
- `zorac/llm.py`: 96% coverage
- `zorac/markdown_custom.py`: 45% coverage (UI rendering, mostly tested via integration)

**Coverage targets:**
- Minimum 80% code coverage
- All core functions tested
- Edge cases and error paths covered

## Code Quality

### Linting

```bash
# Run linter
make lint

# Auto-fix linting issues
make lint-fix
```

### Formatting

```bash
# Format code
make format
```

### Type Checking

```bash
# Run mypy type checking
make type-check
```

### All Checks

```bash
# Run all checks (lint + type + test)
make all-checks
```

## Pre-commit Hooks

Zorac uses pre-commit hooks for automated quality checks:

```bash
# Install pre-commit hooks
make pre-commit-install

# Run pre-commit on all files
make pre-commit

# Pre-commit will automatically run on git commit
```

## Dependencies

### Production Dependencies

- **openai** (>=2.8.1): Client library for vLLM's OpenAI-compatible API
- **tiktoken** (>=0.8.0): Token counting for accurate usage tracking
- **python-dotenv** (>=1.0.0): Environment variable management from .env files
- **rich** (>=14.2.0): Rich terminal formatting, markdown rendering, and live updates

### Development Dependencies

- **pytest** (>=8.0.0): Testing framework
- **pytest-cov** (>=4.1.0): Coverage reporting for pytest
- **pytest-mock** (>=3.12.0): Mock fixtures for pytest
- **ruff** (>=0.8.0): Fast Python linter and formatter
- **mypy** (>=1.8.0): Static type checker
- **pre-commit** (>=3.6.0): Git hook management

## Architecture Overview

### Module Breakdown

**zorac/commands.py** - Command Registry (NEW)
- Centralized registry of all interactive commands
- `COMMANDS`: List of command definitions with descriptions
- `get_help_text()`: Generates formatted help for `/help` display
- `get_system_prompt_commands()`: Provides command info to LLM via system prompt
- Single source of truth for command documentation
- Enables LLM awareness of Zorac's capabilities

**zorac/config.py** - Configuration Management
- Three-tier priority: Environment Variables > Config File > Defaults
- Type-safe getters: `get_int_setting()`, `get_float_setting()`, `get_bool_setting()`
- Runtime configuration via `/config` command
- Persistent storage in `~/.zorac/config.json`

**zorac/console.py** - Rich Console Singleton
- Single Rich console instance for consistent formatting
- Used across all modules for output

**zorac/llm.py** - LLM Operations
- `summarize_old_messages()`: Auto-summarization with status spinner
- Handles API calls to vLLM server
- Token limit management

**zorac/markdown_custom.py** - Custom Markdown Renderer
- `LeftAlignedMarkdown`: Custom markdown renderer with left-aligned headings
- Monkey-patches Rich's `Heading.__rich_console__` to remove centered heading panels
- Provides cleaner, more readable terminal output
- Maintains Rich styling (bold, colored) while forcing left justification

**zorac/main.py** - Main Event Loop
- Interactive REPL with command handling
- `get_initial_system_message()`: Generates enhanced system prompt with command info
- `ConstrainedWidth`: Custom Rich renderable that constrains content to 60% of console width
  - Uses `ConsoleOptions.update(max_width=...)` for stable text wrapping
  - Provides optimal readability on all terminal sizes
- Streaming and non-streaming response modes
- Session auto-save after each response
- Performance metrics tracking

**zorac/session.py** - Session Persistence
- `save_session()`: Saves conversation to JSON
- `load_session()`: Restores conversation from disk
- Error handling for invalid sessions

**zorac/utils.py** - Utility Functions
- `count_tokens()`: Uses tiktoken to count tokens in conversation
- `print_header()`: Displays formatted welcome header
- `check_connection()`: Verifies vLLM server connectivity

## Adding Features

### New Commands

When adding a new command, follow these steps to ensure consistency:

1. **Add to Command Registry** (`zorac/commands.py`):

```python
{
    "command": "/mycommand",
    "description": "Short one-line description for /help",
    "detailed": "Detailed explanation for the LLM system prompt. Describe what the command does and when to use it."
}
```

2. **Implement Command Handler** in `zorac/main.py`:

```python
if user_input.lower() == "/mycommand":
    # Your command logic here
    console.print("[green]Command executed![/green]")
    continue
```

3. **Add Tests** in `tests/test_commands.py` or `tests/test_zorac.py`

4. **Update Documentation**:
   - Add to command list in `docs/USAGE.md`
   - Update `CLAUDE.md` if architecture changes
   - Update `README.md` if user-facing

**Why use the command registry?**
- Single source of truth for all commands
- `/help` command automatically displays your new command
- LLM automatically knows about your command and can suggest it to users
- Ensures consistency across documentation

### New Configuration Settings

1. Add default to `DEFAULT_CONFIG` in `zorac/config.py`
2. Load it in `main()` using appropriate getter
3. Use it in your code
4. Update documentation

### Custom Token Limits

Modify via `.env` file or `/config` command:

```python
MAX_INPUT_TOKENS = get_int_setting("MAX_INPUT_TOKENS", 12000)
```

### UI Customization

**Panel Formatting:**

Modify panel styling in `zorac/main.py`:

```python
console.print(Panel(
    content,
    box=box.ROUNDED,
    style="cyan",
    expand=True  # Full width for left-aligned content
))
```

**Content Width Constraint:**

Adjust the content width percentage in `zorac/main.py`:

```python
CONTENT_WIDTH_PCT = 0.6  # 60% width, change to desired percentage

# Apply to content
constrained_content = ConstrainedWidth(markdown_content, CONTENT_WIDTH_PCT)
```

**Custom Markdown Rendering:**

Modify heading styles in `zorac/markdown_custom.py`:

```python
def _left_aligned_heading_rich_console(self, console, options):
    if self.tag == "h1":
        yield Text(text.plain, style="bold #ffffff")  # Customize H1 style
    elif self.tag == "h2":
        yield Text(text.plain, style="bold #cccccc")  # Customize H2 style
```

**Design Principles:**
- Use width constraint (not padding) for stable layout on window resize
- Left-align all content to match "Assistant:" label
- Remove `vertical_overflow` from Live contexts to prevent scroll duplication
- Constrain width using `ConsoleOptions.update(max_width=...)` for proper text wrapping

## Feature Requirements

The `requirements/` directory contains detailed specifications for features. When adding significant new functionality:

1. **Create a requirements document**: `requirements/feature_name.md`
2. **Include these sections**:
   - Overview and user stories
   - Functional requirements
   - Technical design
   - Implementation plan
   - Testing strategy
   - Documentation updates
   - Acceptance criteria

**Example:** See `requirements/help_feature.md` for a comprehensive example of a feature specification.

**Benefits:**
- Clear specification before implementation
- Easier code review
- Better documentation
- Prevents scope creep

## Building Binaries

### Local Build

```bash
# Install PyInstaller
uv pip install pyinstaller

# Build binary
uv run pyinstaller zorac.spec

# Test the binary
./dist/zorac --help
```

### Release Build

Binaries are automatically built by GitHub Actions when you push a tag:

```bash
git tag v1.0.0
git push origin main --tags
```

This triggers the workflow in `.github/workflows/release.yml` which:
1. Builds binaries for Linux and macOS
2. Creates a GitHub release
3. Attaches binaries to the release

**Note:** Windows is not officially supported. Windows users should use [WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/install).

## Making a Release

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Run all checks**: `make all-checks`
4. **Commit changes**: `git commit -m "Release vX.Y.Z"`
5. **Create tag**: `git tag vX.Y.Z`
6. **Push**: `git push origin main --tags`

GitHub Actions will automatically build and publish the release.

## LLM Command Awareness

Zorac includes an innovative feature where the LLM assistant running inside the chat client is aware of Zorac's own commands and capabilities.

### How It Works

1. **Command Registry** (`zorac/commands.py`):
   - Centralized list of all commands with detailed descriptions
   - Single source of truth for command information

2. **Enhanced System Prompt** (`zorac/main.py`):
   - `get_initial_system_message()` injects command information into the system prompt
   - LLM receives command details at session start

3. **Natural Language Queries**:
   - Users can ask: "How do I save my session?"
   - LLM responds with relevant command information
   - LLM suggests appropriate commands based on user needs

### Token Overhead

The enhanced system prompt adds ~400-450 tokens to each session. This is acceptable because:
- Default context limit is 12,000 tokens
- Overhead is <4% of available context
- Users benefit from having an assistant that knows how to use the tool

### Testing Command Awareness

See `tests/test_zorac.py::TestHelpFeatureIntegration` for tests that verify:
- System message includes all commands
- Token overhead is reasonable
- Help text is properly formatted

## AI-Friendly Documentation

This project includes `CLAUDE.md`, a comprehensive development guide designed to help AI coding assistants understand the project.

**What it contains:**
- Detailed project structure and file purposes
- Architecture overview and design decisions
- Configuration documentation
- Development workflows and commands
- Dependencies and their usage

**Why it's useful:**
AI-powered development tools can use this file to provide better suggestions and write code that matches project conventions.

## Contribution Guidelines

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

### Quick Checklist

Before submitting a PR:

- [ ] All tests pass: `make test`
- [ ] Code is linted: `make lint`
- [ ] Code is formatted: `make format`
- [ ] Type checking passes: `make type-check`
- [ ] Documentation is updated (README, USAGE, CLAUDE.md as needed)
- [ ] New commands added to command registry (`zorac/commands.py`)
- [ ] Requirements document created for significant features (`requirements/`)
- [ ] CHANGELOG.md is updated
- [ ] Commit messages are clear and descriptive

## Troubleshooting Development

### Virtual Environment Issues

```bash
# Remove and recreate virtual environment
rm -rf .venv
uv sync
```

### Test Failures

```bash
# Run tests with verbose output
uv run pytest -vv

# Run specific test
uv run pytest tests/test_zorac.py::TestCountTokens -vv
```

### Import Errors

```bash
# Ensure package is installed in development mode
uv sync --extra dev
```

### Command Registry Issues

If commands aren't showing up in `/help` or the LLM doesn't know about them:

```bash
# Verify command is in the registry
python -c "from zorac.commands import COMMANDS; print([c['command'] for c in COMMANDS])"

# Test help text generation
python -c "from zorac.commands import get_help_text; print(get_help_text())"

# Test system prompt generation
python -c "from zorac.commands import get_system_prompt_commands; print(get_system_prompt_commands())"
```

## Resources

- [vLLM Documentation](https://docs.vllm.ai/en/stable/) - vLLM inference server
- [Rich Documentation](https://rich.readthedocs.io/) - Terminal UI library
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) - API compatibility
- [tiktoken](https://github.com/openai/tiktoken) - Token counting library
- [prompt-toolkit](https://python-prompt-toolkit.readthedocs.io/) - Advanced input handling

## Support

For questions or issues:
1. Check existing [GitHub Issues](https://github.com/chris-colinsky/Zorac/issues)
2. Review documentation in `docs/` directory
3. Open a new issue with details

## License

MIT License - see [LICENSE](../LICENSE) for details.
