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

```text
zorac/
├── zorac/                  # Main package
│   ├── __init__.py        # Package exports
│   ├── __main__.py        # Module entry point
│   ├── commands.py        # Command registry
│   ├── config.py          # Configuration management
│   ├── console.py         # Rich console singleton
│   ├── handlers.py        # CommandHandlersMixin (all cmd_* methods)
│   ├── history.py         # HistoryMixin (command history + Up/Down nav)
│   ├── llm.py             # LLM operations
│   ├── main.py            # Textual TUI application orchestrator
│   ├── markdown_custom.py # Custom markdown renderer
│   ├── session.py         # Session persistence
│   ├── streaming.py       # StreamingMixin (LLM response streaming)
│   ├── utils.py           # Utility functions
│   └── widgets.py         # ChatInput widget (multiline + suggestions)
├── tests/                  # Test suite
│   ├── test_commands.py   # Command registry tests
│   └── test_zorac.py      # Comprehensive tests
├── quant-lab/              # Quantization tooling
│   ├── quantize_mistral.py # AWQ quantization script
│   └── README.md          # Quantization guide
├── docs/                   # Documentation
│   ├── index.md           # MkDocs site home page
│   ├── img/               # Site images (logo, etc.)
│   ├── stylesheets/       # Custom CSS for MkDocs
│   ├── site/              # Educational content (MkDocs)
│   │   ├── concepts/     # How LLMs Work, Quantization, etc.
│   │   ├── guides/       # Server Setup, TUI, Multi-GPU
│   │   ├── walkthroughs/ # Streaming, Quantizing, etc.
│   │   └── decisions/    # Why AWQ, Why Textual
│   ├── INSTALLATION.md
│   ├── CONFIGURATION.md
│   ├── USAGE.md
│   ├── DEVELOPMENT.md     # This file
│   ├── SERVER_SETUP.md    # vLLM server guide
│   └── TEST_TRAINING.md   # Multi-GPU training guide
├── .github/workflows/      # CI/CD
│   ├── ci.yml             # Continuous integration
│   ├── docs.yml           # MkDocs deployment to GitHub Pages
│   └── release.yml        # Release automation
├── mkdocs.yml              # MkDocs site configuration
├── pyproject.toml          # Dependencies and metadata
├── uv.lock                 # Dependency lock file
├── .env.example            # Configuration template
├── README.md               # Project overview
├── CLAUDE.md               # AI assistant guide
├── CHANGELOG.md            # Version history
├── CONTRIBUTING.md         # Contribution guidelines
├── Makefile                # Development commands
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

Current coverage: **51%** with **57 passing tests**

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
- `zorac/utils.py`: 92% coverage
- `zorac/config.py`: 59% coverage
- `zorac/main.py`: 58% coverage
- `zorac/widgets.py`: 47% coverage
- `zorac/history.py`: 29% coverage
- `zorac/streaming.py`: 27% coverage
- `zorac/handlers.py`: 24% coverage

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
- **textual** (>=1.0.0): Modern TUI framework for the full terminal user interface

### Development Dependencies

- **pytest** (>=8.0.0): Testing framework
- **pytest-asyncio** (>=0.23.0): Async test support (configured with `asyncio_mode = "auto"`)
- **pytest-cov** (>=4.1.0): Coverage reporting for pytest
- **pytest-mock** (>=3.12.0): Mock fixtures for pytest
- **ruff** (>=0.8.0): Fast Python linter and formatter
- **mypy** (>=1.8.0): Static type checker
- **pre-commit** (>=3.6.0): Git hook management

### Documentation Dependencies

Install with `uv sync --extra docs`:

- **mkdocs** (>=1.6.0): Static site generator for documentation
- **mkdocs-material** (>=9.5.0): Material theme for MkDocs

## Architecture Overview

### Module Breakdown

**zorac/commands.py** - Command Registry & System Message

- Centralized registry of all interactive commands
- `COMMANDS`: List of command definitions with descriptions
- `get_help_text()`: Generates formatted help for `/help` display
- `get_system_prompt_commands()`: Provides command info to LLM via system prompt
- `get_initial_system_message()`: Builds the system prompt with identity, date, and command awareness
- Single source of truth for command documentation

**zorac/config.py** - Configuration Management

- Three-tier priority: Environment Variables > Config File > Defaults
- Type-safe getters: `get_int_setting()`, `get_float_setting()`, `get_bool_setting()`
- Runtime configuration via `/config` command
- Persistent storage in `~/.zorac/config.json`

**zorac/console.py** - Rich Console Singleton

- Single Rich console instance for consistent formatting
- Used across all modules for output

**zorac/handlers.py** - Command Handlers (Mixin)

- `CommandHandlersMixin`: All `cmd_*()` methods for `/help`, `/quit`, `/config`, etc.
- Mixed into `ZoracApp` via Python MRO
- Outputs via `_log_system()` / `_log_user()` methods from main app

**zorac/history.py** - Command History (Mixin)

- `HistoryMixin`: History load/save and Up/Down arrow navigation
- `_load_history()`: Load from `~/.zorac/history`
- `_save_history()`: Persist last 500 entries with multiline escaping
- `on_key()`: Handles Up/Down arrow keys for readline-like history navigation

**zorac/llm.py** - LLM Operations

- `summarize_old_messages()`: Auto-summarization with status spinner
- Handles API calls to vLLM server
- Token limit management

**zorac/main.py** - Textual TUI Application (Orchestrator)

- `ZoracApp(CommandHandlersMixin, StreamingMixin, HistoryMixin, App)`: Main Textual application class
- `compose()`: Yields `VerticalScroll#chat-log`, `ChatInput#user-input`, `Static#stats-bar`
- `on_mount()` -> `_setup()`: Initialize `AsyncOpenAI` client, verify connection, load session
- `handle_chat()`: Adds user message to chat log, launches `_stream_response()` worker
- Session auto-save after each response
- UI helpers: `_log_system()`, `_log_user()`, `_update_stats_bar()`, `_write_header()`

**zorac/markdown_custom.py** - Custom Markdown Renderer

- `LeftAlignedMarkdown`: Custom markdown renderer with left-aligned headings
- Monkey-patches Rich's `Heading.__rich_console__` to remove centered heading panels
- Provides cleaner, more readable terminal output
- Maintains Rich styling (bold, colored) while forcing left justification

**zorac/streaming.py** - LLM Streaming (Mixin)

- `StreamingMixin`: `_stream_response()` worker method
- Streams LLM response using `Markdown.get_stream()`, updates `Static#stats-bar` in real-time
- Runs as `@work(exclusive=True)` background worker

**zorac/session.py** - Session Persistence

- `save_session()`: Saves conversation to JSON
- `load_session()`: Restores conversation from disk
- Error handling for invalid sessions

**zorac/utils.py** - Utility Functions

- `count_tokens()`: Uses tiktoken to count tokens in conversation
- `print_header()`: Displays formatted welcome header
- `check_connection()`: Verifies vLLM server connectivity

**zorac/widgets.py** - Chat Input Widget

- `ChatInput(TextArea)`: Multiline input widget with Enter to submit, Shift+Enter for newlines
- Auto-resizes from 1 to 5 lines based on content
- Inline command suggestions for `/commands`, accepted with Tab

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

- Full-width output for assistant responses
- Left-align all content to match "Assistant:" label
- Use `Markdown.get_stream()` during streaming for real-time content updates
- `Static#stats-bar` displays real-time metrics during streaming
- Stats bar updates in real-time during streaming, persists after response
- Stats persist in bottom toolbar between interactions

## Publishing Releases

Zorac is distributed via PyPI and Homebrew. Releases are automated via GitHub Actions.

### Distribution Channels

- **PyPI**: `pip install zorac` / `pipx install zorac`
- **Homebrew**: `brew tap chris-colinsky/zorac && brew install zorac`

### Semantic Versioning

- **MAJOR** (X.0.0): Breaking changes, incompatible API changes
- **MINOR** (x.Y.0): New features, backward compatible
- **PATCH** (x.y.Z): Bug fixes, documentation updates only

### Release Process

#### 1. Prepare the Release

```bash
# Create a release branch from main
git checkout main
git pull origin main
git checkout -b release/vX.Y.Z

# Update version in pyproject.toml
# Update CHANGELOG.md with release notes

# Run all quality checks
make all-checks

# Commit changes
git add .
git commit -m "Prepare release vX.Y.Z"
git push origin release/vX.Y.Z
```

#### 2. Create PR and Merge

```bash
# Create a pull request to main
gh pr create --base main --head release/vX.Y.Z --title "Release vX.Y.Z"

# After PR is approved and CI passes, merge via GitHub UI
```

#### 3. Tag and Release

```bash
# Switch to main and pull the merged changes
git switch main
git pull

# Create and push an annotated tag
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z

# Clean up release branch
git branch -d release/vX.Y.Z
```

#### 4. Automated Pipeline

When you push the tag, GitHub Actions automatically:

1. **Runs tests** - Ensures all tests pass
2. **Builds package** - Creates wheel and sdist
3. **Publishes to PyPI** - Via Trusted Publisher (OIDC)
4. **Creates GitHub Release** - With package artifacts
5. **Updates Homebrew formula** - Automatically updates `chris-colinsky/homebrew-zorac` with the new version, URL, and SHA256

**Monitor the release:**

- Visit: https://github.com/chris-colinsky/zorac/actions
- Look for "Release" workflow
- Verify PyPI publish and Homebrew update succeed

**Note:** The Homebrew update requires a `HOMEBREW_TAP_TOKEN` secret (a GitHub PAT with write access to `chris-colinsky/homebrew-zorac`).

#### 5. Verify Release

- Check PyPI: https://pypi.org/project/zorac/
- Check GitHub Release: https://github.com/chris-colinsky/zorac/releases/tag/vX.Y.Z
- Test installation:

  ```bash
  pip install zorac==X.Y.Z
  zorac --help
  ```

- Test Homebrew (after formula update):

  ```bash
  brew upgrade zorac
  zorac --help
  ```

### Release Checklist

Before creating a release:

- [ ] All tests pass: `make test`
- [ ] Code is linted: `make lint`
- [ ] Type checking passes: `make type-check`
- [ ] Version updated in `pyproject.toml`
- [ ] CHANGELOG.md updated with release notes
- [ ] All PRs for this release are merged
- [ ] Documentation is up to date

After release:

- [ ] PyPI page shows correct version
- [ ] `pip install zorac` works
- [ ] GitHub Release created with artifacts
- [ ] Homebrew formula updated (automatic)

### Quick Reference

```bash
# Complete release workflow
git checkout main && git pull
git checkout -b release/vX.Y.Z
# Update pyproject.toml version and CHANGELOG.md
make all-checks
git add . && git commit -m "Prepare release vX.Y.Z"
git push origin release/vX.Y.Z
gh pr create --base main --head release/vX.Y.Z --title "Release vX.Y.Z"
# Merge PR via GitHub UI
git switch main && git pull
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z
git branch -d release/vX.Y.Z
# Homebrew formula updates automatically after PyPI publish
```

### Hotfix Releases

For urgent bug fixes:

```bash
git checkout main && git pull
git checkout -b hotfix/vX.Y.Z
# Fix the bug, then follow the same release process
# Use PATCH version bump (e.g., 1.1.0 -> 1.1.1)
```

### Manual Publishing (Emergency)

If automated publishing fails:

```bash
# Build
python -m build

# Upload (requires PyPI token)
pip install twine
twine upload dist/*
```

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

See [CONTRIBUTING.md](https://github.com/chris-colinsky/Zorac/blob/main/CONTRIBUTING.md) for detailed contribution guidelines.

### Quick Checklist

Before submitting a PR:

- [ ] All tests pass: `make test`
- [ ] Code is linted: `make lint`
- [ ] Code is formatted: `make format`
- [ ] Type checking passes: `make type-check`
- [ ] Documentation is updated (README, USAGE, CLAUDE.md as needed)
- [ ] New commands added to command registry (`zorac/commands.py`)
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
- [Textual](https://textual.textualize.io/) - Modern TUI framework

## Support

For questions or issues:

1. Check existing [GitHub Issues](https://github.com/chris-colinsky/Zorac/issues)
2. Review documentation in `docs/` directory
3. Open a new issue with details

## License

MIT License - see [LICENSE](https://github.com/chris-colinsky/Zorac/blob/main/LICENSE) for details.
