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
│   ├── config.py          # Configuration management
│   ├── console.py         # Rich console singleton
│   ├── llm.py             # LLM operations
│   ├── main.py            # Main event loop
│   ├── session.py         # Session persistence
│   └── utils.py           # Utility functions
├── tests/                  # Test suite
│   ├── __init__.py
│   └── test_zorac.py      # Comprehensive tests
├── docs/                   # Documentation
│   ├── INSTALLATION.md
│   ├── CONFIGURATION.md
│   ├── USAGE.md
│   ├── DEVELOPMENT.md     # This file
│   └── SERVER_SETUP.md    # vLLM server guide
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

Current coverage: **37%** with **28 passing tests**

The test suite covers:
- **Token counting** with various message types
- **Session management** (save/load, error handling)
- **UI header rendering**
- **Message summarization logic**
- **Configuration validation**
- **Integration workflows** (save/load roundtrips)

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

**zorac/main.py** - Main Event Loop
- Interactive REPL with command handling
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

Add command handling in `zorac/main.py`:

```python
if user_input.lower() == "/mycommand":
    # Your command logic here
    console.print("[green]Command executed![/green]")
    continue
```

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

Modify Rich console styling in `zorac/console.py` or panel formatting in `zorac/main.py`:

```python
console.print(Panel(
    content,
    box=box.ROUNDED,
    style="cyan",
    expand=False
))
```

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
1. Builds binaries for Linux, macOS, and Windows
2. Creates a GitHub release
3. Attaches binaries to the release

## Making a Release

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Run all checks**: `make all-checks`
4. **Commit changes**: `git commit -m "Release vX.Y.Z"`
5. **Create tag**: `git tag vX.Y.Z`
6. **Push**: `git push origin main --tags`

GitHub Actions will automatically build and publish the release.

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
- [ ] Documentation is updated
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

## Resources

- [vLLM Documentation](https://docs.vllm.ai/) - vLLM inference server
- [Rich Documentation](https://rich.readthedocs.io/) - Terminal UI library
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) - API compatibility
- [tiktoken](https://github.com/openai/tiktoken) - Token counting library

## Support

For questions or issues:
1. Check existing [GitHub Issues](https://github.com/chris-colinsky/Zorac/issues)
2. Review documentation in `docs/` directory
3. Open a new issue with details

## License

MIT License - see [LICENSE](../LICENSE) for details.
