# Contributing to Zorac

Thank you for considering contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, vLLM version)
- Relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the enhancement
- The motivation/use case for the feature
- Any implementation ideas you have

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes**:
   - Follow the existing code style
   - Keep changes focused and atomic
   - Add comments for complex logic
3. **Test your changes**:
   - Ensure the application runs without errors
   - Test with a running vLLM server
   - Verify all interactive commands work
4. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference any related issues
5. **Submit a pull request**:
   - Describe what your PR does
   - Link any related issues
   - Explain any breaking changes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/chris-colinsky/zorac.git
cd zorac

# Install development dependencies (includes pytest, ruff, mypy, etc.)
uv sync --extra dev

# Install pre-commit hooks
make pre-commit-install

# Or use the shorthand for complete setup
make dev-setup

# Run the application
make run
# or
uv run zorac
```

## Code Style

- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Keep functions focused and well-documented
- Run linter before committing: `make lint-fix`
- Format code with ruff: `make format`
- Run type checker: `make type-check`

## Project Structure

Zorac is organized as a modular Python package:

```
zorac/
├── __init__.py      # Package exports and public API
├── __main__.py      # Entry point for python -m zorac
├── config.py        # Configuration management
├── console.py       # Rich console singleton
├── llm.py          # LLM operations and summarization
├── main.py         # Main event loop and CLI
├── session.py      # Session persistence
└── utils.py        # Utility functions
```

Major architectural changes should be discussed in an issue first.

## Testing

This project has comprehensive automated tests. Please add tests for new features.

```bash
# Run all tests
make test

# Run tests with coverage
make test-coverage

# Run all quality checks (lint + type-check + test)
make all-checks
```

Manual testing checklist:
- [ ] Application starts without errors
- [ ] Can connect to vLLM server
- [ ] Chat responses work correctly
- [ ] All commands work (`/clear`, `/save`, `/load`, `/tokens`, `/summarize`, `/summary`, `/config`, `/quit`)
- [ ] Session persistence works
- [ ] Auto-summarization triggers correctly
- [ ] Token counting displays accurately
- [ ] Environment variables are respected
- [ ] Runtime configuration via `/config` works
- [ ] Command history persists across sessions

## Areas for Contribution

Some ideas for contributions:
- **History search**: Search through conversation history
- **Multiple model support**: Easy switching between models in runtime
- **Session management**: Named sessions, session list/search
- **Performance improvements**: Optimize token counting or caching
- **Enhanced configuration**: Web UI for configuration management
- **Export functionality**: Export conversations to markdown/PDF
- **Plugin system**: Allow custom commands and extensions
- **Better error recovery**: More resilient connection handling
- **Conversation analytics**: Statistics about token usage over time

## Questions?

Feel free to open an issue for any questions about contributing!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
