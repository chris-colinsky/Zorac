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
git clone https://github.com/yourusername/zorac.git
cd zorac

# Install dependencies
uv sync

# Run the application
uv run main.py
```

## Code Style

- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Keep functions focused and well-documented
- Maintain the single-file architecture unless there's a compelling reason to split

## Project Structure

This is intentionally a single-file application (`main.py`) for simplicity. Major architectural changes should be discussed in an issue first.

## Testing

Currently, this project doesn't have automated tests. Contributions that add testing infrastructure are welcome!

Manual testing checklist:
- [ ] Application starts without errors
- [ ] Can connect to vLLM server
- [ ] Chat responses work correctly
- [ ] All commands work (`/clear`, `/save`, `/load`, `/tokens`, `/quit`)
- [ ] Session persistence works
- [ ] Auto-summarization triggers correctly
- [ ] Token counting displays accurately
- [ ] Environment variables are respected

## Areas for Contribution

Some ideas for contributions:
- **Streaming responses**: Implement real-time token streaming
- **Configuration file**: Support for `.env` or config file
- **Better error handling**: Improved error messages and recovery
- **Tests**: Unit tests and integration tests
- **Color output**: Syntax highlighting for responses
- **History search**: Search through conversation history
- **Multiple model support**: Easy switching between models
- **Session management**: Named sessions, session list/search
- **Performance improvements**: Optimize token counting or caching

## Questions?

Feel free to open an issue for any questions about contributing!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
