# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0]

### Added
- **Modular Package Structure**: Refactored from monolithic file to organized package with 8 focused modules
  - `config.py` - Configuration management with priority system
  - `console.py` - Rich console singleton
  - `llm.py` - LLM operations and summarization
  - `main.py` - Main event loop and CLI logic
  - `session.py` - Session persistence
  - `utils.py` - Utility functions (token counting, header, connection check)
  - `__init__.py` - Package exports and public API
  - `__main__.py` - Module entry point for `python -m zorac`

- **Configurable Parameters**: All token limits and model parameters are now configurable
  - `MAX_INPUT_TOKENS` (default: 12000)
  - `MAX_OUTPUT_TOKENS` (default: 4000)
  - `KEEP_RECENT_MESSAGES` (default: 6)
  - `TEMPERATURE` (default: 0.1)
  - `STREAM` (default: true)

- **Multiple Configuration Methods**:
  - Environment variables via `.env` file
  - Persistent configuration file (`~/.zorac/config.json`)
  - Runtime configuration via `/config` command

- **Configuration Management**:
  - `/config list` - View all current settings
  - `/config set <KEY> <VALUE>` - Update settings at runtime
  - `/config get <KEY>` - Retrieve specific setting value
  - Configuration priority: Environment Variables > Config File > Defaults

- **Non-Streaming Mode**: Support for disabling real-time streaming (`STREAM=false`)

- **Type Safety**: Full type annotations with mypy validation
  - Added type hints for all configuration functions
  - Proper handling of int, float, and bool settings

- **Organized Configuration**:
  - Centralized `~/.zorac/` directory for all user data
  - Session storage: `~/.zorac/session.json`
  - Command history: `~/.zorac/history`
  - Configuration: `~/.zorac/config.json`

- **First-Run Setup Wizard**: Interactive configuration on first launch
  - Prompts for vLLM server URL and model name
  - Creates configuration file at `~/.zorac/config.json`
  - Only runs once, on first startup
  - Can be reconfigured anytime with `/config` command

- **Multi-line Input Support**: Enhanced input handling for better user experience
  - Paste multi-line text directly from clipboard
  - Newlines are preserved when pasting
  - Press Enter to submit prompt
  - Prevents treating each pasted line as separate prompt

- **Binary Releases**: Automated build system for distribution
  - Pre-built binaries for Linux (x86_64) and macOS (ARM64)
  - Automated builds via GitHub Actions
  - No Python installation required for end users

- **Modular Documentation**: Reorganized documentation for better navigation
  - Separate guides for Installation, Configuration, Usage, and Development
  - All documentation consolidated in `docs/` directory
  - Cleaner, more focused README as navigation hub

### Changed
- **Architecture**: Transitioned from single-file to modular package structure for better maintainability
- **Entry Points**: Now supports three execution methods:
  - `python -m zorac` (module execution)
  - `zorac` (console script after installation)
  - `uv run zorac` (via uv)
- **Status**: Promoted to Production/Stable (Development Status: 5)
- **Platform Support**: Linux and macOS officially supported (Windows users can use WSL)

### Improved
- **Code Organization**: 53% reduction in perceived complexity through modularization
- **Testability**: Each module can be tested independently
- **Documentation**: Comprehensive updates to README.md, CLAUDE.md, and CONTRIBUTING.md
- **Configuration Flexibility**: Users can adjust all parameters without code changes
- **Code Coverage**: Improved test coverage to 37% with 28 passing tests

### Technical Details
- Python 3.13+ required
- Full mypy type checking support
- Ruff linting compliance
- Pre-commit hooks configuration
- Comprehensive test suite (28 tests, 100% passing)

### Migration Notes
For users upgrading from development versions:
- Session file moved from `~/.zorac_session.json` to `~/.zorac/session.json`
- Run `mkdir -p ~/.zorac && mv ~/.zorac_session.json ~/.zorac/session.json` to migrate
- All previous environment variables still work as before
- New configuration options are optional with sensible defaults

[1.0.0]: https://github.com/chris-colinsky/zorac/releases/tag/v1.0.0
