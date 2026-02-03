# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-02-02

### Changed
- **Distribution**: Migrated from binary releases to PyPI + Homebrew
  - Install via `pip install zorac` or `pipx install zorac`
  - Install via `brew tap chris-colinsky/zorac && brew install zorac`
  - Removed PyInstaller binary builds
  - Automated releases via GitHub Actions with Trusted Publisher (OIDC)

### Removed
- Binary releases (zorac-linux-x86_64, zorac-macos-arm64)
- PyInstaller build files (`zorac.spec`, `run_zorac.py`)

### Improved
- Simpler installation process - standard Python packaging
- Automatic dependency management via pip/brew
- Easier upgrades (`pip install --upgrade zorac` or `brew upgrade zorac`)
- No more macOS Gatekeeper warnings for unsigned binaries

### Migration Notes
- Users of binary releases should switch to pip/pipx/brew installation
- All functionality remains the same
- Configuration in `~/.zorac/` is preserved

## [1.1.0] - 2026-02-01

### Added
- **`/help` Command**: Interactive command to display all available commands with descriptions
  - Shows formatted list of all commands (zorac/main.py:188-191)
  - Includes special formatting for `/config` subcommands
  - Provides quick reference for users without leaving the chat

- **Command Registry System**: Centralized command management infrastructure (zorac/commands.py)
  - `COMMANDS` list with TypedDict definitions for type safety
  - `get_help_text()` - Generates formatted help text for `/help` display
  - `get_system_prompt_commands()` - Generates command descriptions for LLM context
  - Single source of truth for all command definitions

- **LLM Command Awareness**: Enhanced system prompt with command information
  - LLM can now answer questions about Zorac's functionality
  - Suggests relevant commands naturally in conversation
  - Provides usage examples and explanations when helpful

- **Enhanced Markdown Rendering**: Custom renderer with improved layout (zorac/markdown_custom.py)
  - Left-aligned headings (H1, H2, H3) instead of centered panels
  - Cleaner, more readable terminal output
  - Monkey-patches Rich's Heading class for consistent styling

- **Content Width Constraint**: Optimal reading width for terminal output
  - `ConstrainedWidth` class constrains content to 60% of console width (zorac/main.py:41-58)
  - Prevents overly wide text that's hard to read
  - Stable layout that only reflows when window shrinks below constraint
  - Applied to all assistant responses and conversation summaries

- **Multi-GPU Training Guide**: Comprehensive documentation for training setups (docs/TEST_TRAINING.md)
  - 220-line guide covering multi-GPU configuration
  - Detailed setup instructions and optimization tips
  - Training best practices and troubleshooting

- **Test Coverage**: New test suite for command registry (tests/test_commands.py)
  - 154 lines of comprehensive tests
  - Validates command structure, help text generation, and system prompt formatting
  - Ensures command registry integrity

### Changed
- **Stats Display**: Changed from Rich Panel to plain text for cleaner appearance
  - Stats now displayed as dim text without border
  - Reduced visual clutter in terminal output

- **System Message**: Enhanced initial system message with command information
  - Replaces simple "You are a helpful assistant" with command-aware prompt
  - Enables LLM to understand and reference Zorac functionality

- **Documentation Updates**:
  - **README.md**: Improved clarity and added new feature descriptions
  - **DEVELOPMENT.md**: Expanded with 178 additional lines of developer guidance
  - **SERVER_SETUP.md**: Restructured for better organization (213 lines modified)
  - **USAGE.md**: Added 36 lines documenting `/help` command and new features
  - **CLAUDE.md**: Enhanced with 46 lines covering new architecture components

### Improved
- **Code Organization**: Command definitions centralized in dedicated module
- **Type Safety**: Full TypedDict annotations for command structures
- **Documentation**: Comprehensive multi-GPU training documentation
- **User Experience**: More discoverable commands via `/help` and LLM awareness
- **Readability**: Better terminal layout with constrained width and left-aligned headings

### Technical Details
- Added `zorac/commands.py` module (134 lines)
- Added `zorac/markdown_custom.py` module (43 lines)
- Enhanced `zorac/main.py` with 76 additional lines
- Added `tests/test_commands.py` (154 lines)
- Extended `tests/test_zorac.py` with 64 additional lines
- Total: ~1,375 additions, 132 deletions across 13 files

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

[1.2.0]: https://github.com/chris-colinsky/zorac/releases/tag/v1.2.0
[1.1.0]: https://github.com/chris-colinsky/zorac/releases/tag/v1.1.0
[1.0.0]: https://github.com/chris-colinsky/zorac/releases/tag/v1.0.0
