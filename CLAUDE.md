# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Self-hosted local LLM chat client** for running powerful AI models on consumer hardware. Zorac is a terminal-based ChatGPT alternative that emphasizes privacy, zero ongoing costs, and complete offline capability.

Interactive CLI chat client for a vLLM inference server running on local homelab/gaming GPUs (RTX 4090/3090/3080). Uses the OpenAI Python client library to communicate with a vLLM server via OpenAI-compatible API, with persistent sessions, automatic context management, Rich terminal UI, and performance metrics.

**Target Use Cases:** Privacy-focused development, AI experimentation without API costs, offline AI workflows, coding assistants, agentic systems (LangChain/LangGraph)

**Model**: `stelterlab/Mistral-Small-24B-Instruct-2501-AWQ` (24B parameters, 4-bit AWQ quantization, runs at 60-65 tokens/sec on RTX 4090)

**SEO Focus:** Documentation is optimized for discoverability by developers searching for "self-hosted LLM", "local AI", "ChatGPT alternative", "vLLM tutorial", and related terms.

## Documentation Structure

Zorac uses a modular documentation structure:

- **README.md** - Project overview, quick start, and navigation hub
- **docs/INSTALLATION.md** - Complete installation guide for all methods
- **docs/CONFIGURATION.md** - Configuration reference and examples
- **docs/USAGE.md** - Interactive commands and session management
- **docs/DEVELOPMENT.md** - Contributing, testing, and building
- **docs/SERVER_SETUP.md** - vLLM server setup and optimization
- **CLAUDE.md** - This file - AI assistant development guide
- **CHANGELOG.md** - Version history and release notes
- **CONTRIBUTING.md** - Contribution guidelines

When users ask about installation, configuration, or usage, refer them to the appropriate docs/ file.

## File Structure

- **zorac/**: Main package directory containing the application modules
  - **__init__.py**: Package exports and public API
  - **__main__.py**: Entry point for `python -m zorac`
  - **commands.py**: Command registry and help text generation
  - **config.py**: Configuration management and environment variables
  - **console.py**: Rich console singleton for terminal output
  - **llm.py**: LLM interaction and conversation summarization
  - **main.py**: Textual TUI application (ZoracApp), streaming, command handling
  - **markdown_custom.py**: Custom markdown renderer with left-aligned headings
  - **session.py**: Session persistence (save/load functionality)
  - **utils.py**: Utility functions (token counting, header display, connection checks)
- **tests/**: Comprehensive test suite with 28 test cases
  - **test_zorac.py**: Unit and integration tests
  - **__init__.py**: Test package marker
- **docs/**: User and developer documentation
  - **INSTALLATION.md**: Installation guide (Homebrew, pip, pipx, source)
  - **CONFIGURATION.md**: Configuration guide (server, token limits, model parameters)
  - **USAGE.md**: Usage guide (commands, session management, tips & tricks)
  - **DEVELOPMENT.md**: Development guide (testing, contributing, release process)
  - **SERVER_SETUP.md**: vLLM server setup and hardware optimization guide
- **.github/workflows/**: CI/CD automation
  - **release.yml**: PyPI publishing and GitHub releases
- **.env**: Configuration file for server settings (not committed to git)
- **.env.example**: Example configuration template
- **Makefile**: Development commands for testing, linting, formatting
- **.pre-commit-config.yaml**: Pre-commit hooks configuration
- **README.md**: Project overview and navigation hub
- **CHANGELOG.md**: Version history and release notes
- **CONTRIBUTING.md**: Contribution guidelines and development workflow
- **CLAUDE.md**: This file - AI assistant development guide
- **pyproject.toml**: Project metadata, dependencies, and tool configurations (managed by `uv`)
- **.python-version**: Python 3.13 (pinned)

## Architecture Overview

Zorac is organized as a modular Python package with clear separation of concerns:

### Package Modules

**zorac/commands.py** - Command Registry
- `COMMANDS`: Centralized list of all interactive commands with descriptions
- `get_help_text()`: Generate formatted help text for `/help` command display
- `get_system_prompt_commands()`: Generate command information for system prompt (enables LLM command awareness)
- Single source of truth for all command definitions

**zorac/config.py** - Configuration Management
- `load_config()`: Load configuration from `~/.zorac/config.json`
- `save_config(config)`: Save configuration to disk
- `get_setting(key, default)`: Get setting with priority: Env Var > Config File > Default
- `ensure_zorac_dir()`: Ensure `~/.zorac/` directory exists
- Constants: `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY`, `MAX_INPUT_TOKENS`, etc.

**zorac/session.py** - Session Management
- `save_session(messages)`: Persists conversation to `~/.zorac/session.json`
- `load_session()`: Restores conversation from disk

**zorac/utils.py** - Utility Functions
- `count_tokens(messages, encoding_name=None)`: Uses tiktoken to count tokens in conversation history (encoding defaults to configured `TIKTOKEN_ENCODING`)
- `print_header()`: Displays formatted welcome header with Rich panels and ASCII art logo
- `async check_connection(client: AsyncOpenAI)`: Verifies connection to vLLM server (async)

**zorac/llm.py** - LLM Operations
- `async summarize_old_messages(client: AsyncOpenAI, messages, auto=True)`: Auto-summarizes when approaching token limit with status spinner. Uses `VLLM_MODEL` from config.

**zorac/console.py** - Terminal Interface
- `console`: Shared Rich Console instance for all terminal output

**zorac/markdown_custom.py** - Custom Markdown Renderer
- `LeftAlignedMarkdown`: Custom markdown renderer that left-aligns all headings (monkey-patches Rich's Heading class)
- Removes centered heading panels for cleaner, more readable terminal output
- Maintains Rich styling (bold, colored) while forcing left justification

**zorac/main.py** - Main Application
- `ZoracApp(App)`: Main Textual application class providing the full terminal user interface
  - `compose()`: Yields `VerticalScroll#chat-log`, `ChatInput#user-input` (multiline TextArea with command suggestions), `Static#stats-bar`
  - `on_mount()` -> `_setup()`: Initialize `AsyncOpenAI` client, verify connection, load session, write header
  - `on_chat_input_submitted()`: Clears input, saves to history, routes to command handlers or `handle_chat()`
  - `on_key()`: Handles Up/Down arrow keys for command history navigation
  - `handle_chat()`: Adds user message to chat log, launches `_stream_response()` worker
  - `_stream_response()` (`@work`, `exclusive=True`): Streams LLM response using `Markdown.get_stream()`, updates `Static#stats-bar` in real-time
  - Command handlers: All `cmd_*()` methods for `/help`, `/quit`, `/config`, etc., output via `_log_system()` / `_log_user()` instead of `console.print()`
- `main()`: Runs first-time setup before Textual, then calls `ZoracApp().run()`
- `get_initial_system_message()`: Generate system message with command information for LLM awareness

### UI/UX Implementation

**Core Rendering:**
- **Textual Widgets**: All output is rendered via Textual widgets mounted in the application DOM
- **Streaming Markdown**: Assistant responses stream in real-time using Textual's `Markdown.get_stream()` API, yielding tokens into a live-updating `Markdown` widget inside `VerticalScroll#chat-log`
- **Markdown Rendering**: All assistant responses are rendered as formatted markdown via Textual's built-in `Markdown` widget
- **Formatted Panels**: Header displayed in styled panels with rounded box styles
- **Status Indicators**: Spinners and status messages for long-running operations (summarization)
- **Color Coding**: Consistent color scheme (blue=prompt, purple=assistant, green=success, red=error, yellow=warning)

**Input Bar:**
- **Multiline ChatInput Widget**: `ChatInput#user-input` (extends `TextArea`) docked to the bottom with placeholder text "Type your message or /command"
- **Enter/Shift+Enter**: Enter submits the message, Shift+Enter inserts a newline. Works in iTerm2 (via ctrl+j), kitty, and WezTerm (via Kitty keyboard protocol). In terminals where Shift+Enter is indistinguishable from Enter (e.g., macOS Terminal.app), Enter always submits; multiline input is still possible via paste.
- **Auto-Resize**: Input grows from 1 to 5 lines based on content
- **Command Suggestions**: Inline command suggestions as the user types `/` commands, accepted with Tab
- **Stats Bar**: `Static#stats-bar` widget docked below the input, showing contextual information:
  - Before any chat: "Ready" or session info (msg count + tokens)
  - After chat: response stats (tokens, duration, tok/s) and conversation totals

**Layout:**
- **Chat Log**: `VerticalScroll#chat-log` occupies the main area, containing all conversation messages as mounted widgets
- **Input**: `ChatInput#user-input` docked to the bottom for multiline text entry (Enter submits, Shift+Enter for newlines)
- **Stats Bar**: `Static#stats-bar` docked to the bottom below the input, updated in real-time during streaming

**Layout & Typography:**
- **Full-Width Output**: Assistant responses render at full terminal width
- **Left-Aligned Headings**: Custom `LeftAlignedMarkdown` renderer removes centered heading panels
  - H1: Bold white text with spacing
  - H2: Bold light gray text with spacing
  - H3+: Bold gray text
  - All headings left-aligned (not centered) for better readability
- **Code Blocks**: Syntax-highlighted code blocks automatically handled by Textual/Rich markdown
- **Responsive Design**: Content adapts to current console size

**Streaming Stats:**
- During streaming: `Static#stats-bar` is updated in real-time via `stats_bar.update()` showing token count, elapsed time, and tok/s
- Stats persist in the bottom bar across prompts without inline display
- After streaming: `self.stats` dict is updated and the stats bar reflects the latest response metrics

**Implementation Details:**
- `LeftAlignedMarkdown` monkey-patches Rich's `Heading.__rich_console__` method to force left alignment
- Textual CSS styles the input widget, stats bar, and chat log layout
- `_stream_response()` uses `@work(exclusive=True)` to run streaming in a background worker thread

### Key Features
- **Textual TUI**: Full terminal user interface built on the Textual framework with widgets, layout, streaming, and event handling
- **Persistent Sessions**: Auto-saves after each assistant response to `~/.zorac/session.json`
- **Auto-Summarization**: When conversation exceeds 12k tokens, summarizes older messages while preserving the last 6 messages
- **Streaming Responses**: Real-time token streaming with live markdown rendering via Textual's `Markdown.get_stream()` API
- **Rich Terminal UI**: Colored output, formatted panels, markdown rendering, and ASCII art logo
- **Token Tracking**: Real-time monitoring using tiktoken (configurable encoding, default `cl100k_base`)
- **Performance Metrics**: Real-time stats during streaming; persistent `Static#stats-bar` widget shows tokens/second, response time, and token usage
- **Command History**: Persistent history with Up/Down arrow navigation, stored in `~/.zorac/history`
- **Multiline Input**: Shift+Enter inserts newlines (works in iTerm2, kitty, WezTerm), Enter submits. Auto-resizes 1-5 lines. Pasting multiline text is preserved.
- **Command Suggestions**: Inline command suggestions for all `/commands` via `ChatInput` (TextArea with Tab completion)
- **Configurable Code Theme**: Syntax-highlighted code blocks use configurable Pygments theme (`CODE_THEME`, default `monokai`)
- **LLM Command Awareness**: System prompt includes command information, enabling the LLM to answer questions about Zorac functionality
- **Interactive Commands**: `/help`, `/clear`, `/save`, `/load`, `/tokens`, `/summarize`, `/summary`, `/config`, `/quit`, `/exit`

## Configuration

Configuration is managed through a `.env` file in the project root. Copy `.env.example` to `.env` and customize as needed.

### Server Configuration (via .env file or `~/.zorac/config.json`)
- `VLLM_BASE_URL`: Default `http://localhost:8000/v1` (vLLM server endpoint)
- `VLLM_API_KEY`: Default `"EMPTY"` (vLLM doesn't require authentication)
- `VLLM_MODEL`: Default `stelterlab/Mistral-Small-24B-Instruct-2501-AWQ`
- `ZORAC_DIR`: Optional - Custom zorac directory (defaults to `~/.zorac`)
- `ZORAC_SESSION_FILE`: Optional - Custom session file location (defaults to `~/.zorac/session.json`)
- `ZORAC_HISTORY_FILE`: Optional - Custom history file location (defaults to `~/.zorac/history`)
- `ZORAC_CONFIG_FILE`: Optional - Custom config file location (defaults to `~/.zorac/config.json`)

### Display
- `CODE_THEME`: Default `monokai` - Pygments theme for syntax-highlighted code blocks (any Pygments style name supported)

**Configuration Priority**: Environment Variable > Config File (`~/.zorac/config.json`) > Default Value

### Token Limits (Configurable)
- `MAX_INPUT_TOKENS`: Default `12000` - Max tokens for system prompt + chat history
- `MAX_OUTPUT_TOKENS`: Default `4000` - Max tokens for model responses
- `KEEP_RECENT_MESSAGES`: Default `6` - Messages preserved during auto-summarization

### Model Parameters (Configurable)
- `TEMPERATURE`: Default `0.1` - Controls randomness (0.0-2.0, lower = more deterministic)
- `STREAM`: Default `true` - Enable/disable real-time token streaming with live markdown rendering

### Token Counting (Configurable)
- `TIKTOKEN_ENCODING`: Default `cl100k_base` - Tiktoken encoding for token counting (change when using a model family with a different tokenizer, e.g. `o200k_base` for GPT-4o)

### Display (Configurable)
- `CODE_THEME`: Default `monokai` - Pygments theme for syntax-highlighted code blocks

**Note**: All token limits and model parameters can be configured via:
1. Environment variables in `.env` file
2. Runtime configuration in `~/.zorac/config.json`
3. The `/config` command during a chat session

## Development Commands

### Environment Setup
```bash
# Python 3.13 is required (specified in .python-version)
# Complete development setup (install deps + pre-commit hooks)
make dev-setup

# Or manually:
uv sync --extra dev
cp .env.example .env
# Edit .env to set your vLLM server URL
```

### Running the Application
```bash
# Run with Make
make run

# Or run as Python module
uv run python -m zorac

# Or run via console script (after installation)
uv run zorac

# Or activate venv and run directly
source .venv/bin/activate
zorac

# You can also override .env settings via environment variables
VLLM_BASE_URL="http://your-server:8000/v1" uv run zorac
```

### Installation as a CLI Tool
```bash
# Install globally (recommended)
uv tool install .

# Upgrade after pulling changes
uv tool upgrade zorac

# Uninstall
uv tool uninstall zorac

# Remove all data (optional)
rm -rf ~/.zorac
```

### Testing & Quality Assurance

```bash
# Run all tests
make test

# Run tests with coverage report (generates HTML report)
make test-coverage

# Run linter
make lint

# Auto-fix linting issues
make lint-fix

# Format code
make format

# Type checking
make type-check

# Run all checks (lint + type + test)
make all-checks

# Setup pre-commit hooks
make pre-commit-install

# Run pre-commit on all files
make pre-commit
```

### Test Coverage

The test suite uses pytest with pytest-asyncio (mode=auto) for async test support. It covers:
- **TestCountTokens**: Token counting with various message types and edge cases
- **TestSessionManagement**: Save/load functionality, error handling, invalid inputs
- **TestPrintHeader**: UI header rendering
- **TestSummarizeOldMessages**: Async message summarization logic and API error handling
- **TestSummarizeFormat**: Summary message format and extraction
- **TestConfiguration**: Configuration validation and defaults
- **TestConfigurationExtended**: Config bounds and save_config behavior
- **TestDirectoryManagement**: `ensure_zorac_dir` creation logic
- **TestIntegration**: End-to-end workflows (save/load roundtrips)
- **TestTokenCountAfterSaveLoad**: Token count consistency across persistence
- **TestCheckConnectionAsync**: Async connection verification (success and failure)
- **TestHelpFeatureIntegration**: System message, help output, and token budget

**Coverage targets:**
- Minimum 80% code coverage
- All core functions tested
- Edge cases and error paths covered

### Interactive Commands (in the CLI)
- `/help` - Show all available commands with descriptions
- `/clear` - Reset conversation to initial system message and save
- `/save` - Manually save session to disk
- `/load` - Reload session from disk (discards unsaved changes)
- `/tokens` - Display current token usage, limits, and remaining capacity
- `/summarize` - Force summarization of conversation history (even if below token limit)
- `/summary` - Display the current conversation summary (if one exists)
- `/config` - Manage configuration settings
  - `/config list` - Show current configuration
  - `/config set <KEY> <VALUE>` - Set a configuration value
  - `/config get <KEY>` - Get a specific configuration value
- `/quit` or `/exit` - Save and exit
- `Ctrl+C` - Interrupt response (continues chat session)

## vLLM Server Setup

For server configuration details, see **docs/SERVER_SETUP.md**. Key points:

- **Hardware**: RTX 4090 (24GB VRAM)
- **Quantization**: AWQ with Marlin kernel (`--quantization awq_marlin`)
- **Context Length**: 16,384 tokens (`--max-model-len 16384`)
- **Concurrent Users**: 32 (`--max-num-seqs 32` - prevents OOM on 24GB cards)
- **Tool Support**: Enabled for LangChain/LangGraph (`--enable-auto-tool-choice --tool-call-parser mistral`)

## Dependencies

### Production Dependencies
- **openai** (>=2.8.1): OpenAI client library for vLLM API (uses `AsyncOpenAI` for async operations)
- **tiktoken** (>=0.8.0): Token counting (encoding configurable via `TIKTOKEN_ENCODING`, default `cl100k_base`)
- **python-dotenv** (>=1.0.0): Environment variable management from .env files
- **rich** (>=14.2.0): Rich terminal formatting, markdown rendering, and live updates
- **textual** (>=1.0.0): Modern TUI framework providing the full terminal user interface with widgets, layout, streaming, and event handling

### Development Dependencies
- **pytest** (>=8.0.0): Testing framework
- **pytest-asyncio** (>=0.23.0): Async test support (configured with `asyncio_mode = "auto"`)
- **pytest-cov** (>=4.1.0): Coverage reporting for pytest
- **pytest-mock** (>=3.12.0): Mock fixtures for pytest
- **ruff** (>=0.8.0): Fast Python linter and formatter
- **mypy** (>=1.8.0): Static type checker
- **pre-commit** (>=3.6.0): Git hook management

## Distribution

Zorac is distributed via:
- **PyPI**: `pip install zorac` / `pipx install zorac`
- **Homebrew**: `brew tap chris-colinsky/zorac && brew install zorac`

### Release Workflow
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Merge changes to `main`
4. Push version tag (e.g., `git tag -a v1.2.0 -m "Release 1.2.0" && git push origin v1.2.0`)
5. GitHub Actions publishes to PyPI via Trusted Publisher (OIDC)
6. Manually update Homebrew formula in `chris-colinsky/homebrew-zorac`

## Documentation Strategy

### SEO/AEO Optimization
The project documentation is optimized for search engines and AI answer engines (ChatGPT, Claude, Perplexity) to maximize discoverability:

**Target Keywords:**
- Primary: "self-hosted LLM", "local AI", "ChatGPT alternative", "vLLM setup"
- Secondary: "RTX 4090 AI", "homelab LLM", "offline AI", "privacy-focused AI"
- Long-tail: "how to run llm locally", "free chatgpt alternative", "gaming pc ai server"

**README.md Structure:**
- SEO-optimized title and subtitle emphasizing "self-hosted" and "local"
- "Why Self-Host?" section addressing common search queries
- Supported Hardware table for GPU-specific searches
- Use Cases section for application-specific queries
- Comparison table (Zorac vs Cloud APIs) for decision-making searches
- Comprehensive FAQ answering common questions
- "Also Useful For" section expanding use case coverage

**docs/SERVER_SETUP.md Strategy:**
- Beginner-friendly intro emphasizing "$0/month" and "complete privacy"
- Targets tutorial-seeking users ("how to host LLM locally")
- Mentions multiple GPU models for broader hardware search coverage

**GitHub Metadata (see GITHUB_METADATA.md):**
- 20+ repository topics covering all relevant keywords
- Description optimized for GitHub search and Google indexing
- Social preview image recommended for visual appeal
- Suggestions for awesome-list submissions and community engagement
