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
  - **main.py**: Main event loop and interactive CLI logic
  - **session.py**: Session persistence (save/load functionality)
  - **utils.py**: Utility functions (token counting, header display, connection checks)
- **tests/**: Comprehensive test suite with 28 test cases
  - **test_zorac.py**: Unit and integration tests
  - **__init__.py**: Test package marker
- **docs/**: User and developer documentation
  - **INSTALLATION.md**: Complete installation guide (binary, source, development)
  - **CONFIGURATION.md**: Configuration guide (server, token limits, model parameters)
  - **USAGE.md**: Usage guide (commands, session management, tips & tricks)
  - **DEVELOPMENT.md**: Development guide (testing, contributing, building binaries)
  - **SERVER_SETUP.md**: vLLM server setup and hardware optimization guide
- **.github/workflows/**: CI/CD automation
  - **release.yml**: Automated binary builds and GitHub releases
- **.env**: Configuration file for server settings (not committed to git)
- **.env.example**: Example configuration template
- **Makefile**: Development commands for testing, linting, formatting
- **.pre-commit-config.yaml**: Pre-commit hooks configuration
- **run_zorac.py**: Entry point script for PyInstaller binary builds
- **zorac.spec**: PyInstaller specification for binary packaging
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
- `count_tokens(messages)`: Uses tiktoken to count tokens in conversation history
- `print_header()`: Displays formatted welcome header with Rich panels and ASCII art logo
- `check_connection(client)`: Verifies connection to vLLM server

**zorac/llm.py** - LLM Operations
- `summarize_old_messages(client, messages)`: Auto-summarizes when approaching token limit with status spinner

**zorac/console.py** - Terminal Interface
- `console`: Shared Rich Console instance for all terminal output

**zorac/main.py** - Main Application
- `main()`: Interactive loop handling user input, streaming API calls, and session management
- `get_initial_system_message()`: Generate system message with command information for LLM awareness
- `setup_readline()`: Configure command history and persistent storage

### UI/UX Implementation
- **Rich Console**: All output uses Rich library for colored, formatted terminal display
- **Live Streaming**: Assistant responses stream in real-time using `Live()` context manager
- **Markdown Rendering**: All assistant responses are rendered as formatted markdown
- **Formatted Panels**: Header and stats displayed in styled panels with rounded/simple box styles
- **Status Indicators**: Spinners and status messages for long-running operations (summarization)
- **Color Coding**: Consistent color scheme (blue=user, purple=assistant, green=success, red=error, yellow=warning)

### Key Features
- **Persistent Sessions**: Auto-saves after each assistant response to `~/.zorac/session.json`
- **Auto-Summarization**: When conversation exceeds 12k tokens, summarizes older messages while preserving the last 6 messages
- **Streaming Responses**: Real-time token streaming with live markdown rendering
- **Rich Terminal UI**: Colored output, formatted panels, markdown rendering, and ASCII art logo
- **Token Tracking**: Real-time monitoring using tiktoken (cl100k_base encoding)
- **Performance Metrics**: Displays tokens/second, response time, and token usage after each interaction
- **Command History**: Persistent readline history stored in `~/.zorac/history`
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

**Configuration Priority**: Environment Variable > Config File (`~/.zorac/config.json`) > Default Value

### Token Limits (Configurable)
- `MAX_INPUT_TOKENS`: Default `12000` - Max tokens for system prompt + chat history
- `MAX_OUTPUT_TOKENS`: Default `4000` - Max tokens for model responses
- `KEEP_RECENT_MESSAGES`: Default `6` - Messages preserved during auto-summarization

### Model Parameters (Configurable)
- `TEMPERATURE`: Default `0.1` - Controls randomness (0.0-2.0, lower = more deterministic)
- `STREAM`: Default `true` - Enable/disable real-time token streaming with live markdown rendering

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

The test suite covers:
- **TestCountTokens**: Token counting with various message types and edge cases
- **TestSessionManagement**: Save/load functionality, error handling, invalid inputs
- **TestPrintHeader**: UI header rendering
- **TestSummarizeOldMessages**: Message summarization logic and API error handling
- **TestConfiguration**: Configuration validation
- **TestIntegration**: End-to-end workflows (save/load roundtrips, token consistency)

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
- **openai** (>=2.8.1): OpenAI client library for vLLM API
- **tiktoken** (>=0.8.0): Token counting (uses cl100k_base encoding)
- **python-dotenv** (>=1.0.0): Environment variable management from .env files
- **rich** (>=14.2.0): Rich terminal formatting, markdown rendering, and live updates

### Development Dependencies
- **pytest** (>=8.0.0): Testing framework
- **pytest-cov** (>=4.1.0): Coverage reporting for pytest
- **pytest-mock** (>=3.12.0): Mock fixtures for pytest
- **ruff** (>=0.8.0): Fast Python linter and formatter
- **mypy** (>=1.8.0): Static type checker
- **pre-commit** (>=3.6.0): Git hook management

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
