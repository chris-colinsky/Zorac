# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Self-hosted local LLM chat client** for running powerful AI models on consumer hardware. Zorac is a terminal-based ChatGPT alternative that emphasizes privacy, zero ongoing costs, and complete offline capability.

Interactive CLI chat client for a vLLM inference server running on local homelab/gaming GPUs (RTX 4090/3090/3080). Uses the OpenAI Python client library to communicate with a vLLM server via OpenAI-compatible API, with persistent sessions, automatic context management, Rich terminal UI, and performance metrics.

**Target Use Cases:** Privacy-focused development, AI experimentation without API costs, offline AI workflows, coding assistants, agentic systems (LangChain/LangGraph)

**Model**: `stelterlab/Mistral-Small-24B-Instruct-2501-AWQ` (24B parameters, 4-bit AWQ quantization, runs at 60-65 tokens/sec on RTX 4090)

**SEO Focus:** Documentation is optimized for discoverability by developers searching for "self-hosted LLM", "local AI", "ChatGPT alternative", "vLLM tutorial", and related terms.

## File Structure

- **zorac.py**: Main interactive CLI application (single-file implementation)
- **tests/**: Comprehensive test suite with 30+ test cases
  - **test_zorac.py**: Unit and integration tests
  - **__init__.py**: Test package marker
- **.env**: Configuration file for server settings (not committed to git)
- **.env.example**: Example configuration template
- **Makefile**: Development commands for testing, linting, formatting
- **.pre-commit-config.yaml**: Pre-commit hooks configuration
- **SERVER_SETUP.md**: Complete self-hosting guide with hardware optimization
- **README.md**: User documentation optimized for SEO/discoverability
- **GITHUB_METADATA.md**: Repository metadata and SEO recommendations
- **pyproject.toml**: Project metadata, dependencies, and tool configurations (managed by `uv`)
- **.python-version**: Python 3.13 (pinned)

## Architecture Overview

**zorac.py** is a single-file application with the following key components:

### Core Functions
- `count_tokens(messages)`: Uses tiktoken to count tokens in conversation history
- `save_session(messages)`: Persists conversation to `~/.zorac_session.json`
- `load_session()`: Restores conversation from disk
- `print_header()`: Displays formatted welcome header with Rich panels
- `summarize_old_messages(client, messages)`: Auto-summarizes when approaching token limit with status spinner
- `main()`: Interactive loop handling user input, streaming API calls, and session management

### UI/UX Implementation
- **Rich Console**: All output uses Rich library for colored, formatted terminal display
- **Live Streaming**: Assistant responses stream in real-time using `Live()` context manager
- **Markdown Rendering**: All assistant responses are rendered as formatted markdown
- **Formatted Panels**: Header and stats displayed in styled panels with rounded/simple box styles
- **Status Indicators**: Spinners and status messages for long-running operations (summarization)
- **Color Coding**: Consistent color scheme (blue=user, purple=assistant, green=success, red=error, yellow=warning)

### Key Features
- **Persistent Sessions**: Auto-saves after each assistant response to `~/.zorac_session.json`
- **Auto-Summarization**: When conversation exceeds 12k tokens, summarizes older messages while preserving the last 6 messages
- **Streaming Responses**: Real-time token streaming with live markdown rendering
- **Rich Terminal UI**: Colored output, formatted panels, and markdown rendering using Rich library
- **Token Tracking**: Real-time monitoring using tiktoken (cl100k_base encoding)
- **Performance Metrics**: Displays tokens/second, response time, and token usage after each interaction
- **Interactive Commands**: `/clear`, `/save`, `/load`, `/tokens`, `/summarize`, `/summary`, `/quit`, `/exit`

## Configuration

Configuration is managed through a `.env` file in the project root. Copy `.env.example` to `.env` and customize as needed.

### Server Configuration (via .env file)
- `VLLM_BASE_URL`: Default `http://localhost:8000/v1` (vLLM server endpoint)
- `VLLM_API_KEY`: Default `"EMPTY"` (vLLM doesn't require authentication)
- `VLLM_MODEL`: Default `stelterlab/Mistral-Small-24B-Instruct-2501-AWQ`
- `ZORAC_SESSION_FILE`: Optional - Custom session file location (defaults to `~/.zorac_session.json`)

### Token Limits
- `MAX_INPUT_TOKENS`: `12000` - Max tokens for system prompt + chat history
- `MAX_OUTPUT_TOKENS`: `4000` - Max tokens for model responses
- `KEEP_RECENT_MESSAGES`: `6` - Messages preserved during auto-summarization

### Model Parameters (in API calls)
- **Temperature**: `0.1` (deterministic responses)
- **Streaming**: `stream=True` (real-time token display with live markdown rendering)
- **Max Tokens**: `4000` (maximum response length)

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

# Or run with uv (uses .env file)
uv run zorac.py

# Or activate venv and run directly
source .venv/bin/activate
python zorac.py

# You can also override .env settings via environment variables
VLLM_BASE_URL="http://your-server:8000/v1" uv run zorac.py
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
- `/clear` - Reset conversation to initial system message and save
- `/save` - Manually save session to disk
- `/load` - Reload session from disk (discards unsaved changes)
- `/tokens` - Display current token usage, limits, and remaining capacity
- `/summarize` - Force summarization of conversation history (even if below token limit)
- `/summary` - Display the current conversation summary (if one exists)
- `/quit` or `/exit` - Save and exit
- `Ctrl+C` - Interrupt response (continues chat session)

## vLLM Server Setup

For server configuration details, see **SERVER_SETUP.md**. Key points:

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

**SERVER_SETUP.md Strategy:**
- Beginner-friendly intro emphasizing "$0/month" and "complete privacy"
- Targets tutorial-seeking users ("how to host LLM locally")
- Mentions multiple GPU models for broader hardware search coverage

**GitHub Metadata (see GITHUB_METADATA.md):**
- 20+ repository topics covering all relevant keywords
- Description optimized for GitHub search and Google indexing
- Social preview image recommended for visual appeal
- Suggestions for awesome-list submissions and community engagement
