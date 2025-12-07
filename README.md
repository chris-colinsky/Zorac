# Zorac - Self-Hosted Local LLM Chat Client

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![vLLM](https://img.shields.io/badge/vLLM-compatible-purple.svg)

A fun terminal chat client for running **local LLMs on consumer hardware**. Chat with powerful AI models like Mistral-24B privately on your own RTX 4090/3090 - no cloud, no costs, complete privacy.

Perfect for developers who want a **self-hosted ChatGPT alternative** running on their gaming PC or homelab server.  Also good for local AI coding assistants, agentic workflows and agent development.

> Named after ZORAC, the intelligent Ganymean computer from James P. Hogan's *The Gentle Giants of Ganymede*.

## Why Self-Host Your LLM?

- 💰 **Zero ongoing costs** - No API fees, run unlimited queries
- 🔒 **Complete privacy** - Your data never leaves your machine
- ⚡ **Low latency** - Sub-second responses on local hardware
- 🎮 **Use existing hardware** - Your gaming GPU works great for AI
- 🔧 **Full control** - Customize models, parameters, and behavior
- 📡 **Work offline** - No internet required after initial setup

## Features

- **Interactive CLI** - Natural conversation flow with continuous input prompts
- **Rich Terminal UI** - Beautiful formatted output with colors, panels, and markdown rendering
- **Streaming Responses** - Real-time token streaming with live markdown display
- **Persistent Sessions** - Automatically saves and restores conversation history
- **Smart Context Management** - Automatically summarizes old messages when approaching token limits
- **Token Tracking** - Real-time monitoring of token usage with tiktoken
- **Performance Metrics** - Displays tokens/second, response time, and resource usage
- **Session Commands** - Save, load, and manage conversation history
- **Auto-Recovery** - Graceful error handling keeps the chat running

## Demo

### Rich Terminal UI with Live Streaming

![Zorac Chat Interface](screenshots/zorac-screenshot-1.png)

*Interactive chat with real-time streaming responses, markdown rendering, and performance metrics*

### Token Management & Commands

![Token Usage and Commands](screenshots/zorac-screenshot-2.png)

*Built-in commands for session management and token tracking*

## Supported Hardware

This works on **consumer gaming GPUs**:

| GPU | VRAM | Model Size | Performance |
|-----|------|------------|-------------|
| RTX 4090 | 24GB | Up to 24B (AWQ) | 60-65 tok/s ⭐ |
| RTX 3090 Ti | 24GB | Up to 24B (AWQ) | 55-60 tok/s |
| RTX 3090 | 24GB | Up to 24B (AWQ) | 55-60 tok/s |
| RTX 4080 | 16GB | Up to 14B (AWQ) | 45-50 tok/s |
| RTX 4070 Ti | 12GB | Up to 7B (AWQ) | 40-45 tok/s |
| RTX 3080 | 10GB | Up to 7B (AWQ) | 35-40 tok/s |

⭐ Recommended configuration. See [SERVER_SETUP.md](SERVER_SETUP.md) for optimization details.

## Use Cases

- 💬 **Local ChatGPT alternative** - Private conversations, no data collection
- 👨‍💻 **Coding assistant** - Works with Continue.dev, Cline, and other AI coding tools
- 🤖 **Agentic workflows** - LangChain/LangGraph running entirely local
- 📝 **Content generation** - Write, summarize, analyze - all offline
- 🔬 **AI experimentation** - Test prompts and models without API costs
- 🎓 **Learning AI/ML** - Understand LLM inference without cloud dependencies

## Why This Model?

This application is optimized for `Mistral-Small-24B-Instruct-2501-AWQ`.

- **Superior Intelligence**: At 24 billion parameters, it offers significantly better reasoning and instruction following than standard 7B/8B models, making it ideal for complex agentic workflows.
- **Consumer Hardware Ready**: Using 4-bit AWQ quantization allows this powerful model to fit entirely within the 24GB VRAM of a single RTX 4090 or RTX 3090.
- **High Performance**: The AWQ formulation with the Marlin kernel enables extremely fast inference (60-65 t/s), providing a responsive, real-time chat experience that heavier 70B models cannot match on single cards.

## Zorac vs. Cloud LLM APIs

| Feature | Zorac (Self-Hosted) | OpenAI API | Anthropic API |
|---------|---------------------|------------|---------------|
| Monthly cost | **$0** | ~$50-500+ | ~$50-500+ |
| Privacy | ✅ Complete | ❌ Data sent to cloud | ❌ Data sent to cloud |
| Latency | ✅ <100ms | ⚠️ 200-1000ms | ⚠️ 200-1000ms |
| Works offline | ✅ Yes | ❌ Requires internet | ❌ Requires internet |
| Rate limits | ✅ None | ⚠️ Yes | ⚠️ Yes |
| Initial cost | ~$1600 (GPU) | $0 | $0 |
| Model variety | ⚠️ One at a time | ✅ Multiple models | ✅ Multiple models |
| Best for | Privacy, experimentation, unlimited use | Production, variety | Production, quality |

## Quick Start

For detailed instructions on setting up the vLLM inference server, please refer to [SERVER_SETUP.md](SERVER_SETUP.md).

### Prerequisites

This project uses `uv` for dependency management. You must have `uv` installed.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using Homebrew (macOS/Linux)
brew install uv
```

### Installation

```bash
# Clone the repository
git clone https://github.com/chris-colinsky/zorac.git
cd zorac

# Install dependencies (uv will automatically create a virtual environment)
uv sync

# Copy the example .env file and configure your server
cp .env.example .env
# Edit .env to set your vLLM server URL
```

### Running the Client

```bash
# Activate the virtual environment (optional if using uv run)
# source .venv/bin/activate

# Run using uv
uv run zorac.py
```

## Usage

### Basic Interaction

Simply type your message at the `You:` prompt and press Enter. The assistant will respond, and the conversation will be automatically saved.

```
You: Explain quantum entanglement
Assistant: [Response...]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history and start fresh |
| `/save` | Manually save the current session |
| `/load` | Reload session from disk |
| `/tokens` | Display current token usage and limits |
| `/summarize` | Force summarization of conversation history |
| `/summary` | Display the current conversation summary (if exists) |
| `/quit` or `/exit` | Save session and exit |
| `Ctrl+C` | Interrupt current operation without exiting |

### Example Usage

```bash
# Start the chat client
$ uv run zorac.py

# Ask questions naturally
You: Explain how Python async/await works

# Get streaming responses with markdown formatting
Assistant: [Streams formatted response with code examples...]
Stats: 245 tokens in 3.8s (64.5 tok/s) | Total: 4 msgs | Tokens: 312/12000

# Use commands for session management
You: /tokens
Current: 312 tokens | Limit: 12000 | Remaining: 11688

You: /quit
Session saved. Goodbye!
```


## Configuration

### Server Configuration

The client uses a `.env` file for configuration. After installation, copy `.env.example` to `.env` and customize:

```bash
# .env file
VLLM_BASE_URL=http://localhost:8000/v1  # Change to your vLLM server URL
VLLM_API_KEY=EMPTY
VLLM_MODEL=stelterlab/Mistral-Small-24B-Instruct-2501-AWQ

# Optional: Custom session file location
# ZORAC_SESSION_FILE=/path/to/custom/session.json
```

You can also override settings using environment variables:

```bash
# Override .env settings
VLLM_BASE_URL="http://YOUR_SERVER_IP:8000/v1" uv run zorac.py
```

### Token Limits

Adjust these constants at the top of `zorac.py`:

```python
SESSION_FILE = Path.home() / ".zorac_session.json"  # Session storage location
MAX_INPUT_TOKENS = 12000   # Maximum tokens for input (system + history)
MAX_OUTPUT_TOKENS = 4000   # Maximum tokens for model responses
KEEP_RECENT_MESSAGES = 6   # Messages to preserve during auto-summarization
```

### Model Parameters

```python
temperature=0.1,      # Lower = more deterministic (0.0-2.0)
max_tokens=4000,      # Maximum response length
stream=True           # Real-time streaming with live markdown rendering
```

## How It Works

### Session Management

1. **Auto-Save**: After each assistant response, the conversation is saved to `~/.zorac_session.json`
2. **Auto-Load**: When you start the client, it automatically loads your previous session
3. **Manual Control**: Use `/save` and `/load` commands for manual session management

### Token Management

The client uses `tiktoken` to accurately count tokens in your conversation:

- Monitors total token count in real-time
- Displays usage after each response
- Prevents exceeding model context limits

### Automatic Summarization

When your conversation exceeds `MAX_INPUT_TOKENS`:

1. The client automatically triggers summarization
2. Older messages are condensed into a summary
3. The most recent 6 messages are preserved intact
4. The summary is injected as a system message
5. This maintains context while staying within limits

Example:
```
� Token limit approaching. Summarizing conversation history...
 Summarized 15 messages. Kept 6 recent messages.
```

## Performance Metrics

After each response, the client displays:

- **Completion tokens**: Number of tokens in the response
- **Response time**: How long generation took
- **Tokens/second**: Generation speed (TPS)
- **Total messages**: Conversation length
- **Token usage**: Current usage vs. limit

```
Stats: 147 tokens in 2.34s (62.82 tok/s) | Total msgs: 12 | Tokens: ~3421/12000
```

## Development

### Testing & Quality

This project includes comprehensive testing, linting, and formatting tools:

```bash
# Install development dependencies
make install-dev

# Run all tests
make test

# Run tests with coverage report
make test-coverage

# Run linter
make lint

# Auto-fix linting issues
make lint-fix

# Format code
make format

# Type checking
make type-check

# Run all checks (lint + type-check + test)
make all-checks

# Complete development setup
make dev-setup
```

**Test Coverage:**
- Unit tests for all core functions
- Integration tests for save/load workflows
- Mock-based testing for API interactions
- 30+ test cases covering edge cases

**Code Quality Tools:**
- **pytest**: Testing framework with coverage reporting
- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checker
- **pre-commit**: Git hooks for automated quality checks

### Project Structure

```
zorac/
├── zorac.py             # Main application
├── .env                 # Configuration (not in git)
├── .env.example         # Configuration template
├── pyproject.toml       # Dependencies and metadata
├── uv.lock              # Dependency lock file
├── README.md            # This file
├── CLAUDE.md            # Development guide for AI assistants
├── SERVER_SETUP.md      # vLLM server setup documentation
├── LICENSE              # MIT License
├── CONTRIBUTING.md      # Contribution guidelines
└── .gitignore           # Git ignore patterns
```

### Dependencies

- **openai** (>=2.8.1): Client library for vLLM's OpenAI-compatible API
- **tiktoken** (>=0.8.0): Token counting for accurate usage tracking
- **python-dotenv** (>=1.0.0): Environment variable management from .env files
- **rich** (>=14.2.0): Rich terminal formatting, markdown rendering, and live updates

### Key Functions

- `count_tokens(messages)`: Counts tokens in conversation using tiktoken
- `save_session(messages)`: Saves conversation to JSON file
- `load_session()`: Loads conversation from JSON file
- `print_header()`: Displays formatted welcome header with Rich panels
- `summarize_old_messages(client, messages)`: Summarizes old messages when token limit is reached
- `main()`: Main interactive loop with streaming responses and Rich UI

### Adding Features

The codebase is structured for easy extension:

1. **New Commands**: Add to the command handling section in `main()`
2. **Custom Token Limits**: Modify constants at the top of `zorac.py`
3. **Different Models**: Update the model name in the API call
4. **UI Customization**: Modify Rich console styling, colors, and panel formatting
5. **Markdown Features**: Extend markdown rendering with custom Rich themes

### AI-Friendly Documentation

This project includes `CLAUDE.md`, a comprehensive development guide designed to help AI coding assistants (like Claude Code, GitHub Copilot, Cursor, etc.) understand the project architecture and conventions.

**What it contains:**
- Detailed project structure and file purposes
- Architecture overview and design decisions
- Configuration documentation
- Development workflows and commands
- Dependencies and their usage

**Why it's useful:**
If you're using AI-powered development tools, this file provides context that helps them give better suggestions, write code that matches project conventions, and understand the codebase structure. Human developers may also find it useful as comprehensive reference documentation.

## Troubleshooting

### Connection Issues

```
Error: Connection refused
```

**Solution**: Ensure your vLLM server is running and accessible at the configured URL.

```bash
# Test server connectivity
curl http://localhost:8000/v1/models
```

### Token Errors

```
Error: context_length_exceeded
```

**Solution**: The auto-summarization should prevent this, but if it occurs:
1. Use `/clear` to start fresh
2. Reduce `MAX_INPUT_TOKENS` in the code
3. Increase `KEEP_RECENT_MESSAGES` to preserve more context

### Session File Issues

```
Error loading session: [Errno 2] No such file or directory
```

**Solution**: This is normal on first run. A new session file will be created automatically.

To reset your session:
```bash
rm ~/.zorac_session.json
```

## Requirements

- Python 3.13+
- vLLM server running with OpenAI-compatible API
- Network access to the vLLM server

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## FAQ

### Can I run this without a GPU?

No, this requires an NVIDIA GPU with at least 10GB VRAM. CPU-only inference is too slow for interactive chat (would take minutes per response).

### How does this compare to running Ollama?

Zorac uses vLLM for faster inference (60+ tok/s vs Ollama's 20-30 tok/s on the same hardware) and supports more advanced features like tool calling for agentic workflows. Ollama is easier to set up but slower for production use.

### Do I need to be online?

Only for the initial model download (~14GB for Mistral-24B-AWQ). After that, everything runs completely offline on your local machine.

### Is this legal? Can I use this commercially?

Yes! Mistral-Small is Apache 2.0 licensed, which allows free commercial use. vLLM is also open source (Apache 2.0). No restrictions.

### What about AMD GPUs or Mac M-series chips?

This guide is specifically for NVIDIA GPUs using CUDA. For AMD GPUs, you'd need ROCm support (experimental). For Mac M-series, check out MLX or llama.cpp instead.

### How much does it cost to run?

Electricity cost for an RTX 4090 running at ~300W is roughly $0.05-0.10 per hour (depending on your electricity rates). Far cheaper than API costs for heavy usage.

### Can I use multiple GPUs?

Yes! vLLM supports tensor parallelism across multiple GPUs. This lets you run larger models or increase throughput.

### What other models can I run?

Any model with vLLM support: Llama, Qwen, Phi, DeepSeek, etc. Just change the `VLLM_MODEL` in your `.env` file. Check [vLLM's supported models](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Also Useful For

- Testing prompts and AI workflows before committing to API costs
- Running AI assistants on air-gapped or secure networks
- Learning about LLM inference optimization and quantization
- Building offline-first AI applications
- AI development in regions with unreliable internet
- Training teams on LLM usage without cloud dependencies
- Prototyping agentic systems with LangChain/LangGraph

## Tips

- Use `/tokens` regularly to monitor your conversation size
- The auto-summarization kicks in at 12k tokens, but you can manually `/clear` anytime
- Sessions persist across restarts - use `/clear` for a fresh start
- Press `Ctrl+C` to interrupt long responses without losing your session
- Adjust `temperature` for different response styles (0.1 = focused, 1.0 = creative)

## Support

For vLLM server setup and configuration, see the [vLLM documentation](https://docs.vllm.ai/).

For issues with this client, please check:
1. Server connectivity
2. Model name matches your vLLM server
3. Python version compatibility (3.13+)
