# Configuration Guide

Zorac provides flexible configuration through environment variables, configuration files, and runtime commands.

## Configuration Priority

Settings are loaded in this order (later sources override earlier ones):

1. **Default Values** - Built-in defaults
2. **Config File** - `~/.zorac/config.json`
3. **Environment Variables** - `.env` file or shell environment

## Server Configuration

### Basic Setup

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Server Connection
VLLM_BASE_URL=http://localhost:8000/v1  # Change to your vLLM server URL
VLLM_API_KEY=EMPTY                       # vLLM doesn't require authentication
VLLM_MODEL=stelterlab/Mistral-Small-24B-Instruct-2501-AWQ

# Token Limits (optional - defaults shown)
# MAX_INPUT_TOKENS=12000
# MAX_OUTPUT_TOKENS=4000
# KEEP_RECENT_MESSAGES=6

# Model Parameters (optional - defaults shown)
# TEMPERATURE=0.1
# STREAM=true

# Token Counting (optional - defaults shown)
# TIKTOKEN_ENCODING=cl100k_base

# Display (optional - defaults shown)
# CODE_THEME=monokai

# Optional: Custom paths
# ZORAC_DIR=/path/to/custom/zorac/directory
# ZORAC_SESSION_FILE=/path/to/custom/session.json
# ZORAC_HISTORY_FILE=/path/to/custom/history
# ZORAC_CONFIG_FILE=/path/to/custom/config.json
```

### Environment Variable Override

You can override `.env` settings from the command line:

```bash
# Temporary override for one session
VLLM_BASE_URL="http://192.168.1.100:8000/v1" zorac

# Or with uv run
VLLM_BASE_URL="http://192.168.1.100:8000/v1" uv run zorac
```

## Configuration Settings

### Server Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server endpoint |
| `VLLM_API_KEY` | `EMPTY` | API key (vLLM doesn't require one) |
| `VLLM_MODEL` | `stelterlab/Mistral-Small-24B-Instruct-2501-AWQ` | Model name |

### Token Limits

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_INPUT_TOKENS` | `12000` | Maximum tokens for input (system + history) |
| `MAX_OUTPUT_TOKENS` | `4000` | Maximum tokens for model responses |
| `KEEP_RECENT_MESSAGES` | `6` | Messages to preserve during auto-summarization |

### Model Parameters

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| `TEMPERATURE` | `0.1` | `0.0-2.0` | Lower = more deterministic, Higher = more creative |
| `STREAM` | `true` | `true/false` | Enable real-time streaming with live markdown rendering |

### Display

| Setting | Default | Description |
|---------|---------|-------------|
| `CODE_THEME` | `monokai` | Pygments theme for syntax-highlighted code blocks |

Common themes: `monokai`, `dracula`, `github-dark`, `one-dark`, `solarized-dark`, `solarized-light`, `nord`, `gruvbox-dark`, `native`. Any Pygments style name is supported.

### Token Counting

| Setting | Default | Description |
|---------|---------|-------------|
| `TIKTOKEN_ENCODING` | `cl100k_base` | Tiktoken encoding used for token counting and context management |

Zorac uses [tiktoken](https://github.com/openai/tiktoken) to count tokens for context window management, auto-summarization triggers, and performance metrics. Different model families use different tokenizers, so accurate token counting requires matching the encoding to your model.

**Available encodings:**

| Encoding | Models | Use When |
|----------|--------|----------|
| `cl100k_base` | GPT-4, GPT-3.5, Mistral, most current models | Default — correct for the bundled Mistral model and most vLLM-served models |
| `o200k_base` | GPT-4o, GPT-4o-mini | You're proxying through an OpenAI-compatible endpoint serving GPT-4o family models |
| `p50k_base` | Codex, text-davinci-002/003 | You're running older code-focused models |
| `r50k_base` (gpt2) | GPT-2, older GPT-3 | You're running legacy models |

Most users should leave this at the default. Change it only if you switch to a model family with a different tokenizer — using the wrong encoding won't break anything, but token counts will be slightly inaccurate, which can cause auto-summarization to trigger too early or too late.

### File Locations

All file locations are configurable via environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `ZORAC_DIR` | `~/.zorac` | Main configuration directory |
| `ZORAC_SESSION_FILE` | `~/.zorac/session.json` | Session storage location |
| `ZORAC_HISTORY_FILE` | `~/.zorac/history` | Command history location |
| `ZORAC_CONFIG_FILE` | `~/.zorac/config.json` | Configuration file location |

## Runtime Configuration

### Using the /config Command

Modify settings without restarting Zorac:

```bash
# View all current settings
You: /config list

# Update server URL
You: /config set VLLM_BASE_URL http://192.168.1.100:8000/v1

# Adjust model parameters
You: /config set TEMPERATURE 0.7
You: /config set MAX_OUTPUT_TOKENS 2000
You: /config set STREAM false

# Change token counting encoding (if using a different model family)
You: /config set TIKTOKEN_ENCODING o200k_base

# Change code syntax highlighting theme
You: /config set CODE_THEME dracula

# Get a specific setting
You: /config get TEMPERATURE
```

### Configuration File

Runtime changes are saved to `~/.zorac/config.json`:

```json
{
  "VLLM_BASE_URL": "http://192.168.1.100:8000/v1",
  "TEMPERATURE": "0.7",
  "MAX_OUTPUT_TOKENS": "2000",
  "STREAM": "false"
}
```

You can also edit this file directly, but changes require restarting Zorac.

## Common Configuration Scenarios

### Remote Server

Connect to a vLLM server on another machine:

```bash
# In .env file
VLLM_BASE_URL=http://192.168.1.100:8000/v1
```

### Creative Writing Mode

Adjust temperature for more creative responses:

```bash
# In .env file
TEMPERATURE=0.9
MAX_OUTPUT_TOKENS=8000
```

### Faster Responses (Non-Streaming)

Disable streaming for complete responses:

```bash
# In .env file
STREAM=false
```

### Extended Context

Increase token limits for longer conversations:

```bash
# In .env file
MAX_INPUT_TOKENS=24000
KEEP_RECENT_MESSAGES=10
```

### Custom Data Directory

Store configuration in a custom location:

```bash
# In .env file
ZORAC_DIR=/path/to/my/zorac/data
```

## Troubleshooting Configuration

### Server Connection Issues

```
Error: Connection refused
```

**Check:**
1. Verify server is running: `curl http://localhost:8000/v1/models`
2. Check `VLLM_BASE_URL` is correct
3. Ensure firewall allows connections

### Configuration Not Applied

**Remember the priority order:**
- Environment variables override config file
- Config file overrides defaults
- `/config set` updates config file only

### Reset Configuration

```bash
# Remove config file to reset to defaults
rm ~/.zorac/config.json

# Or remove entire directory
rm -rf ~/.zorac
```

## Next Steps

- [Usage Guide](USAGE.md) - Learn how to use Zorac
- [Commands Reference](COMMANDS.md) - All available commands
- [Server Setup](SERVER_SETUP.md) - Set up your vLLM inference server
