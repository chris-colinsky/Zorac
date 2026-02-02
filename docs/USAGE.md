# Usage Guide

This guide covers how to use Zorac effectively for interactive chat sessions.

## Getting Started

Start Zorac using your preferred installation method:

```bash
# If installed as binary
zorac

# If installed from source
uv run zorac

# If using uv tool install
zorac
```

## Basic Interaction

Simply type your message at the `You:` prompt and press Enter:

```
You: Explain quantum entanglement
Assistant: [Response streams in real-time with markdown formatting...]
Stats: 245 tokens in 3.8s (64.5 tok/s) | Total: 4 msgs | Tokens: 312/12000
```

The assistant will respond, and the conversation is automatically saved.

### Multi-line Input

Zorac supports pasting multi-line text from your clipboard:

- **Press Enter** - Submit your prompt
- **Paste multi-line text** - Works seamlessly from clipboard, newlines are preserved

Example:
```
You: Please explain the following code:
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
[Press Enter to submit]
```

Simply copy any multi-line text and paste it into the prompt. The newlines will be preserved, and you can submit with Enter.

## Getting Help

To see all available commands at any time, use the `/help` command:

```bash
You: /help

Available Commands:
  /help              - Show all available commands
  /quit or /exit     - Save and exit the application
  /clear             - Reset conversation to initial system message
  /save              - Manually save session to disk
  /load              - Reload session from disk
  /tokens            - Display current token usage statistics
  /summarize         - Force conversation summarization
  /summary           - Display the current conversation summary
  /config            - Manage configuration settings
    /config list     - Show current configuration
    /config set      - Set a configuration value
    /config get      - Get a specific configuration value
```

You can also ask the LLM natural language questions about commands:

```bash
You: How do I save my session?
Assistant: You can save your session using the /save command. Sessions are also automatically saved after each assistant response...

You: What commands are available?
Assistant: Here are the available commands:
- /help - Shows all available commands
- /clear - Resets the conversation...
[continues listing commands]
```

## Available Commands

All commands start with `/`:

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands with descriptions |
| `/clear` | Clear conversation history and start fresh |
| `/save` | Manually save the current session |
| `/load` | Reload session from disk |
| `/tokens` | Display current token usage and limits |
| `/summarize` | Force summarization of conversation history |
| `/summary` | Display the current conversation summary (if exists) |
| `/config` | Manage configuration settings (list, set, get) |
| `/quit` or `/exit` | Save session and exit |
| `Ctrl+C` | Interrupt current operation without exiting |

## Session Management

### Auto-Save

After each assistant response, your conversation is automatically saved to `~/.zorac/session.json`.

### Manual Save/Load

```bash
# Manually save current session
You: /save
✓ Session saved to ~/.zorac/session.json

# Reload from disk (discards unsaved changes)
You: /load
✓ Session reloaded (12 messages, ~3421 tokens)
```

### Clear History

```bash
You: /clear
✓ Conversation history cleared and saved!
```

## Token Management

### Check Token Usage

```bash
You: /tokens

📊 Token usage:
   Current: ~3421 tokens
   Limit: 12000 tokens
   Remaining: ~8579 tokens
   Messages: 12
```

### Automatic Summarization

When your conversation exceeds `MAX_INPUT_TOKENS` (default: 12000):

1. Zorac automatically triggers summarization
2. Older messages are condensed into a summary
3. The most recent 6 messages (configurable) are preserved intact
4. The summary is injected as a system message
5. This maintains context while staying within limits

Example output:

```
⏳ Token limit approaching. Summarizing conversation history...
✓ Summarized 15 messages. Kept 6 recent messages.
```

### Manual Summarization

You can force summarization before reaching the limit:

```bash
You: /summarize
⏳ Summarizing conversation...
✓ Session saved with summary
```

### View Current Summary

```bash
You: /summary

📝 Current Conversation Summary:
┌────────────────────────────────────────┐
│ [Summary content displayed here...]    │
└────────────────────────────────────────┘
```

## Configuration Commands

See the [Configuration Guide](CONFIGURATION.md) for full details.

### View Configuration

```bash
You: /config list

Configuration:
  VLLM_BASE_URL:      http://localhost:8000/v1
  VLLM_MODEL:         stelterlab/Mistral-Small-24B-Instruct-2501-AWQ
  MAX_INPUT_TOKENS:   12000
  MAX_OUTPUT_TOKENS:  4000
  KEEP_RECENT_MESSAGES: 6
  TEMPERATURE:        0.1
  STREAM:             true
  Config File:        ~/.zorac/config.json
```

### Update Settings

```bash
# Change server URL
You: /config set VLLM_BASE_URL http://192.168.1.100:8000/v1

# Adjust temperature
You: /config set TEMPERATURE 0.7

# Disable streaming
You: /config set STREAM false
```

### Get Specific Setting

```bash
You: /config get TEMPERATURE
TEMPERATURE = 0.1
```

## Performance Metrics

After each response, Zorac displays performance statistics:

```
Stats: 147 tokens in 2.34s (62.82 tok/s) | Total msgs: 12 | Tokens: ~3421/12000
```

**Metrics explained:**
- **147 tokens** - Number of tokens in the assistant's response
- **2.34s** - Time taken to generate the response
- **62.82 tok/s** - Tokens per second (generation speed)
- **Total msgs: 12** - Total messages in conversation
- **Tokens: ~3421/12000** - Current token usage vs. limit

## Tips & Best Practices

### Monitor Token Usage

Use `/tokens` regularly to monitor your conversation size:

```bash
You: /tokens
```

The auto-summarization kicks in at 12k tokens, but you can manually `/clear` anytime.

### Interrupt Long Responses

Press `Ctrl+C` to interrupt long responses without losing your session:

```bash
You: Write a very long essay...
Assistant: [Starts streaming...]
^C
Interrupted. Type /quit to exit or continue chatting.
```

### Adjust Response Style

Change temperature for different response styles:

- `0.1` = Focused, deterministic (default)
- `0.7` = Balanced creativity
- `1.0` = Very creative

```bash
You: /config set TEMPERATURE 0.7
```

### Persistent Sessions

Sessions persist across restarts. When you restart Zorac, it automatically loads your previous conversation:

```bash
$ zorac
✓ Loaded previous session (12 messages, ~3421 tokens)
```

### Fresh Start

Use `/clear` when you want to start a new conversation:

```bash
You: /clear
✓ Conversation history cleared and saved!
```

## Example Workflow

Here's a typical Zorac session:

```bash
# Start Zorac
$ zorac
✓ Loaded previous session (4 messages, ~892 tokens)

# Check what was discussed
You: /summary

# Continue conversation
You: Can you explain that concept in more detail?
Assistant: [Detailed explanation...]
Stats: 342 tokens in 5.2s (65.8 tok/s) | Total: 6 msgs | Tokens: ~1234/12000

# Check token usage
You: /tokens
Current: ~1234 tokens | Limit: 12000 | Remaining: ~10766

# Adjust settings for more creative responses
You: /config set TEMPERATURE 0.8

# Continue chatting...
You: Write a creative story about that concept
Assistant: [Creative story...]

# Save and exit
You: /quit
✓ Session saved. Goodbye!
```

## Troubleshooting

### Connection Issues

```
Error: Connection refused
```

**Solution:** Ensure your vLLM server is running:

```bash
# Test server connectivity
curl http://localhost:8000/v1/models
```

See [Configuration Guide](CONFIGURATION.md#troubleshooting-configuration) for more help.

### Token Errors

```
Error: context_length_exceeded
```

**Solution:**
1. Use `/clear` to start fresh
2. Reduce `MAX_INPUT_TOKENS`: `/config set MAX_INPUT_TOKENS 8000`
3. Increase messages kept: `/config set KEEP_RECENT_MESSAGES 10`

### Session File Issues

```
Error loading session: [Errno 2] No such file or directory
```

**Solution:** This is normal on first run. A new session file will be created automatically.

To reset your session:

```bash
rm ~/.zorac/session.json
```

## Next Steps

- [Configuration Guide](CONFIGURATION.md) - Customize Zorac settings
- [Development Guide](DEVELOPMENT.md) - Contribute to Zorac
- [Server Setup](SERVER_SETUP.md) - Set up your vLLM inference server
