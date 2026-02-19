# Context Management

How Zorac manages the conversation context window — counting tokens, auto-summarizing older messages, and balancing between conversation memory and available context budget.

---

## The Context Window Problem

### What the Context Window Is

Every LLM has a **context window** — a fixed limit on the total number of tokens it can process at once. This window includes everything: the system prompt, the entire conversation history, and the model's response.

For Mistral-Small-24B as configured in Zorac:

```
Context window:     16,384 tokens (set by --max-model-len)
├── System prompt:    ~200 tokens
├── Chat history:    variable (grows with conversation)
├── Current message:  variable
└── Model response:  up to 4,000 tokens (MAX_OUTPUT_TOKENS)
```

The available space for conversation history is:

```
16,384 - 200 (system) - 4,000 (response budget) = ~12,184 tokens
```

This is why Zorac defaults to `MAX_INPUT_TOKENS=12000` — it reserves enough room for the model to generate a full response.

### Why It Has a Fixed Size

The context window size is determined by the model's architecture and the available GPU memory. The KV cache (which stores the attention state for all tokens in the context) consumes VRAM proportional to the context length:

```
KV cache memory ≈ 2 × num_layers × hidden_size × context_length × bytes_per_element
```

On an RTX 4090 with a 24B AWQ model, the KV cache for 16k tokens uses approximately 8GB of VRAM — a significant chunk of the 24GB available. Doubling the context to 32k would require ~16GB for the KV cache alone, leaving barely enough for the model weights.

### What Happens When You Exceed It

If the conversation history exceeds the context window, one of two things happens:

1. **API error** — vLLM returns an error because the input exceeds `--max-model-len`
2. **Truncation** — Some servers silently drop tokens from the beginning of the input, losing early conversation context without warning

Neither outcome is good. Zorac prevents both by proactively managing the context window through token counting and auto-summarization.

---

## Token Counting in Zorac

### Using tiktoken

Zorac counts tokens using [tiktoken](https://github.com/openai/tiktoken), the same tokenizer library used by OpenAI's APIs. The counting logic lives in `zorac/utils.py`:

```python
def count_tokens(messages, encoding_name=None):
    encoding = tiktoken.get_encoding(resolved_encoding)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Message envelope overhead
        for _, value in message.items():
            if isinstance(value, str):
                num_tokens += len(encoding.encode(value))
    num_tokens += 2  # Reply priming
    return num_tokens
```

The function follows OpenAI's token counting methodology:

- **4 tokens per message** for the envelope: `<im_start>{role}\n{content}<im_end>\n`
- **Encoded content** — each string value is tokenized and counted
- **2 tokens** at the end for reply priming: `<im_start>assistant`

### Encoding Families and Model Matching

Different model families use different tokenizers. Zorac defaults to `cl100k_base` (the encoding used by GPT-4 and most modern models), which provides a reasonable approximation for Mistral models.

If you're running a model with a significantly different tokenizer, you can change the encoding:

```bash
# Via /config command
/config set TIKTOKEN_ENCODING o200k_base

# Via environment variable
TIKTOKEN_ENCODING=o200k_base zorac
```

!!! info "Approximation Is Sufficient"
    The token count doesn't need to be exact — it's used to trigger summarization (a gradual process) rather than to enforce a hard cutoff. Being off by a few percent is perfectly fine.

### Where Token Counts Appear in the UI

Token counts surface in several places in Zorac's interface:

| Location | What It Shows |
|----------|---------------|
| **Stats bar** (after chat) | Response tokens, duration, tok/s, plus total conversation tokens |
| **Stats bar** (session loaded) | Message count and total token usage |
| **`/tokens` command** | Current token usage, limits, and remaining capacity |
| **Auto-summarization warning** | Appears when token count exceeds `MAX_INPUT_TOKENS` |

The stats bar is a `Static` widget pinned to the bottom of the TUI, updated in real-time during streaming and after each interaction.

---

## Auto-Summarization

### When It Triggers

Before every chat message is sent to the LLM, Zorac checks the total token count:

```python
# In handle_chat() — zorac/main.py
current_tokens = count_tokens(self.messages)
if current_tokens > MAX_INPUT_TOKENS:
    await self._summarize_messages()
```

When the conversation exceeds `MAX_INPUT_TOKENS` (default: 12,000), auto-summarization kicks in. This threshold leaves a comfortable buffer below the actual context window (16,384 tokens) to account for the model's response.

You can also trigger summarization manually at any time with the `/summarize` command — useful if you want to compress the conversation proactively.

### How the Summary Is Generated

Summarization works by sending the older messages to the LLM with a dedicated summarization prompt:

```python
summary_response = await client.chat.completions.create(
    model=VLLM_MODEL,
    messages=[
        {"role": "system",
         "content": "You are a helpful assistant that creates concise summaries."},
        {"role": "user",
         "content": f"Please create a concise summary of this conversation "
                    f"history, preserving key facts, decisions, and context:"
                    f"\n\n{conversation_text}"},
    ],
    temperature=0.1,
    stream=False,
)
```

The summarization call:

- Uses a **separate system prompt** (focused on summarization, not being "Zorac")
- Uses **low temperature** (0.1) for consistent, factual summaries
- Runs **without streaming** because the complete summary is needed before proceeding
- Shows a **spinner** in the UI ("Summarizing...") because the API call takes a few seconds

### What Gets Preserved (KEEP_RECENT_MESSAGES)

Not everything gets summarized. The message list is split into three segments:

```
Before summarization:
[system, msg1, msg2, msg3, ..., msgN-5, msgN-4, msgN-3, msgN-2, msgN-1, msgN]
 ^                                ^──── KEEP_RECENT_MESSAGES (6) ────^
 |── kept ──|── old (summarized) ─|──── kept (recent) ──────────────|

After summarization:
[system, summary, msgN-5, msgN-4, msgN-3, msgN-2, msgN-1, msgN]
```

1. **System message** — Always preserved (first message, contains Zorac's identity and instructions)
2. **Old messages** — Everything between the system message and the recent messages gets summarized into a single paragraph
3. **Recent messages** — The last `KEEP_RECENT_MESSAGES` (default: 6) messages are kept verbatim for conversational continuity

The recent messages are preserved because they represent the immediate context of the conversation — what was just discussed. Summarizing them would lose the nuance needed for coherent follow-up responses.

### The Summary Message Format

The summary is stored as a system message with a recognizable prefix:

```python
summary_message = {
    "role": "system",
    "content": f"Previous conversation summary: {summary}",
}
```

This format allows:

- The LLM to understand it's reading a summary of prior conversation (not instructions)
- The `/summary` command to find and display the summary by checking for the `"Previous conversation summary:"` prefix
- Clear distinction from the main system message

If summarization fails (network error, model error), Zorac falls back to keeping only the system message plus recent messages. Losing old context is better than crashing.

---

## Configuration

### MAX_INPUT_TOKENS

**Default:** `12000`

The maximum number of tokens allowed for the system prompt plus conversation history. When this limit is exceeded, auto-summarization triggers.

This should be set lower than your model's actual context window (`--max-model-len`) to leave room for the model's response. A good rule of thumb:

```
MAX_INPUT_TOKENS = max_model_len - MAX_OUTPUT_TOKENS - buffer
```

For Zorac's defaults: `16384 - 4000 - 384 ≈ 12000`

### MAX_OUTPUT_TOKENS

**Default:** `4000`

The maximum number of tokens the model can generate per response. This is passed directly to the API as `max_tokens`:

```python
stream_response = await self.client.chat.completions.create(
    model=self.vllm_model,
    messages=self.messages,
    max_tokens=self.max_output_tokens,  # ← This setting
    stream=True,
)
```

Higher values allow longer responses but consume more of the context budget. For coding tasks where you want complete code blocks, 4000 is a good default. For shorter Q&A, you could reduce this to 2000.

### KEEP_RECENT_MESSAGES

**Default:** `6`

The number of recent messages preserved during auto-summarization. This creates a "sliding window" effect:

- **Higher values** (8-10) — More immediate context preserved, but summarization saves less space
- **Lower values** (2-4) — More aggressive compression, but the model may lose track of recent topics
- **Default (6)** — Preserves the last 3 user-assistant exchanges, which is usually enough for conversational coherence

### Choosing Values for Your Use Case

| Use Case | MAX_INPUT_TOKENS | MAX_OUTPUT_TOKENS | KEEP_RECENT_MESSAGES |
|----------|-----------------|-------------------|---------------------|
| **General chat** | 12000 | 4000 | 6 |
| **Coding assistant** | 12000 | 4000 | 8 |
| **Quick Q&A** | 8000 | 2000 | 4 |
| **Long-form writing** | 10000 | 6000 | 4 |
| **Short context (8k model)** | 4000 | 2000 | 4 |

All three settings can be changed at runtime via the `/config` command:

```
/config set MAX_INPUT_TOKENS 10000
/config set MAX_OUTPUT_TOKENS 6000
/config set KEEP_RECENT_MESSAGES 4
```

Or in your `.env` file for persistent defaults:

```bash
MAX_INPUT_TOKENS=10000
MAX_OUTPUT_TOKENS=6000
KEEP_RECENT_MESSAGES=4
```
