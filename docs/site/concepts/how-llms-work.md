# How LLMs Work

Understanding the fundamentals of Large Language Model inference — from tokenization to text generation — so you can make informed decisions when building local AI applications.

---

## Tokenization

### What Tokens Are

LLMs don't read text the way humans do. Before any processing happens, text is split into **tokens** — small chunks that the model treats as individual units. A token might be a whole word, a word fragment, a punctuation mark, or even a single character.

For example, the sentence "Hello, how are you?" might tokenize as:

```
["Hello", ",", " how", " are", " you", "?"]
```

But longer or less common words get split into subword pieces:

```
"tokenization" → ["token", "ization"]
"ChatGPT"      → ["Chat", "G", "PT"]
```

The key insight: **models don't see characters or words — they see token IDs** (integers that map to entries in a fixed vocabulary). The vocabulary is determined during training and cannot change afterward.

### Why Token Choice Matters

Token counts directly affect two things you care about when running a local LLM:

1. **Context window usage** — Every token in your conversation history (system prompt + user messages + assistant responses) counts toward the model's context limit. More tokens = less room for new conversation.
2. **Cost and speed** — Inference time scales roughly linearly with token count. More output tokens = longer wait times. On an RTX 4090 running AWQ-quantized Mistral-Small-24B, you get about 60-65 tokens per second.

This is why Zorac tracks token counts so carefully — it's the fundamental unit of resource management.

### tiktoken and Encoding Families

Zorac uses [tiktoken](https://github.com/openai/tiktoken), OpenAI's fast tokenizer library, to count tokens. The specific tokenization scheme is called an **encoding**, and different model families use different encodings:

| Encoding | Used By | Vocab Size |
|----------|---------|------------|
| `cl100k_base` | GPT-4, GPT-3.5, most modern models | ~100,000 |
| `o200k_base` | GPT-4o | ~200,000 |

Zorac defaults to `cl100k_base`, which is a reasonable approximation for most models including Mistral. You can change this via the `TIKTOKEN_ENCODING` configuration setting if you're running a model with a significantly different tokenizer.

!!! info "Approximate, Not Exact"
    tiktoken gives an **estimate** of token usage. Different models may tokenize the same text slightly differently. But for context management purposes — deciding when to summarize — an approximation within a few percent is more than sufficient.

---

## The Transformer Architecture

### Attention Mechanism (Simplified)

The Transformer architecture (introduced in the 2017 paper "Attention Is All You Need") is the foundation of every modern LLM. Its core innovation is the **attention mechanism**, which allows the model to consider relationships between all tokens in the input simultaneously.

Think of it this way: when you read the sentence "The cat sat on the mat because **it** was tired," you instantly know "it" refers to "the cat." The attention mechanism gives the model this same ability — for every token, it calculates how much to "pay attention" to every other token.

This is computed as a weighted sum:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Where Q (Query), K (Key), and V (Value) are learned projections of the input tokens. In practice, models use **multi-head attention** — multiple attention computations running in parallel, each learning to focus on different types of relationships (syntactic, semantic, positional, etc.).

### How Layers Process Tokens

A Transformer model stacks many identical layers (Mistral-Small-24B has dozens). Each layer applies:

1. **Self-attention** — Each token looks at all other tokens to gather context
2. **Feed-forward network** — A dense neural network processes each token's enriched representation independently
3. **Layer normalization** — Stabilizes the numbers to prevent them from growing too large or small

The output of one layer feeds into the next. Early layers tend to capture local patterns (grammar, syntax), while deeper layers capture more abstract concepts (meaning, reasoning, world knowledge).

### The KV Cache

During text generation, the model processes tokens one at a time. But each new token needs to "attend" to all previous tokens. Without optimization, this would mean recomputing attention over the entire conversation history for every single new token — extremely wasteful.

The **KV cache** (Key-Value cache) solves this by storing the K and V projections from all previous tokens. When generating token N, the model only computes Q for the new token and reuses the cached K and V from tokens 1 through N-1.

This is why the KV cache consumes significant GPU memory, and why it matters for Zorac:

```
Model weights (AWQ 4-bit):  ~14 GB
KV cache (16k context):     ~8 GB
CUDA overhead:              ~2 GB
────────────────────────────────────
Total VRAM:                 ~24 GB  (the full RTX 4090)
```

The `--max-model-len` flag in vLLM controls the maximum context length, which directly determines the maximum KV cache size. Longer contexts = more VRAM for the KV cache = less room for concurrent requests.

---

## Text Generation

### Autoregressive Generation

LLMs generate text **one token at a time**, from left to right. The model takes the full conversation so far (system prompt + messages + everything generated so far) and predicts a probability distribution over the entire vocabulary for the next token.

```
Input:  "The capital of France is"
Output: {"Paris": 0.92, "Lyon": 0.02, "the": 0.01, ...}
```

The model picks a token from this distribution (how it picks depends on the sampling strategy), appends it to the input, and repeats. This process continues until a stop condition is met.

This is why streaming feels responsive even though the model is doing substantial computation — you see each token as it's generated, rather than waiting for the entire response.

### Temperature and Sampling

The **temperature** parameter controls how the model selects from the probability distribution:

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.0 | Always pick the highest-probability token (greedy) | Factual questions, code generation |
| 0.1 | Mostly deterministic with slight variation | Zorac's default — focused but not robotic |
| 0.7 | Balanced creativity and coherence | Creative writing, brainstorming |
| 1.0 | Sample proportionally to probabilities | More diverse outputs |
| 2.0 | Highly random, chaotic outputs | Rarely useful in practice |

Mathematically, temperature scales the logits (raw scores) before applying softmax. Higher temperature flattens the distribution (more randomness), lower temperature sharpens it (more deterministic).

Zorac defaults to `TEMPERATURE=0.1` because for coding assistance and factual conversation, you usually want consistent, reliable answers. You can change this at runtime via `/config set TEMPERATURE 0.7`.

### Stop Conditions

Generation continues until one of these conditions is met:

1. **End-of-sequence token** — The model outputs a special token (e.g., `</s>`) indicating it has finished its response
2. **Max output tokens** — The `MAX_OUTPUT_TOKENS` limit is reached (default: 4,000 in Zorac)
3. **Context window full** — The combined input + generated output fills the model's context window
4. **User cancellation** — In Zorac, pressing Ctrl+C stops generation mid-stream

---

## Practical Implications

### Why Context Length Has Limits

The attention mechanism computes relationships between **every pair of tokens**. For a sequence of length N, this means N^2 attention scores — memory and computation grow quadratically with context length.

For Mistral-Small-24B with a 16,384-token context:

```
16,384 x 16,384 = ~268 million attention score computations per layer
```

This is why vLLM's `--max-model-len` matters so much for VRAM usage, and why Zorac implements auto-summarization — keeping the conversation within the context window is essential for both performance and correctness.

### Why Bigger Models Are Smarter

Model intelligence roughly correlates with parameter count. More parameters means:

- **More capacity** — The model can store more knowledge and more nuanced patterns
- **Better reasoning** — Deeper networks can compose more complex logical chains
- **Fewer hallucinations** — Larger models tend to be more factually reliable

This is why running a 24B model (Mistral-Small) rather than a 7B model (Mistral-7B) makes a meaningful difference for coding assistance and complex conversations — despite both fitting on an RTX 4090 with quantization.

The trade-off is speed: larger models generate tokens more slowly. At 4-bit AWQ quantization on an RTX 4090, Mistral-Small-24B achieves 60-65 tok/s — fast enough for comfortable interactive use.

### The Speed/Quality Trade-off

Every decision in local LLM deployment involves balancing speed against quality:

| Decision | Faster | Higher Quality |
|----------|--------|----------------|
| Model size | Smaller (7B) | Larger (24B, 70B) |
| Quantization | Lower precision (2-bit) | Higher precision (FP16) |
| Context length | Shorter (4k) | Longer (32k) |
| Batch size | More concurrent requests | Fewer, faster individual responses |

Zorac's defaults represent a carefully chosen balance: a 24B model at 4-bit quantization with 16k context gives strong quality at interactive speeds on a single consumer GPU. The [Quantization](quantization.md) and [Inference Servers](inference-servers.md) concept pages explore these trade-offs in depth.
