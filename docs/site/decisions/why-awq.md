# ADR: Why AWQ Quantization

An Architecture Decision Record explaining why Zorac defaults to AWQ quantization with Marlin kernels, what alternatives were evaluated, and the performance data behind the decision.

!!! info "Related Reading"
    For a broader comparison of quantization formats, see the [Quantization Concepts](../concepts/quantization.md) page. This ADR focuses specifically on the decision-making process for Zorac's default configuration.

---

## Context

### Running 24B Parameters on 24GB VRAM

Zorac's target model — Mistral-Small-24B-Instruct — has 24 billion parameters. At full FP16 precision, each parameter occupies 2 bytes:

```
24B params x 2 bytes = 48 GB
```

An RTX 4090 has 24GB of VRAM. We need to cut the model's memory footprint by at least half, while also leaving room for the KV cache (the memory used to track conversation context).

Quantization — reducing the precision of model weights from 16 bits to 4 bits — is the standard solution. The question is **which** quantization format to use.

### The Performance Requirement (Interactive Chat)

Zorac is an interactive chat application. Users expect responsive conversation, not batch processing. This sets a minimum performance bar:

- **Time to first token:** < 1 second
- **Generation speed:** > 30 tok/s for comfortable reading (> 60 tok/s preferred)
- **Memory:** Must fit model + 16k token KV cache in 24GB

Any quantization format that doesn't meet these requirements is unsuitable for the primary use case.

---

## Alternatives Evaluated

### FP16 (No Quantization)

**Memory:** ~48 GB — Does not fit on a single RTX 4090.

FP16 was immediately ruled out. To run the full-precision model, you'd need either:
- A GPU with 48+ GB VRAM (A6000, H100 — enterprise pricing)
- Tensor parallelism across 2x 24GB GPUs (complex setup, halves throughput per card)
- A smaller model (7B instead of 24B — significant quality loss)

### GPTQ

**Memory:** ~12 GB at 4-bit — Fits with room for KV cache.

GPTQ is a mature quantization format that works well with vLLM. Testing showed:

| Metric | GPTQ | AWQ + Marlin |
|--------|------|-------------|
| Speed | 45-55 tok/s | 60-65 tok/s |
| Quality (perplexity) | Good | Slightly better |
| File size | ~15 GB | ~14 GB |

GPTQ is a solid choice, but AWQ with Marlin kernels is consistently faster on RTX 40-series hardware. GPTQ remains a good fallback when an AWQ version of a model isn't available.

### GGUF

**Memory:** Variable (Q4 ~14 GB) — Fits, but performance is poor on NVIDIA GPUs.

GGUF is the native format for llama.cpp and Ollama. It's designed for CPU and mixed CPU/GPU inference, not for maximizing throughput on dedicated NVIDIA GPUs.

Testing results:

```
GGUF Q4 on RTX 4090 via vLLM:   ~6 tok/s
AWQ Marlin on RTX 4090 via vLLM: ~60-65 tok/s
```

A 10x speed difference makes GGUF unsuitable for interactive chat on NVIDIA hardware. It's an excellent choice for CPU-based or Apple Silicon deployments, but that's not Zorac's target platform.

### bitsandbytes (NF4)

**Memory:** ~12 GB at 4-bit — Fits well.

bitsandbytes is primarily a training library. It supports NF4 (NormalFloat 4-bit) quantization, which is optimized for QLoRA fine-tuning rather than inference speed.

Testing results:

```
bitsandbytes NF4 on RTX 4090:  ~15-20 tok/s
AWQ Marlin on RTX 4090:        ~60-65 tok/s
```

bitsandbytes is the right choice for training (it's used in Zorac's multi-GPU fine-tuning setup). For inference, it's significantly slower than AWQ.

---

## Decision: AWQ + Marlin

### Benchmark Results

All tests run on RTX 4090 (24GB), Mistral-Small-24B, vLLM, prompt length ~200 tokens, generation length ~500 tokens:

| Format | tok/s | Time to First Token | Total VRAM |
|--------|-------|--------------------| -----------|
| AWQ + Marlin | 60-65 | ~300ms | ~22 GB |
| AWQ (generic kernel) | ~6 | ~300ms | ~22 GB |
| GPTQ | 45-55 | ~350ms | ~23 GB |
| GGUF Q4 | ~6 | ~500ms | ~22 GB |
| bitsandbytes NF4 | 15-20 | ~400ms | ~22 GB |

AWQ with Marlin kernels delivers the highest throughput by a significant margin. The difference between `awq_marlin` and plain `awq` is the most important configuration detail — it's a 10x performance difference from a single flag.

### Quality Preservation

AWQ (Activation-aware Weight Quantization) identifies the most important weights by analyzing activation patterns — how weights interact with typical inputs. These critical weights are protected during quantization, preserving model quality where it matters most.

For Mistral-Small-24B, the AWQ quantized version performs within 1-2% of the FP16 original on standard benchmarks. In conversational use, the difference is imperceptible — users cannot reliably distinguish AWQ responses from full-precision responses.

### vLLM Integration

vLLM has first-class support for AWQ with Marlin kernels:

```bash
vllm serve "model-name" --quantization awq_marlin
```

The integration is mature, well-tested, and actively maintained. Marlin kernels are specifically optimized for NVIDIA Ada Lovelace (RTX 40-series) and Ampere (RTX 30-series) architectures — the exact GPUs that Zorac targets.

### Memory Budget Analysis

With AWQ at 4-bit precision, the memory budget on an RTX 4090 works out cleanly:

```
Model weights (AWQ 4-bit):   ~14 GB
KV cache (16k context):      ~8 GB
CUDA overhead:               ~2 GB
───────────────────────────────────
Total:                       ~24 GB
```

This leaves the KV cache sufficient for a 16,384-token context window — long enough for extended conversations, especially with Zorac's auto-summarization keeping conversations within budget.

---

## Trade-offs

### Quantization Artifacts

No quantization is lossless. AWQ at 4-bit precision introduces small numerical errors that can occasionally manifest as:

- Slightly different word choices compared to FP16
- Minor degradation on tasks requiring precise numerical reasoning
- Rare hallucinations that wouldn't occur in the full-precision model

In practice, these artifacts are minimal and far outweighed by the ability to run the model at all on consumer hardware.

### Model Availability

Not every model on Hugging Face has an AWQ quantized version. While popular models (Mistral, Llama, Qwen) are well-covered, niche or newly released models may only be available in FP16 or GGUF.

Mitigation: GPTQ is a solid fallback with good vLLM support. For models only available in GGUF, consider using Ollama or llama.cpp instead of vLLM.

### Framework Lock-in

AWQ with Marlin kernels is specifically optimized for vLLM on NVIDIA GPUs. This means:

- **Switching to Ollama or llama.cpp** would require a different model format (GGUF)
- **Running on AMD GPUs** would require a different quantization format
- **Running on CPU** is not supported

For Zorac's target platform (NVIDIA RTX on Linux), this isn't a meaningful limitation. But users with different hardware should choose the format that matches their stack.

---

## Outcome

### Performance in Production

AWQ + Marlin has been running in production with Zorac since the project's inception. Real-world performance matches the benchmarks:

- **Average throughput:** 60-65 tok/s on RTX 4090
- **Time to first token:** 200-500ms depending on prompt length
- **Stability:** No crashes or memory issues related to quantization
- **Quality:** Users report no noticeable difference from cloud-hosted models

### User Feedback

The 60+ tok/s generation speed is one of Zorac's most-cited features. At this speed, reading the response is the bottleneck, not generation — which is the goal for any interactive AI tool.

### When to Reconsider

This decision should be revisited if:

- **New quantization formats** achieve better speed/quality ratios (e.g., FP8 on Hopper/Blackwell GPUs)
- **vLLM drops Marlin support** in favor of a different kernel library
- **Zorac targets non-NVIDIA platforms** (Apple Silicon, AMD ROCm)
- **Model architecture changes** make AWQ less effective (unlikely for transformer-based models)

For now, AWQ + Marlin remains the optimal choice for running large models at interactive speeds on consumer NVIDIA hardware.
