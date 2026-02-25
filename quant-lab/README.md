# Quantization

Single-script tool for quantizing large language models to AWQ format for use with vLLM.

**Current target:** `mistralai/Mistral-Small-24B-Instruct-2501` → 4-bit W4A16 GEMM (NVIDIA Marlin/vLLM compatible)

## Requirements

- Python 3.11.14 (managed via `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA RTX 4090 (24GB VRAM) + **128GB system RAM recommended** for quantization (the full bf16 model weighs ~48GB; with calibration overhead and OS, 64GB systems will likely swap — tested on a 256GB system)

## Usage

```bash
# Install dependencies
uv sync

# Run quantization
uv run quantize_mistral.py
```

The script will:
1. Download the base model from HuggingFace Hub (on the first run)
2. Load 256 calibration samples from `HuggingFaceH4/ultrachat_200k`
3. Apply AWQ quantization (W4A16_ASYM, group_size=128) layer-by-layer on GPU
4. Save the quantized model to `/path-to-output/models/Mistral-Small-24B-Instruct-2501-AWQ`

## Serving

```bash
vllm serve /path-to-output/models/Mistral-Small-24B-Instruct-2501-AWQ
```

## Configuration

Edit the constants at the top of `quantize_mistral.py`:

| Variable                  | Default                                     | Description                   |
|---------------------------|---------------------------------------------|-------------------------------|
| `model_id`                | `mistralai/Mistral-Small-24B-Instruct-2501` | HuggingFace model to quantize |
| `quant_path`              | `.../Mistral-Small-24B-Instruct-2501-AWQ`   | Output directory              |
| `NUM_CALIBRATION_SAMPLES` | `256`                                       | Number of calibration samples |
| `MAX_SEQUENCE_LENGTH`     | `512`                                       | Max token length per sample   |

## Why a Single RTX 4090 Is All You Need

### How the quantization pipeline works

The script loads the full model into **system RAM** (`device_map=None`) and hands it to llmcompressor's `SequentialPipeline`, which processes one transformer layer at a time. For a 24B parameter model in bf16, this requires ~48GB of system RAM just for the model weights, plus additional overhead for calibration activations. Combined with OS and other processes, **128GB of system RAM is recommended** — a 64GB system will likely hit swap during quantization, significantly slowing the process or causing OOM failures.

1. Pull one transformer layer from CPU into GPU VRAM
2. Run all calibration samples through that single layer
3. Compute AWQ scaling factors for that layer's weights
4. Quantize that layer's weights in-place
5. Move the layer back to CPU
6. Repeat for the next layer

Because only **one layer at a time** ever lives on the GPU, and individual layers of a 24B model are small (a few hundred MB), a single RTX 4090 handles the entire calibration pass comfortably within its 24GB of VRAM. There is no stage where two layers are processed in parallel — AWQ is inherently sequential because each layer's calibration depends on the output activations of the previous layer.

### Why more GPUs wouldn't help

The theoretical upside of spreading the model across multiple GPUs would be eliminating the PCIe round-trip overhead. With sequential onloading, each of the ~32 transformer layers travels across the PCIe bus twice (CPU→GPU, then GPU→CPU). But this transfer time is not the dominant bottleneck — the calibration compute itself is.

Two GPUs can't calibrate two layers simultaneously because layer N+1's input depends on layer N's output. Adding GPUs doesn't reduce the number of sequential steps — it just skips the PCIe transfer, which is a minor optimization.

The single-GPU sequential approach is the correct, safe, and well-tested path for AWQ with llmcompressor.

### Summary

| Phase                                 | GPU active | Notes                                     |
|---------------------------------------|------------|-------------------------------------------|
| Model loading (`device_map=None`)     | None       | Loaded entirely to system RAM (~48GB for 24B bf16) |
| AWQ calibration (sequential pipeline) | RTX 4090   | One layer at a time; 24GB VRAM is plenty  |
| Saving compressed model               | None       | Written from CPU/RAM to disk              |


### Model description

- What it is: an AWQ-quantized version of the base model
- Link to the base model card
- Brief statement of why the quantization was done (efficient inference on consumer/prosumer GPUs with vLLM)

### Quantization details

Readers need enough details to reproduce or audit the quantization:

| Field               | Value                                                           | What it means                                                                                                                                                                                                                                                                                                                                                                            |
|---------------------|-----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Method              | AWQ (Activation-aware Weight Quantization)                      | The quantization algorithm. AWQ analyses activations on a calibration dataset to identify which weights are most important, then applies per-channel scaling to protect them before rounding to low-bit integers. This preserves accuracy better than naive round-to-nearest quantization.                                                                                               |
| Scheme              | W4A16_ASYM                                                      | **W4** = weights stored as 4-bit integers. **A16** = activations remain in 16-bit (fp16/bf16) during inference — only weights are quantized, not the matmul inputs. **ASYM** = asymmetric quantization: the zero-point is not forced to be symmetric around zero, which gives better coverage of skewed weight distributions at the cost of a slightly more complex dequantization step. |
| Group size          | 128                                                             | Weights are divided into groups of 128 consecutive values, and each group gets its own independent scale (and zero-point for ASYM). Smaller groups = more scales to store but better accuracy; 128 is the standard vLLM-compatible value.                                                                                                                                                |
| Format              | `compressed-tensors` (`pack-quantized`)                         | The on-disk format written by llmcompressor. vLLM supports this natively — no separate `autoawq` package is needed.                                                                                                                                                                                                                                                                      |
| Ignored layers      | `lm_head` (kept at full precision)                              | The final linear projection from hidden states to vocabulary logits. It is excluded from quantization because it directly produces the output probability distribution and is disproportionately sensitive to precision loss — small errors here affect every token prediction.                                                                                                          |
| Tool                | [llmcompressor](https://github.com/vllm-project/llm-compressor) | The library that implements the AWQ calibration and weight-packing pipeline. Records the exact software used so that someone reproducing the quantization uses the same algorithm implementation.                                                                                                                                                                                        |
| Calibration dataset | `HuggingFaceH4/ultrachat_200k` (train_sft split)                | The dataset whose text is fed through the model during calibration. AWQ uses real activations (not random noise) to compute weight scales, so the choice of dataset influences which weight patterns are treated as important. A chat/instruction dataset is appropriate here since the model is an instruction-tuned variant.                                                           |
| Calibration samples | 256                                                             | How many examples from the dataset were used. More samples = more stable scale estimates but longer calibration time. 256 is a common default; diminishing returns set in quickly above ~512.                                                                                                                                                                                            |
| Max sequence length | 512 tokens                                                      | Each calibration sample was truncated/padded to this length before being passed through the model. Shorter sequences speed up calibration; using sequences that are too short relative to typical inference length can cause the calibrated scales to be less representative.                                                                                                            |

### Usage example

Show the minimal vLLM serving command and, optionally, a Python snippet:

```bash
vllm serve dark-side-of-the-code/Mistral-Small-24B-Instruct-2501-AWQ  --dtype auto
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")
response = client.chat.completions.create(
    model="dark-side-of-the-code/Mistral-Small-24B-Instruct-2501-AWQ",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Limitations and accuracy

- Quantization introduces a small accuracy degradation compared to the bf16 base model; the model card should acknowledge this even if you haven't benchmarked it
- If you do run evals (e.g., lm-evaluation-harness), include the scores alongside the base model's numbers
- Note that the model inherits all limitations and intended-use restrictions from the base model

### License

Mistral models are released under the [Mistral Research License](https://mistral.ai/licenses/MRL-0.1.md). The quantized weights are a derivative work — verify that your intended use (research vs. commercial) complies with the license before publishing.
