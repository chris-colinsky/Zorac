# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`quant-lab` is a single-script tool for quantizing large language models to AWQ (Activation-aware Weight Quantization) format for use with vLLM. The current target is `mistralai/Mistral-Small-24B-Instruct-2501`, quantized to W4A16_ASYM (4-bit weights, 16-bit activations) compatible with NVIDIA Marlin/vLLM kernels.

## Environment

- Python 3.11.14 (managed via `.python-version`)
- Package manager: **uv** (use `uv` for all dependency and environment management)
- Virtual environment: `.venv/`

## Commands

```bash
# Install dependencies
uv sync

# Run the quantization script
uv run quantize_mistral.py

# Add a new dependency
uv add <package>
```

## Hardware Context

The script is designed for a single RTX 4090 (24GB VRAM) using sequential layer-by-layer calibration (`device_map=None` loads the full model into system RAM). **128GB system RAM recommended** — the bf16 model weighs ~48GB in memory, and 64GB systems will likely swap. Tested on a 256GB system.

## Key Configuration (quantize_mistral.py)

- **`model_id`**: HuggingFace model to quantize (downloads from Hub on first run)
- **`quant_path`**: Output directory for the resulting AWQ model
- **AWQ recipe**: W4A16_ASYM with group_size=128, applied via llmcompressor's `SequentialPipeline` — standard vLLM-compatible values; change with care

## Output

The quantized model is saved to `quant_path` and can be served directly with vLLM pointing to that directory.
