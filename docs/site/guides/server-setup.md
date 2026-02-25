# Server Setup Guide

A step-by-step guide to setting up a vLLM inference server on your own hardware — from driver installation to systemd configuration and performance tuning.

!!! tip "Reference Documentation"
    This guide walks through the setup process step by step with explanations. For a concise reference of the server configuration, see the [Server Setup Reference](../../SERVER_SETUP.md).

---

## Before You Start

### Hardware Requirements

vLLM requires an NVIDIA GPU with sufficient VRAM to hold the quantized model plus the KV cache. Here's what different GPUs can handle:

| GPU | VRAM | Max Model Size (4-bit) | Expected Speed |
|-----|------|----------------------|----------------|
| RTX 3060 | 12GB | ~7B parameters | 30-40 tok/s |
| RTX 3080 | 10GB | ~7B parameters | 35-45 tok/s |
| RTX 3090 | 24GB | ~24B parameters | 45-55 tok/s |
| RTX 4090 | 24GB | ~24B parameters | 60-65 tok/s |
| RTX 5090 | 32GB | ~32B parameters | 70+ tok/s |

Zorac's default model (Mistral-Small-24B-AWQ) requires a 24GB card. If you have less VRAM, use a smaller model like Mistral-7B.

**Minimum system requirements:**

- NVIDIA GPU with 10+ GB VRAM (24GB recommended)
- NVIDIA drivers version 550.x or higher
- 16GB+ system RAM
- ~30GB disk space (for model download and vLLM installation)
- Linux recommended (Ubuntu 22.04 or 24.04 LTS)

### Software Prerequisites

You need three things installed before setting up vLLM:

1. **NVIDIA drivers** — Proprietary drivers (not Nouveau)
2. **Python 3.10+** — Ubuntu 24.04 ships with Python 3.12
3. **uv** (recommended) or pip — For managing the Python environment

### Choosing a Model

Your model choice depends on your VRAM budget:

| VRAM | Recommended Model | Parameters | Quantization |
|------|-------------------|------------|-------------|
| 24GB | Mistral-Small-24B-AWQ | 24B | AWQ 4-bit |
| 16GB | Mistral-7B-AWQ | 7B | AWQ 4-bit |
| 12GB | Mistral-7B-AWQ | 7B | AWQ 4-bit |
| 10GB | Qwen2.5-7B-AWQ | 7B | AWQ 4-bit |

Look for models on [Hugging Face](https://huggingface.co/models) with "AWQ" in the name — these are pre-quantized and ready for vLLM. The model will be downloaded automatically the first time you start the server.

---

## Installation

### NVIDIA Drivers

Verify that NVIDIA drivers are installed and working:

```bash
nvidia-smi
```

You should see output showing your GPU model, driver version (550.x+), and CUDA version (12.x). If the command is not found:

```bash
# Ubuntu
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

After rebooting, run `nvidia-smi` again to confirm.

### Python Environment with uv

We recommend using [uv](https://docs.astral.sh/uv/) for managing the Python environment. It's faster than pip and handles virtual environments cleanly.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew
brew install uv
```

Create a dedicated directory and virtual environment for vLLM:

```bash
mkdir -p ~/vllm-serve && cd ~/vllm-serve
python3 -m venv venv
source venv/bin/activate
```

### Installing vLLM

Install vLLM inside the virtual environment:

```bash
pip install vllm
```

For Mistral models with tool calling support, also install the Mistral tokenizer:

```bash
pip install mistral-common
```

Verify the installation:

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## Configuration

### The vLLM Serve Command

The simplest way to start vLLM on a **single GPU** (with display attached):

```bash
vllm serve "dark-side-of-the-code/Mistral-Small-24B-Instruct-2501-AWQ" \
    --quantization compressed-tensors \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --kv-cache-dtype fp8 \
    --host 0.0.0.0 \
    --port 8000
```

If your inference GPU is **headless** (monitor plugged into a different GPU), you can push higher:

```bash
vllm serve "dark-side-of-the-code/Mistral-Small-24B-Instruct-2501-AWQ" \
    --quantization compressed-tensors \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 \
    --kv-cache-dtype fp8 \
    --host 0.0.0.0 \
    --port 8000
```

The first run will download the model from Hugging Face (~14GB). Subsequent starts load from the local cache.

### Key Flags Explained

| Flag | Purpose | Single GPU (with display) | Headless GPU (no display) |
|------|---------|---------------------------|---------------------------|
| `--quantization compressed-tensors` | Native vLLM format (llmcompressor) | Same | Same |
| `--kv-cache-dtype fp8` | Halve KV cache memory usage | Same | Same |
| `--max-model-len` | Maximum context length | `16384` (16k) | `32768` (32k — full native) |
| `--gpu-memory-utilization` | VRAM fraction to use | `0.85` (leaves room for display) | `0.92` (no display overhead) |
| `--max-num-seqs 32` | Max concurrent request slots | Same | Same |
| `--host 0.0.0.0` | Listen on all interfaces | Same | Same |
| `--port 8000` | HTTP port | Same | Same |

!!! info "Why the difference?"
    Ubuntu's desktop environment, web browsers, window compositing, and display output consume 500MB–1.5GB of VRAM on the GPU driving the monitor. If your inference GPU also drives the display, you must reserve that headroom (`0.85`). If you have a second GPU handling display duties, the inference GPU has zero display overhead — you can safely use `0.92` and unlock the full 32k context window.

!!! tip "Ubuntu Server: Max out your single GPU"
    Running Ubuntu Server or any headless Linux distribution (no desktop environment, no display manager)? Your single GPU has zero display overhead — no second GPU needed. Use the headless settings (`0.92` / `32768`) to unlock Mistral-Small-24B's full native 32k context window on a single card. This is the ideal setup for a dedicated inference server.

!!! tip "Why `compressed-tensors` instead of `awq_marlin`?"
    Zorac's model is quantized with [llmcompressor](https://github.com/vllm-project/llm-compressor) (the vLLM project's official quantization tool) rather than autoawq. The `compressed-tensors` output format is native to vLLM — no extra packages needed at serve time, and vLLM automatically selects the optimal kernel (Marlin on RTX 30/40-series) from the model's metadata. If you're using a model quantized with autoawq instead, use `--quantization awq_marlin`. See the [Why AWQ](../decisions/why-awq.md) decision record for the full rationale.

### Creating a Systemd Service

For a server that starts automatically on boot and restarts on crashes, create a systemd service:

```bash
sudo nano /etc/systemd/system/vllm.service
```

```toml
[Unit]
Description=vLLM Inference Server
After=network.target

[Service]
User=your-username
WorkingDirectory=/home/your-username/vllm-serve
Environment="PATH=/home/your-username/vllm-serve/venv/bin:/usr/bin"
Environment="CUDA_DEVICE_ORDER=PCI_BUS_ID"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PYTHONUNBUFFERED=1"

# For headless inference GPU (monitor on a different GPU):
ExecStart=/home/your-username/vllm-serve/venv/bin/vllm serve \
    "dark-side-of-the-code/Mistral-Small-24B-Instruct-2501-AWQ" \
    --quantization compressed-tensors \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 \
    --kv-cache-dtype fp8 \
    --host 0.0.0.0 \
    --port 8000
# Single GPU with display: use --max-model-len 16384 --gpu-memory-utilization 0.85

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Replace `your-username` with your actual username, and adjust `CUDA_VISIBLE_DEVICES` if you have multiple GPUs (use the GPU ID shown by `nvidia-smi`).

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
```

### Environment Variables

Key environment variables in the systemd service:

| Variable | Purpose |
|----------|---------|
| `CUDA_DEVICE_ORDER=PCI_BUS_ID` | Ensures GPU numbering matches `nvidia-smi` output. Critical for multi-GPU systems. |
| `CUDA_VISIBLE_DEVICES=0` | Pins vLLM to a specific GPU. Change the number for multi-GPU setups. |
| `PYTHONUNBUFFERED=1` | Ensures log output appears immediately in `journalctl`. |

---

## Performance Tuning

### GPU Memory Utilization

The `--gpu-memory-utilization` flag controls what fraction of total VRAM vLLM can use. The right value depends on whether your inference GPU also drives a monitor:

**Headless inference GPU** (monitor on a different GPU):

| Value | Available VRAM (24GB card) | Use Case |
|-------|---------------------------|----------|
| `0.95` | 22.8 GB | Maximum — leaves minimal CUDA context headroom |
| `0.92` | 22.1 GB | Zorac default for headless — safe with room for 32k context |
| `0.90` | 21.6 GB | Conservative headless |

**Single GPU with display:**

| Value | Available VRAM (24GB card) | Use Case |
|-------|---------------------------|----------|
| `0.85` | 20.4 GB | Recommended — leaves room for desktop environment |
| `0.80` | 19.2 GB | Conservative, leaves room for other GPU tasks |

!!! warning "Display overhead matters"
    Ubuntu's desktop environment, web browsers, window compositing, and display output consume 500MB–1.5GB of VRAM on the GPU driving the monitor. Setting `--gpu-memory-utilization` above 0.90 on a GPU that also drives a display often causes initialization crashes. If your inference GPU is headless — either because you have a second GPU driving the monitor, or because you're running Ubuntu Server with no desktop environment — this limitation doesn't apply.

### Max Model Length vs VRAM

Longer context windows consume more VRAM for the KV cache. With `--kv-cache-dtype fp8`, the KV cache uses roughly half the memory compared to FP16, making larger context windows practical:

| Context Length | KV Cache Size (FP8) | Headless (0.92 util) | With display (0.85 util) |
|---------------|---------------------|----------------------|--------------------------|
| 8,192 | ~0.9 GB | Comfortable | Comfortable |
| 16,384 | ~1.8 GB | Comfortable | Good balance (recommended) |
| 32,768 | ~3.6 GB | Good balance (recommended) | Tight — may OOM under load |

Mistral-Small-24B natively supports a 32k context window. With a headless inference GPU and FP8 KV cache, you can use the full native 32k context — whether that's a dedicated GPU in a multi-GPU system or a single GPU on Ubuntu Server with no desktop environment. With a single GPU also running a display, stick to 16k.

If you're hitting OOM errors during long conversations, reduce `--max-model-len` to `8192`.

### Concurrent Request Limits

vLLM's V1 engine pre-allocates memory for concurrent request slots at startup. The default (256) assumes a production server with substantial VRAM headroom. On a 24GB consumer card, this can cause an OOM crash before the server even starts.

Setting `--max-num-seqs 32` reduces this pre-allocation to a reasonable level for personal use. For a single-user Zorac setup, even `--max-num-seqs 8` would be sufficient.

### Monitoring with nvidia-smi

Monitor GPU usage during operation:

```bash
# Real-time monitoring (updates every second)
watch -n1 nvidia-smi

# Show just memory usage
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv -l 1
```

Healthy metrics during Zorac chat:
- **Memory used:** ~20-23 GB (model + KV cache, varies with context length and GPU config)
- **GPU utilization:** Spikes to 80-100% during generation, drops to 0% between messages
- **Temperature:** 50-70C under load (check your cooling)

---

## Connecting Zorac

### Setting VLLM_BASE_URL

If vLLM is running on the same machine as Zorac, the default URL works:

```
VLLM_BASE_URL=http://localhost:8000/v1
```

For a remote server on your local network:

```
VLLM_BASE_URL=http://192.168.1.100:8000/v1
```

Set this in Zorac's `.env` file, via the `/config` command, or as an environment variable:

```bash
# Environment variable
VLLM_BASE_URL=http://192.168.1.100:8000/v1 zorac

# Or via /config in Zorac
/config set VLLM_BASE_URL http://192.168.1.100:8000/v1
```

### Testing the Connection

Before starting Zorac, verify the server is responding:

```bash
curl http://localhost:8000/v1/models
```

You should see a JSON response listing the loaded model. When Zorac starts, it also runs a connection check automatically and reports the result.

### Remote Server Access

If the vLLM server is on a different machine (e.g., a homelab server), ensure:

1. **vLLM is listening on all interfaces** — Use `--host 0.0.0.0`, not `--host 127.0.0.1`
2. **Firewall allows port 8000** — `sudo ufw allow 8000` on Ubuntu
3. **Network is reachable** — The client machine can reach the server's IP

!!! info "No Authentication Required"
    vLLM doesn't require API keys by default. Zorac uses `VLLM_API_KEY=EMPTY` as a placeholder. If you're exposing the server to a broader network, consider adding a reverse proxy with authentication.

---

## Troubleshooting

### OOM Crashes on Startup

**Symptom:** vLLM crashes immediately with "CUDA out of memory" during initialization.

**Common causes:**

1. **`--max-num-seqs` too high** — Reduce to 32 or lower
2. **`--gpu-memory-utilization` too high** — If your GPU drives a monitor, use 0.85 (not 0.92+)
3. **`--max-model-len` too large** — Reduce to 16384 (or 8192 if still crashing)
4. **Another process using GPU memory** — Check with `nvidia-smi`

**Fix:** Start with conservative values and increase gradually:

```bash
vllm serve "model" \
    --quantization compressed-tensors \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.80 \
    --max-num-seqs 8
```

### Slow Inference (Wrong Kernel)

**Symptom:** Generation speed is ~6 tok/s instead of the expected 60+ tok/s.

**Cause:** Using the wrong quantization format. Models quantized with llmcompressor should use `compressed-tensors`, while models from other tools may use `awq` or `awq_marlin`.

**Fix:** Ensure you're using the correct `--quantization` flag for your model's format:

```bash
# For llmcompressor-quantized models (Zorac default)
vllm serve "model" --quantization compressed-tensors

# For autoawq-quantized models
vllm serve "model" --quantization awq_marlin
```

### Connection Refused

**Symptom:** Zorac shows "Connection failed!" or `curl` returns "Connection refused."

**Checklist:**

1. **Is vLLM running?** — `systemctl status vllm` or check for the process
2. **Is it listening on the right port?** — `ss -tlnp | grep 8000`
3. **Is it bound to the right interface?** — `--host 0.0.0.0` for remote access, not `--host 127.0.0.1`
4. **Is the model loaded?** — Check logs with `journalctl -u vllm -f`. Model loading can take 30-60 seconds.
5. **Firewall?** — `sudo ufw status` and ensure port 8000 is allowed
