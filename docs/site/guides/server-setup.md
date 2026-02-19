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

The simplest way to start vLLM:

```bash
vllm serve "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ" \
    --quantization awq_marlin \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --host 0.0.0.0 \
    --port 8000
```

The first run will download the model from Hugging Face (~14GB). Subsequent starts load from the local cache.

### Key Flags Explained

| Flag | Purpose | Why This Value |
|------|---------|----------------|
| `--quantization awq_marlin` | Use Marlin-optimized AWQ kernels | 10x faster than generic AWQ on RTX 30/40 series |
| `--max-model-len 16384` | Maximum context length | 16k tokens balances context size vs VRAM usage |
| `--gpu-memory-utilization 0.85` | VRAM fraction to use | Leaves 15% headroom for OS/display driver |
| `--max-num-seqs 32` | Max concurrent request slots | Prevents OOM from vLLM's memory pre-allocation |
| `--host 0.0.0.0` | Listen on all network interfaces | Required for remote access from other machines |
| `--port 8000` | HTTP port | vLLM's default, matches Zorac's default URL |

!!! warning "Don't Forget `awq_marlin`"
    Using `--quantization awq` (without `_marlin`) falls back to a generic kernel that's ~10x slower. This is the single most impactful configuration flag.

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

ExecStart=/home/your-username/vllm-serve/venv/bin/vllm serve \
    "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ" \
    --quantization awq_marlin \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --host 0.0.0.0 \
    --port 8000

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

The `--gpu-memory-utilization` flag controls what fraction of total VRAM vLLM can use:

| Value | Available VRAM (24GB card) | Use Case |
|-------|---------------------------|----------|
| `0.95` | 22.8 GB | Dedicated server (no desktop) — may crash on init |
| `0.90` | 21.6 GB | Headless server recommended |
| `0.85` | 20.4 GB | Server with desktop environment (Zorac default) |
| `0.80` | 19.2 GB | Conservative, leaves room for other GPU tasks |

!!! warning "Don't Set Too High"
    Setting `--gpu-memory-utilization` above 0.90 on a system with a desktop environment often causes initialization crashes. The display driver needs some VRAM.

### Max Model Length vs VRAM

Longer context windows consume more VRAM for the KV cache. For Mistral-Small-24B on a 24GB card:

| Context Length | KV Cache Size | Fits with 0.85 util? |
|---------------|---------------|---------------------|
| 8,192 | ~4 GB | Comfortable |
| 16,384 | ~8 GB | Good balance (default) |
| 32,768 | ~16 GB | Tight — may OOM under load |

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
- **Memory used:** ~20-22 GB (model + KV cache)
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
2. **`--gpu-memory-utilization` too high** — Try 0.85 or 0.80
3. **`--max-model-len` too large** — Reduce to 8192
4. **Another process using GPU memory** — Check with `nvidia-smi`

**Fix:** Start with conservative values and increase gradually:

```bash
vllm serve "model" \
    --quantization awq_marlin \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.80 \
    --max-num-seqs 8
```

### Slow Inference (Wrong Kernel)

**Symptom:** Generation speed is ~6 tok/s instead of the expected 60+ tok/s.

**Cause:** Using the wrong quantization flag. The generic AWQ kernel is much slower than the Marlin-optimized version.

**Fix:** Ensure you're using `--quantization awq_marlin` (not `--quantization awq`):

```bash
# Wrong (slow)
vllm serve "model" --quantization awq

# Right (fast)
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
