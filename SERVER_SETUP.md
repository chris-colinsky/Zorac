# Complete Guide: Self-Host Mistral-24B on RTX 4090

**Run a ChatGPT-class LLM on your own hardware - zero monthly costs, complete privacy**

This guide shows you how to set up a high-performance, **self-hosted LLM inference server** on consumer hardware. Perfect for:
- 🏠 **Homelab enthusiasts** running private AI infrastructure
- 💻 **Software engineers** wanting local AI coding assistants
- 🔒 **Privacy-conscious developers** who want data to stay local
- 🎮 **Anyone with a gaming PC** (RTX 3090/4090/3080)

**Total cost:** $0/month after initial setup. No API fees, unlimited queries.

This repository documents the complete configuration for a vLLM inference server on Ubuntu 24.04 LTS, specifically tuned for NVIDIA RTX 4090 (24GB) serving Mistral-Small-24B for autonomous agentic workflows (LangChain/LangGraph).

## Hardware Specifications
- **OS**: Ubuntu 24.04.3 LTS
- **Motherboard**: Asus WS x299 Sage
- **CPU**: Intel® Core™ i9-10940X × 28
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **RAM**: 256GB (System)
- **Storage**: 2TB SSD system disk
- **RAID** 16TB NVME raid 0 - Highpoint SSD7101A


## 1. Prerequisites & System Prep

### GPU Drivers
Ensure proprietary NVIDIA drivers are installed and loaded.
```bash
nvidia-smi
# Should show Driver Version: 550.x or higher and CUDA 12.x
```
If commands are missing:
```bash
sudo apt install ubuntu-drivers-common && sudo ubuntu-drivers autoinstall
```

### Python Environment
Ubuntu 24.04 ships with Python 3.12. Do not attempt to use Python 3.11 or older via PPAs, as this causes conflict with system headers.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using Homebrew (macOS/Linux)
brew install uv
```

## 2. Installation
Use a dedicated virtual environment to isolate vLLM from the system.

```bash
# Create directory
mkdir -p ~/vllm-serve
cd ~/vllm-serve

# Create and activate venv
# Note: uv creates the virtual environment in .venv by default
uv venv

# Install vLLM (This pulls torch and CUDA dependencies automatically)
uv add vllm
```

## 3. Model Selection: The Perfect Fit for RTX 4090

We are using `Mistral-Small-24B-Instruct-2501`. This model represents the "sweet spot" for a single RTX 4090: it is the most intelligent model that fits comfortably within the 24GB VRAM limit while maintaining high-speed generation.

- **Unquantized (FP16)**: Requires ~48GB VRAM. This would require dual GPUs or stepping down to a less capable 7B model.
- **GGUF**: While efficient for RAM, it is not optimized for vLLM's tensor parallelism; results in CPU-like speeds (~6 t/s).
- **AWQ (4-bit)**: Compresses the model to ~14GB VRAM, leaving ~10GB for the KV Cache (Context Window). This allows us to run a 24B model at 60-65 tokens/sec, a feat previously impossible on single consumer cards.

**Target Model**: `stelterlab/Mistral-Small-24B-Instruct-2501-AWQ`

## 4. The "Goldilocks" Configuration
Getting a 24B model to run efficiently on a 24GB card requires specific tuning.

### The Launch Command
```bash
vllm serve "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ" \
  --quantization awq_marlin \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 32 \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

### Explanation of Flags (Tuning Opportunities)
- `--quantization awq_marlin`: The standard awq kernel is slow on 40-series cards. `awq_marlin` boosts speed from ~6 t/s to ~60-65 t/s.
- `--max-model-len 16384`: Limits context to 16k tokens. 32k is possible but risks OOM (Out Of Memory) crashes on a 24GB card when the KV cache fills up.
- `--gpu-memory-utilization 0.85`: Leaves ~15% VRAM for the Desktop Environment and overhead. Setting this to 0.95 will crash during initialization.
- `--max-num-seqs 32`: The vLLM V1 engine attempts to pre-allocate memory for 256 concurrent users, causing an OOM crash on startup. Reducing this to 32 frees up enough VRAM to load the model.
- `--enable-auto-tool-choice` & `--tool-call-parser mistral`: Required for LangChain/LangGraph compatibility. Without these, the API returns a 400 Error when the agent tries to use tools.

## 5. Persistence (Systemd Service)
To ensure the server starts on boot and restarts if it crashes.

Create the file:
```bash
sudo nano /etc/systemd/system/vllm.service
```

Paste the following:
```ini
[Unit]
Description=vLLM Inference Server
After=network.target

[Service]
# UPDATE 'User' and 'WorkingDirectory' to match your actual user
User=commander
WorkingDirectory=/home/zorac/Sandbox/vllm-serve
Environment="PATH=/home/zorac/Sandbox/vllm-serve/.venv/bin:/usr/bin"

# The Start Command
ExecStart=/home/zorac/Sandbox/vllm-serve/.venv/bin/vllm serve \
    "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ" \
    --quantization awq_marlin \
    --dtype auto \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral

# Restart behavior
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
```

## 6. Monitoring & Troubleshooting

### Check Server Status
```bash
# View live logs
sudo journalctl -u vllm -f
```

### Check Performance
If speed drops below 40 t/s, the GPU might be stuck in a low-power P-State.

```bash
# Monitor usage
watch -n 1 nvidia-smi
```

If Pwr is low (<100W) during generation, force high-performance mode:
```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 2100,2900
```

### Common Warnings
- `The tokenizer ... with an incorrect regex pattern`: Safe to ignore. It is a known HuggingFace config issue with Mistral 3 but does not affect generation quality for standard tasks.

## 7. Client Usage
For instructions on how to use the interactive chat client, please refer to [README.md](README.md).

