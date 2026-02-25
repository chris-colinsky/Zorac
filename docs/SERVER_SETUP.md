# Complete Guide: Self-Host Mistral-24B on RTX 4090

## Run a ChatGPT-class LLM on your own hardware - zero monthly costs, complete privacy

This guide shows you how to set up a high-performance, **self-hosted LLM inference server** on consumer hardware using my own AI workstation as an example.

Building your own is perfect for:

- **Homelab enthusiasts** running private AI infrastructure
- **AI engineers** wanting local AI coding assistants, or building agents
- **Privacy-conscious developers** who want data to stay local
- **Anyone with a gaming PC** (RTX 3090/4090/5090)

**Total cost:** $0/month after initial setup. No API fees, unlimited queries.

This repository documents the complete configuration for a vLLM inference server on Ubuntu 24.04 LTS, specifically tuned for NVIDIA RTX 4090 (24GB) serving Mistral-Small-24B for autonomous agentic workflows (LangChain/LangGraph) as well as how to use a dual GPU setup for fine-tuning.

## Hardware Specifications

The system utilizes a split-role GPU strategy to manage mixed architectures (Ampere + Ada Lovelace) and power constraints.

- **OS**: Ubuntu 24.04.3 LTS
- **Motherboard**: Asus WS x299 Sage
- **CPU**: Intel® Core™ i9-10940X × 28 (14 Cores / 28 Threads, AVX-512)
- **RAM**: 256GB
- **System Storage**: 2TB SSD
- **RAID** 16TB NVMe RAID 0 (Highpoint SSD7101A)
- **PSU:** Corsair HX1500i (1500W Platinum)

### GPU Roles

| ID (PCI) | Model | VRAM | Architecture | Role | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPU 0** | **RTX 3090 Ti** | 24GB | Ampere | **Display + Training Pool** | Drives the monitor. High transient power spikes (~650W). Idle during inference. |
| **GPU 1** | **RTX 4090** | 24GB | Ada Lovelace | **Headless Compute Accelerator** | Dedicated to vLLM. Zero display overhead — all 24GB available for inference. |

### Why Two GPUs Matter (Even for Inference)

Having a second GPU isn't just for training — it unlocks significantly better inference performance by offloading display duties from the inference card.

**The problem with a single GPU:** Ubuntu's desktop environment, web browsers, window compositing, and display output consume between 500MB and 1.5GB of VRAM depending on resolution and number of monitors. This forces conservative memory settings (`--gpu-memory-utilization 0.85`) and limits the context window to 16k tokens.

**The dual-GPU solution:** By plugging the monitor into the RTX 3090 Ti (GPU 0) and running vLLM headless on the RTX 4090 (GPU 1), the inference card has zero display overhead. This unlocks:

| Setting | Single GPU (with display) | Dual GPU (headless inference) | Gain |
| :--- | :--- | :--- | :--- |
| `--gpu-memory-utilization` | `0.85` (~20.4 GB) | `0.92` (~22.1 GB) | **+1.7 GB usable VRAM** |
| `--max-model-len` | `16384` (16k) | `32768` (32k) | **2x context window** |
| Display VRAM overhead | ~1-1.5 GB | 0 GB | **Eliminated** |

The math works because of `--kv-cache-dtype fp8`: doubling the context window from 16k to 32k only costs ~1.8GB of additional VRAM in FP8 mode. The ~2.6GB freed by going headless and increasing utilization more than covers this.

> **Single GPU users:** If you only have one GPU driving both display and inference, use `--gpu-memory-utilization 0.85` and `--max-model-len 16384`. See the [Server Setup Guide](site/guides/server-setup.md) for single-GPU configuration details.

> **Ubuntu Server users (single GPU):** If you're running Ubuntu Server or any headless Linux distribution (no desktop environment, no display manager), your single GPU has zero display overhead — even without a second GPU. This means you can use the headless settings (`--gpu-memory-utilization 0.92` and `--max-model-len 32768`) to max out Mistral-Small-24B's full native 32k context window on a single card.

> **Critical Power Warning:**
> The combined peak transient load of a 3090 Ti + 4090 can exceed 1600W, risking a PSU OCP trip.
> **Strict power limits must be applied before engaging both cards simultaneously (Training Mode).**

---

## 1. Prerequisites & System Prep

### BIOS Settings (ASUS WS X299 Sage)

Required for addressing 48GB total VRAM and maximizing PCIe throughput.

- **Above 4G Decoding:** `Enabled`
- **Re-Size BAR Support:** `Enabled`
- **Launch CSM:** `Disabled`

### Prerequisites

1. **NVIDIA Drivers:** Version 550.x+
2. **Python:** 3.12 (System Default on Ubuntu 24.04)
3. **uv:** Fast Python package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

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

## 2. Inference Service (vLLM)

vLLM is configured as a systemd service, pinned strictly to **GPU 1 (RTX 4090)** to utilize the `compressed-tensors` quantization format for maximum throughput (~60-65 t/s).

### Installation

```bash
mkdir -p ~/vllm-serve && cd ~/vllm-serve
python3 -m venv venv
source venv/bin/activate
pip install vllm mistral-common
```

**Note:** mistral-common is required for proper Tool Calling support with Mistral models.

### Configuration (Systemd)

- File: /etc/systemd/system/vllm.service

### Key Configuration Details

Getting a 24B model to run efficiently on a 24GB card requires specific tuning.

- `CUDA_DEVICE_ORDER=PCI_BUS_ID`: Critical. Fixes vLLM crash when using UUIDs.
- `CUDA_VISIBLE_DEVICES=1`: Pins the process to the RTX 4090.
- `--quantization compressed-tensors`: Native vLLM format produced by llmcompressor — no extra packages needed at serve time.
- `--kv-cache-dtype fp8`: Stores the KV cache in 8-bit floating point, roughly halving KV cache memory and freeing VRAM for longer contexts.
- `--max-num-seqs 32`: Prevents OOM on startup by reducing concurrency buffer.
- `--tokenizer-mode mistral`: Enables native Tool Parsing capabilities (do not use custom chat template with this mode).
- `PYTHONUNBUFFERED=1`: Ensures real-time logging.

```toml
[Unit]
Description=vLLM Inference Server
After=network.target

[Service]
# UPDATE 'User' and 'WorkingDirectory' to match your actual user
User=<user>
WorkingDirectory=/home/<user>/Sandbox/vllm-serve

# --- GPU PINNING (Force 4090) ---
Environment="CUDA_DEVICE_ORDER=PCI_BUS_ID"
Environment="CUDA_VISIBLE_DEVICES=1"

# --- LOGGING CONFIG ---
Environment="PYTHONUNBUFFERED=1"
Environment="VLLM_LOGGING_LEVEL=INFO"

ExecStart=/home/commander/Sandbox/vllm-serve/venv/bin/vllm serve \
    "dark-side-of-the-code/Mistral-Small-24B-Instruct-2501-AWQ" \
    --tokenizer "dark-side-of-the-code/Mistral-Small-24B-Instruct-2501-AWQ" \
    --tokenizer-mode mistral \
    --quantization compressed-tensors \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --enable-log-requests \
    --enable-log-outputs \
    --disable-log-stats \
    --kv-cache-dtype fp8

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Explanation of Flags (Tuning Opportunities)

- `--quantization compressed-tensors`: The native vLLM quantization format produced by llmcompressor. vLLM automatically selects the optimal kernel (Marlin on Ada Lovelace GPUs) based on the model's quantization config — no manual kernel selection needed.
- `--kv-cache-dtype fp8`: Stores the key-value cache in FP8 instead of FP16, roughly halving KV cache memory usage. This is what makes a 32k context window possible on a 24GB card — without FP8, the KV cache alone would consume ~16GB at 32k tokens.
- `--max-model-len 32768`: The model's full native 32k context window. This is possible because the 4090 runs headless (no display overhead) and the FP8 KV cache keeps memory usage manageable. **Single GPU with display:** reduce to `16384`.
- `--gpu-memory-utilization 0.92`: With the 4090 running headless (monitor plugged into the 3090 Ti), there is no display VRAM overhead and we can safely use 92% (~22.1GB). **Single GPU with display:** reduce to `0.85` to leave room for the desktop environment.
- `--max-num-seqs 32`: The vLLM V1 engine attempts to pre-allocate memory for 256 concurrent users, causing an OOM crash on startup. Reducing this to 32 frees up enough VRAM to load the model.
- `--enable-auto-tool-choice` & `--tool-call-parser mistral`: Required for LangChain/LangGraph compatibility. Without these, the API returns a 400 Error when the agent tries to use tools.

### Management

```bash
# Apply changes
sudo systemctl daemon-reload
# Start/Restart
sudo systemctl restart vllm
# View Logs (Real-time)
sudo journalctl -u vllm -f -o cat
```

### Model Selection: The Perfect Fit for RTX 4090

We are using `Mistral-Small-24B-Instruct-2501`. This model represents the "sweet spot" for a single RTX 4090: it is the most intelligent model that fits comfortably within the 24GB VRAM limit while maintaining high-speed generation.

- **Unquantized (FP16)**: Requires ~48GB VRAM. This would require dual GPUs or stepping down to a less capable 7B model.
- **GGUF**: While efficient for RAM, it is not optimized for vLLM's tensor parallelism; results in CPU-like speeds (~6 t/s).
- **AWQ (4-bit)**: Compresses the model to ~14GB VRAM, leaving ~10GB for the KV Cache (Context Window). Quantized with llmcompressor into `compressed-tensors` format, which vLLM loads natively and serves with optimized Marlin kernels on RTX 30/40-series GPUs. This allows us to run a 24B model at 60-65 tokens/sec, a feat previously impossible on single consumer cards.

**Target Model**: `dark-side-of-the-code/Mistral-Small-24B-Instruct-2501-AWQ`

## 3. Fine-Tuning Environment (Multi-GPU)

Training uses both cards to pool VRAM (48GB Total). This requires a separate environment and strict safety protocols.

### Power Safety Protocol (Mandatory)

Before starting any training run, you must cap the GPUs to prevent transient spikes from tripping the PSU.

```bash
# Cap both cards at 350W (Total GPU Load ~700W)
sudo nvidia-smi -pl 350
```

### Installation (via uv)

For detailed multi-GPU training setup, environment configuration, and test scripts, see [TEST_TRAINING.md](TEST_TRAINING.md).

### Running a Job

1. Stop Inference: sudo systemctl stop vllm (Frees up the 4090).
2. Cap Power: sudo nvidia-smi -pl 350.
3. Launch: accelerate launch train_script.py.

## 4. Monitoring & Troubleshooting

### GPU Monitoring (btop)

The standard Ubuntu snap for btop does not support NVIDIA GPUs. A custom binary was compiled from source (v1.3.2) to enable this.

- Run: `btop`
- Toggle GPU View: Press `5` on the keyboard.

### Web Dashboard (Cockpit)

- URL: `https://<SERVER_IP>:9090`
- Usage: Used for viewing system logs (`journalctl`), storage health, and terminal access without SSH.

## 5. Client Usage

For instructions on how to use the interactive chat client, please refer to [README.md](https://github.com/chris-colinsky/Zorac/blob/main/README.md).
