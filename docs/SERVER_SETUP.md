# Complete Guide: Self-Host Mistral-24B on RTX 4090

**Run a ChatGPT-class LLM on your own hardware - zero monthly costs, complete privacy**

This guide shows you how to set up a high-performance, **self-hosted LLM inference server** on consumer hardware using my own AI workstation as an example.

Building your own is perfect for:
- **Homelab enthusiasts** running private AI infrastructure
- **AI engineers** wanting local AI coding assistants, or building agents
- **Privacy-conscious developers** who want data to stay local
- **Anyone with a gaming PC** (RTX 3090/4090/5090)

**Total cost:** $0/month after initial setup. No API fees, unlimited queries.

This repository documents the complete configuration for a vLLM inference server on Ubuntu 24.04 LTS, specifically tuned for NVIDIA RTX 4090 (24GB) serving Mistral-Small-24B for autonomous agentic workflows (LangChain/LangGraph) as well as how to use the dual GPU setup for fine-tuning.

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
| **GPU 0** | **RTX 3090 Ti** | 24GB | Ampere | **Training Pool** | High transient power spikes (~650W). Idle during inference. |
| **GPU 1** | **RTX 4090** | 24GB | Ada Lovelace | **Inference Host** | Dedicated to vLLM. Uses `awq_marlin` kernels. |

> **Critical Power Warning:**
> The combined peak transient load of a 3090 Ti + 4090 can exceed 1600W, risking a PSU OCP trip.
> **Strict power limits must be applied before engaging both cards simultaneously (Training Mode).**

---

## 1. Prerequisites & System Prep

### BIOS Settings (ASUS WS X299 Sage)
Required for addressing 48GB total VRAM and maximizing PCIe throughput.
* **Above 4G Decoding:** `Enabled`
* **Re-Size BAR Support:** `Enabled`
* **Launch CSM:** `Disabled`

### Prerequisites
1.  **NVIDIA Drivers:** Version 550.x+
2.  **Python:** 3.12 (System Default on Ubuntu 24.04)
3.  **uv:** Fast Python package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

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
vLLM is configured as a systemd service, pinned strictly to **GPU 1 (RTX 4090)** to utilize the `awq_marlin` quantization kernel for maximum throughput (~60-65 t/s).

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

### Key Configuration Details:

Getting a 24B model to run efficiently on a 24GB card requires specific tuning.

- `CUDA_DEVICE_ORDER=PCI_BUS_ID`: Critical. Fixes vLLM crash when using UUIDs.
- `CUDA_VISIBLE_DEVICES=1`: Pins the process to the RTX 4090.
- `--quantization awq_marlin`: Optimized 4-bit kernel for Ada Lovelace.
- `--max-num-seqs 32`: Prevents OOM on startup by reducing concurrency buffer.
- `--tokenizer-mode mistral`: Enables native Tool Parsing capabilities (do not use custom chat template with this mode).
- `PYTHONUNBUFFERED=1`: Ensures real-time logging.

```toml
[Unit]
Description=vLLM Inference Server
After=network.target

[Service]
# UPDATE 'User' and 'WorkingDirectory' to match your actual user
User=commander
WorkingDirectory=/home/commander/Sandbox/vllm-serve
Environment="PATH=/home/commander/Sandbox/vllm-serve/venv/bin:/usr/bin"

# --- GPU PINNING (Force 4090) ---
Environment="CUDA_DEVICE_ORDER=PCI_BUS_ID"
Environment="CUDA_VISIBLE_DEVICES=1"

# --- LOGGING CONFIG ---
Environment="PYTHONUNBUFFERED=1"
Environment="VLLM_LOGGING_LEVEL=INFO"

ExecStart=/home/commander/Sandbox/vllm-serve/venv/bin/vllm serve \
    "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ" \
    --tokenizer "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ" \
    --tokenizer-mode mistral \
    --quantization awq_marlin \
    --dtype auto \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --enable-log-requests \
    --enable-log-outputs \
    --disable-log-stats

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Explanation of Flags (Tuning Opportunities)
- `--quantization awq_marlin`: The standard awq kernel is slow on 40-series cards. `awq_marlin` boosts speed from ~6 t/s to ~60-65 t/s.
- `--max-model-len 16384`: Limits context to 16k tokens. 32k is possible but risks OOM (Out Of Memory) crashes on a 24GB card when the KV cache fills up.
- `--gpu-memory-utilization 0.85`: Leaves ~15% VRAM for the Desktop Environment and overhead. Setting this to 0.95 will crash during initialization.
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
- **AWQ (4-bit)**: Compresses the model to ~14GB VRAM, leaving ~10GB for the KV Cache (Context Window). This allows us to run a 24B model at 60-65 tokens/sec, a feat previously impossible on single consumer cards.

**Target Model**: `stelterlab/Mistral-Small-24B-Instruct-2501-AWQ`


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
For instructions on how to use the interactive chat client, please refer to [README.md](README.md).
