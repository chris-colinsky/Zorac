# Multi-GPU Training

How to use multiple consumer GPUs for fine-tuning LLMs — covering power management, environment setup, and running training jobs alongside an inference server.

!!! tip "Reference Documentation"
    This guide explains the concepts and reasoning behind multi-GPU training on consumer hardware. For the specific commands and configuration used in Zorac's setup, see the [Server Setup Reference](../../SERVER_SETUP.md).

---

## Why Fine-Tune?

### When Fine-Tuning Makes Sense

Pre-trained models like Mistral-Small-24B are general-purpose — they perform well across a wide range of tasks. But sometimes you need a model that excels at a specific domain:

- **Custom writing style** — Training on your own writing to match your voice
- **Domain-specific knowledge** — Legal documents, medical terminology, internal codebases
- **Specialized tasks** — Structured data extraction, specific code patterns, custom output formats
- **Behavior alignment** — Adjusting how the model responds to certain types of queries

Fine-tuning takes a pre-trained model and continues training on your specific data, adapting its behavior without starting from scratch.

### LoRA vs Full Fine-Tuning

There are two main approaches to fine-tuning:

| Approach | Memory Required | Training Time | Quality |
|----------|----------------|---------------|---------|
| **Full fine-tuning** | ~4x model size | Hours to days | Best, but risks overfitting |
| **LoRA** (Low-Rank Adaptation) | ~1.2x model size | Minutes to hours | Very good for most tasks |

**Full fine-tuning** updates all model parameters. For a 24B model at FP16, this requires ~96GB of memory for weights + gradients + optimizer states — far more than what fits on consumer GPUs even with multi-GPU setups.

**LoRA** freezes the original model weights and adds small trainable "adapter" layers (typically 0.1-1% of the total parameters). This dramatically reduces memory requirements:

```
Full fine-tuning (24B FP16): ~96 GB minimum
LoRA fine-tuning (24B FP16):  ~48 GB (fits on 2x 24GB GPUs)
LoRA + QLoRA (24B 4-bit):     ~24 GB (fits on a single 24GB GPU)
```

For consumer hardware, LoRA (or QLoRA, which combines LoRA with quantization) is the practical choice.

### Hardware Requirements

Multi-GPU fine-tuning pools VRAM from multiple cards. For Zorac's setup:

```
GPU 0: RTX 3090 Ti (24GB) — Training pool
GPU 1: RTX 4090 (24GB)    — Training pool (normally inference)
───────────────────────────
Total:              48GB VRAM
```

With 48GB pooled VRAM, you can fine-tune models up to ~24B parameters using LoRA at FP16, or full fine-tune models up to ~7B parameters.

!!! info "You Don't Need Two GPUs"
    QLoRA (quantized LoRA) can fine-tune a 24B model on a single 24GB GPU. Multi-GPU just gives you more headroom and faster training. Many fine-tuning tasks work perfectly well with a single card.

---

## Power Safety

### Understanding Transient Power Spikes

Consumer GPUs are designed for gaming, where power draw is relatively steady. Training workloads create **transient power spikes** — brief surges that can exceed the GPU's rated TDP by 50% or more.

A single RTX 4090 has a 450W TDP but can spike to ~600W during training. An RTX 3090 Ti (350W TDP) can spike to ~650W. Running both simultaneously:

```
RTX 3090 Ti transient peak:  ~650W
RTX 4090 transient peak:     ~600W
CPU + system:                ~200W
──────────────────────────────────
Worst case total:           ~1450W
```

A 1500W PSU has overcurrent protection (OCP) that trips when power draw exceeds its rated capacity. Without power limits, simultaneous training on both GPUs **will** trip the PSU, causing an immediate system shutdown.

### Setting Power Limits with nvidia-smi

Before starting any multi-GPU training run, cap both GPUs:

```bash
# Cap both cards at 350W each (700W total GPU load)
sudo nvidia-smi -pl 350
```

This limits each GPU's sustained power draw to 350W, keeping the combined load well within PSU capacity:

```
2x GPUs at 350W cap:  700W
CPU + system:         200W
─────────────────────────
Total:                900W  (safe on 1500W PSU)
```

!!! warning "Always Set Power Limits Before Training"
    This is not optional. Failing to set power limits on a multi-GPU consumer system risks PSU trips that can corrupt data and potentially damage hardware. Make it the first step of every training session.

To check current power limits:

```bash
nvidia-smi -q -d POWER | grep -E "Power Limit|Power Draw"
```

### PSU Sizing for Multi-GPU

If you're building a multi-GPU training system, size your PSU with transient spikes in mind:

| GPU Configuration | Minimum PSU | Recommended PSU |
|-------------------|-------------|-----------------|
| Single RTX 4090 | 850W | 1000W |
| RTX 3090 + RTX 4090 | 1200W | 1500W |
| 2x RTX 4090 | 1300W | 1600W |

Always use a high-quality 80+ Platinum or Titanium rated PSU. Lower-rated PSUs have tighter OCP thresholds and are more likely to trip under transient loads.

---

## Environment Setup

### Separate Virtual Environments

Training dependencies (PyTorch, Transformers, PEFT, Accelerate) are different from inference dependencies (vLLM). Keep them in separate virtual environments to avoid version conflicts:

```bash
# Inference environment (already set up)
~/vllm-serve/venv/

# Training environment (new)
mkdir -p ~/training && cd ~/training
python3 -m venv venv
source venv/bin/activate
```

### Installing Training Dependencies

Install the core training stack:

```bash
pip install torch transformers datasets accelerate peft bitsandbytes
```

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch — the deep learning framework |
| `transformers` | Hugging Face model loading and training utilities |
| `datasets` | Dataset loading and preprocessing |
| `accelerate` | Multi-GPU training orchestration |
| `peft` | Parameter-Efficient Fine-Tuning (LoRA, QLoRA) |
| `bitsandbytes` | 4-bit quantization for QLoRA |

### Accelerate Configuration

Hugging Face's `accelerate` library handles multi-GPU coordination. Run the configuration wizard:

```bash
accelerate config
```

Answer the prompts:

```
- Compute environment: This machine
- Machine type: multi-GPU
- Number of GPUs: 2
- Mixed precision: fp16 (or bf16 if supported)
```

This creates a configuration file at `~/.cache/huggingface/accelerate/default_config.yaml` that tells `accelerate` how to distribute work across your GPUs.

To verify the setup:

```bash
accelerate env
```

This should show both GPUs and the selected configuration.

---

## Running Training Jobs

### Stopping the Inference Server

Training uses all available GPU memory. The vLLM inference server must be stopped first to free GPU 1 (the RTX 4090):

```bash
sudo systemctl stop vllm
```

Verify both GPUs are free:

```bash
nvidia-smi
# Both GPUs should show 0 MiB memory usage (or minimal driver overhead)
```

### Launching with Accelerate

Use `accelerate launch` to distribute training across both GPUs:

```bash
# Activate the training environment
source ~/training/venv/bin/activate

# Set power limits FIRST
sudo nvidia-smi -pl 350

# Launch training
accelerate launch train_script.py \
    --model_name "mistralai/Mistral-Small-24B-Instruct-2501" \
    --dataset "your-dataset" \
    --output_dir "./output" \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4
```

`accelerate launch` automatically:

1. Detects both GPUs from your accelerate config
2. Spawns a training process on each GPU
3. Handles gradient synchronization between GPUs
4. Manages mixed precision (FP16) training

### Monitoring Training Progress

During training, monitor GPU utilization and power:

```bash
# Terminal 1: GPU monitoring
watch -n1 nvidia-smi

# Terminal 2: Training logs (if using wandb or tensorboard)
tensorboard --logdir ./output/runs
```

Healthy training metrics:

- **GPU memory:** Both GPUs at 80-95% utilization
- **Power draw:** Both under your set power limit (350W)
- **Temperature:** 70-85C (check cooling if above 85C)
- **GPU utilization:** 90-100% during forward/backward pass, brief drops during data loading

!!! info "Training Speed"
    Fine-tuning a 24B model with LoRA across two 24GB GPUs processes roughly 1-5 samples per second depending on sequence length and batch size. A small dataset (1000 samples, 3 epochs) typically completes in under an hour.

---

## Returning to Inference

### Restarting vLLM

After training completes, restart the inference server:

```bash
# Remove power limits (return to default)
sudo nvidia-smi -pl 450  # RTX 4090 default TDP

# Restart vLLM
sudo systemctl start vllm

# Verify it's running
sudo journalctl -u vllm -f -o cat
```

Wait for the log message indicating the model has loaded and the server is ready (typically 30-60 seconds).

### Testing the Fine-Tuned Model

If you fine-tuned with LoRA, you have adapter weights that can be merged with the base model or loaded separately:

**Option 1: Merge and serve the merged model**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("base-model-name")
model = PeftModel.from_pretrained(base_model, "./output/checkpoint-final")
merged = model.merge_and_unload()
merged.save_pretrained("./merged-model")
```

Then serve the merged model with vLLM:

```bash
vllm serve "./merged-model" --quantization awq_marlin ...
```

**Option 2: Update the Zorac model configuration**

If you're serving a new model, update Zorac to point to it:

```bash
/config set VLLM_MODEL your-merged-model-name
```

Test the fine-tuned model with prompts from your training domain to verify the fine-tuning worked as expected.
