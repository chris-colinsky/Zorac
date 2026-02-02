# **Test Training Guide (Multi-GPU)**

This guide outlines how to perform a "Hello World" training run to verify that **both** the RTX 4090 and RTX 3090 Ti can work together under load.

**Prerequisites:**

* You have sudo access (for power limits and stopping services).

## **1. Stop Inference Service (Critical)**

vLLM is currently holding \~20GB of VRAM on the 4090\. You must stop it to free the card for training.

`sudo systemctl stop vllm`

*Verify both GPUs are idle (Memory Usage \~0MB):*

`nvidia-smi`

## **2. Safety: Apply Power Limits**

**WARNING:** Training places a constant 100% load on both cards. To protect the 1500W PSU from transient spikes, you **must** apply these limits before starting.

\# Cap both cards at 350W (700W total GPU load)
`sudo nvidia-smi -pl 350`

## **3. Setup Training Environment**

Create a separate folder and virtual environment for training to keep the stable vllm-serve environment clean. Use uv for faster setup.

1\. Install uv (if not already installed)
``` bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
source $HOME/.cargo/env  # Load uv into path if just installed
```

2\. Create directory
```bash
mkdir -p ~/training-test
cd ~/training-test
```

3\. Create Virtual Environment
```bash
uv venv
```

4\. Activate (Note: uv defaults to .venv)
```bash
source .venv/bin/activate
```

5\. Install Training Libraries (HuggingFace stack)
```bash
uv pip install torch transformers accelerate datasets trl peft bitsandbytes
```

## **4. Configure Accelerate**

HuggingFace accelerate handles the multi-GPU complexity for us. Generate a config file that knows you have 2 GPUs.

Run this command and verify the output file matches the configuration below:

`accelerate config default`

*This usually detects both GPUs automatically.*

**To be safe, overwrite the config with this Multi-GPU specific setup:**

```bash
mkdir -p ~/.cache/huggingface/accelerate
nano ~/.cache/huggingface/accelerate/default_config.yaml
```

**Paste this content:**
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## **5. Create the Test Script**

Create a python script that fine-tunes a tiny model (GPT-2) just to prove data is flowing to both cards.

create the file `test_train.py`

```python
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# 1. Force CUDA Order to match nvidia-smi (0=3090, 1=4090)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def main():
    # 2. Check Device Visibility
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    # 3. Load a Tiny Model & Dataset
    model_name = "gpt2" # Tiny model for speed test
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Simple dummy dataset
    dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train[:100]")

    def tokenize_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=128)
        # CRITICAL FIX: Set labels = input_ids so the model can calculate loss
        tokenized["labels"] = tokenized["input_ids"][:]
        return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Training Arguments (optimized for Multi-GPU)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8, # 8 per card = 16 total batch
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-5,
        bf16=True, # Use BF16 (Matches your hardware & config)
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    print("\n>>> STARTING TRAINING ON ALL GPUS <<<\n")
    trainer.train()
    print("\n>>> TRAINING COMPLETE! <<<\n")

if __name__ == "__main__":
    main()
```

## **6. Run the Test**

Launch the training run using accelerate. This will automatically spawn processes on both GPUs.

`accelerate launch test_train.py`

### **What to Watch For**

1. **Terminal Output:** You should see a progress bar.
2. **Hardware Monitor:** Open a second terminal and run btop (press 5).
   * **Success:** Both the RTX 4090 and RTX 3090 Ti should show high Utilization (80-100%) and high VRAM usage.
   * **Power:** Check the power draw. It should be consistent (around 300W-350W per card due to our cap).

## **7. Cleanup & Restore Inference**

Once the test finishes successfully:

**1. Deactivate Training Env:**

`deactivate`

**2. Restore Power Limits (Optional):**

You can leave them capped at 350W (safer) or return them to stock (450W).

\# Return to stock (Optional)
`sudo nvidia-smi -pl 450`

**3. Restart vLLM:**

`sudo systemctl start vllm`

**4. Verify vLLM is back:**

\# Wait 10 seconds, then check logs
`sudo journalctl \-u vllm \-n 20 \--no-pager`
