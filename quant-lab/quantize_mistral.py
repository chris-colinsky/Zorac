import os

from datasets import load_dataset
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

# configuration
model_id = "mistralai/Mistral-Small-24B-Instruct-2501"
quant_path = "/path-to-output/models/Mistral-Small-24B-Instruct-2501-AWQ"

# calibration dataset settings
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512


def main():
    print("--- Starting AWQ Quantization (llm-compressor) ---")
    print(f"Target Model: {model_id}")

    # load tokenizer
    # fix_mistral_regex was added in transformers 4.50; guard for compatibility with <4.50
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, fix_mistral_regex=True
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # load model to CPU — oneshot handles sequential onloading to GPU layer by layer
    print("Loading base model to CPU...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map=None)

    # prepare calibration dataset
    print("Preparing calibration dataset...")
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]"
    )
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    ds = ds.map(preprocess)

    # AWQ recipe: W4A16_ASYM = 4-bit weights, 16-bit activations, group_size=128, asymmetric
    recipe = """
default_stage:
  default_modifiers:
    AWQModifier:
      scheme: W4A16_ASYM
      targets: [Linear]
      ignore: [lm_head]
"""

    # calibrate and quantize (sequential onloading to GPU enabled by default)
    print("Beginning quantization (this will take a while)...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # save
    os.makedirs(quant_path, exist_ok=True)
    print(f"Quantization successful! Saving to: {quant_path}")
    model.save_pretrained(quant_path, save_compressed=True)
    tokenizer.save_pretrained(quant_path)

    print(f"Done. Serve with: vllm serve {quant_path}")


if __name__ == "__main__":
    main()
