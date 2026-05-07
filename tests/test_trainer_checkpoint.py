import os
import shutil
import torch

from peft import PeftModel

from dynaweb_config import DynaWebConfig
from models.agent import AgentPolicy


CHECKPOINT_DIR = "tmp_test_lora_checkpoint"


def main():
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)

    config = DynaWebConfig()
    config.use_stub_models = False
    config.device = "auto"

    print("Loading AgentPolicy...")
    policy = AgentPolicy(config)

    model = policy.model
    tokenizer = policy.tokenizer

    assert model is not None, "Model did not load."
    assert tokenizer is not None, "Tokenizer did not load."

    print("Checking trainable parameters...")

    trainable_count = 0
    total_count = 0

    for name, param in model.named_parameters():
        total_count += param.numel()
        if param.requires_grad:
            trainable_count += param.numel()

    print("Trainable params:", trainable_count)
    print("Total params:", total_count)
    print("Trainable %:", 100 * trainable_count / total_count)

    assert trainable_count > 0, "No trainable LoRA parameters found."
    assert trainable_count < total_count, "Too many parameters are trainable; base model may not be frozen."

    print("Saving LoRA adapter checkpoint...")

    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    assert os.path.exists(CHECKPOINT_DIR), "Checkpoint directory was not created."

    files = os.listdir(CHECKPOINT_DIR)
    print("Saved files:")
    for f in files:
        print(" -", f)

    assert "adapter_config.json" in files, "Missing adapter_config.json"

    has_adapter_weights = (
        "adapter_model.safetensors" in files
        or "adapter_model.bin" in files
    )

    assert has_adapter_weights, "Missing LoRA adapter weights."

    print("Checkpoint save test passed.")

    print("Testing checkpoint reload...")

    reloaded_model = PeftModel.from_pretrained(
        model.base_model.model,
        CHECKPOINT_DIR,
        is_trainable=True,
    )

    assert reloaded_model is not None, "Reloaded LoRA model is None."

    reloaded_trainable = 0

    for name, param in reloaded_model.named_parameters():
        if param.requires_grad:
            reloaded_trainable += param.numel()

    print("Reloaded trainable params:", reloaded_trainable)

    assert reloaded_trainable > 0, "Reloaded model has no trainable LoRA parameters."

    print("Checkpoint reload test passed.")

    print("\nTrainer checkpoint test passed.")
    print("LoRA adapter can be saved and reloaded successfully.")

    shutil.rmtree(CHECKPOINT_DIR)


if __name__ == "__main__":
    main()