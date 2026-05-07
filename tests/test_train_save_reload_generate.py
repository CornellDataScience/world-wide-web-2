import os
import shutil
import torch

from peft import PeftModel

from dynaweb_config import DynaWebConfig
from models.agent import AgentPolicy
from training.mixer import TrajectoryMixer
from data.types import WebInteraction, WebState, WebAction, StateChange


CHECKPOINT_DIR = "tmp_test_trained_lora_adapter"


def make_fake_interaction(i: int):
    examples = [
        ("You are on a search page.", "Find the contact page.", "click [3]"),
        ("You are on a product page.", "Add item to cart.", "click [7]"),
        ("A form is visible.", "Enter the username.", "type [2] [mihir] [1]"),
        ("You are on a login page.", "Submit the form.", "click [5]"),
        ("Search results are visible.", "Open the first result.", "click [1]"),
    ]

    obs, task, action_str = examples[i]

    pre = WebState.from_acc_tree(
        acc_tree=obs,
        url=f"https://example.com/{i}",
    )

    post = WebState.from_acc_tree(
        acc_tree=f"Result after action {i}",
        url=f"https://example.com/{i}/next",
    )

    action = WebAction.from_string(action_str)

    state_change = StateChange(
        pre_state=pre,
        action=action,
        post_state=post,
    )

    return WebInteraction(
        task=task,
        website="example.com",
        state_changes=[state_change],
        reward=1.0,
        final_state=post,
        is_real=True,
    )


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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.train()

    mixer = TrajectoryMixer(
        sft_dataset=None,
        config=config,
        tokenizer=tokenizer,
    )

    mixed = [make_fake_interaction(i) for i in range(5)]

    batch = mixer.to_gspo_batch(
        mixed=mixed,
        device=device,
    )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    print("input_ids shape:", input_ids.shape)
    print("attention_mask shape:", attention_mask.shape)
    print("labels shape:", labels.shape)

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    assert len(trainable_params) > 0, "No trainable LoRA params found."

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-5,
    )

    optimizer.zero_grad(set_to_none=True)

    print("Running one training step...")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss

    print("Loss:", loss.item())

    assert torch.isfinite(loss), "Loss is NaN or Inf."

    loss.backward()
    optimizer.step()

    print("Saving trained LoRA adapter...")

    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    files = os.listdir(CHECKPOINT_DIR)

    print("Saved checkpoint files:")
    for f in files:
        print(" -", f)

    assert "adapter_config.json" in files, "Missing adapter_config.json"
    assert (
        "adapter_model.safetensors" in files
        or "adapter_model.bin" in files
    ), "Missing adapter model weights."

    print("Reloading base AgentPolicy...")

    reloaded_policy = AgentPolicy(config)
    base_model = reloaded_policy.model
    reloaded_tokenizer = reloaded_policy.tokenizer

    assert base_model is not None, "Reloaded base model is None."
    assert reloaded_tokenizer is not None, "Reloaded tokenizer is None."

    print("Loading trained LoRA adapter into reloaded model...")

    reloaded_model = PeftModel.from_pretrained(
        base_model.base_model.model,
        CHECKPOINT_DIR,
        is_trainable=False,
    )

    reloaded_model.eval()

    prompt = (
        "You are an AI assistant performing tasks on a web browser.\n"
        "TASK: Find the contact page.\n"
        "CURRENT PAGE:\n"
        "URL: https://example.com\n"
        "[link id=3 'Contact']\n"
        "Provide REASON then ACTION:"
    )

    encoded = reloaded_tokenizer(
        prompt,
        return_tensors="pt",
    )

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()
    }

    print("Generating from reloaded trained adapter...")

    with torch.no_grad():
        generated = reloaded_model.generate(
            **encoded,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=reloaded_tokenizer.eos_token_id,
        )

    decoded = reloaded_tokenizer.decode(
        generated[0],
        skip_special_tokens=False,
    )

    print("\nGenerated text:")
    print(decoded)

    assert len(decoded) > len(prompt), "Generation did not produce new text."

    print("\nTrain-save-reload-generate test passed.")
    print("One training step completed.")
    print("Trained LoRA adapter saved.")
    print("Trained LoRA adapter reloaded.")
    print("Reloaded adapter can generate text.")

    shutil.rmtree(CHECKPOINT_DIR)


if __name__ == "__main__":
    main()