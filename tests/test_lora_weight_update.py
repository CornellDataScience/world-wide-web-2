import torch

from dynaweb_config import DynaWebConfig
from models.agent import AgentPolicy
from training.mixer import TrajectoryMixer
from data.types import WebInteraction, WebState, WebAction, StateChange


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


def snapshot_lora_params(model):
    snapshots = {}

    for name, param in model.named_parameters():
        if param.requires_grad and "lora" in name.lower():
            snapshots[name] = param.detach().float().cpu().clone()

    return snapshots


def main():
    config = DynaWebConfig()
    config.use_stub_models = False
    config.device = "auto"

    print("Loading AgentPolicy...")
    policy = AgentPolicy(config)

    model = policy.model
    tokenizer = policy.tokenizer

    assert model is not None, "Model did not load."
    assert tokenizer is not None, "Tokenizer did not load."

    model.train()

    mixer = TrajectoryMixer(
        sft_dataset=None,
        config=config,
        tokenizer=tokenizer,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    lora_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and "lora" in name.lower()
    ]

    assert len(lora_params) > 0, "No trainable LoRA parameters found."

    before = snapshot_lora_params(model)

    print("Number of trainable LoRA tensors:", len(before))

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=1e-5,
    )

    optimizer.zero_grad(set_to_none=True)

    print("Running forward pass...")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss
    print("Loss:", loss.item())

    assert torch.isfinite(loss), "Loss is NaN or Inf."

    print("Running backward pass...")
    loss.backward()

    nonzero_grad_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad and "lora" in name.lower():
            if param.grad is not None:
                grad_norm = param.grad.detach().float().norm().item()
                if grad_norm > 0:
                    nonzero_grad_count += 1
                    print("Non-zero LoRA grad:", name, "norm:", grad_norm)

    assert nonzero_grad_count > 0, "No LoRA parameters had non-zero gradients."

    print("Running optimizer step...")
    optimizer.step()

    changed_count = 0

    for name, param in model.named_parameters():
        if name in before:
            after = param.detach().float().cpu()

            if not torch.equal(before[name], after):
                changed_count += 1
                print("Changed LoRA param:", name)

    assert changed_count > 0, "No LoRA parameters changed after optimizer step."

    print("\nLoRA weight update test passed.")
    print("At least one LoRA parameter had non-zero gradient.")
    print("At least one LoRA parameter changed after optimizer step.")


if __name__ == "__main__":
    main()