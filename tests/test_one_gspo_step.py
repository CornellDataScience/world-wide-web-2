import copy
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


def get_first_trainable_param_snapshot(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            return name, param.detach().clone()
    raise RuntimeError("No trainable parameter found.")


def main():
    config = DynaWebConfig()
    config.use_stub_models = False
    config.device = "auto"

    print("Loading AgentPolicy...")
    policy = AgentPolicy(config)

    model = policy.model
    tokenizer = policy.tokenizer

    assert model is not None
    assert tokenizer is not None

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

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    assert len(trainable_params) > 0, "No trainable LoRA params found."

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-5,
    )

    before_name, before_value = get_first_trainable_param_snapshot(model)

    print("Tracking parameter:", before_name)

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

    grad_found = False

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.detach().float().norm().item()
            print("First LoRA grad:", name, "norm:", grad_norm)
            grad_found = True
            break

    assert grad_found, "No gradients found on LoRA parameters."

    print("Running optimizer step...")
    optimizer.step()

    after_param = dict(model.named_parameters())[before_name].detach()

    changed = not torch.allclose(
        before_value.float().cpu(),
        after_param.float().cpu(),
        atol=0,
        rtol=0,
    )

    assert changed, "Tracked LoRA parameter did not change after optimizer step."

    print("\nOne GSPO training step test passed.")
    print("Forward pass works.")
    print("Loss is finite.")
    print("Backward pass works.")
    print("LoRA gradients exist.")
    print("Optimizer updates LoRA weights.")


if __name__ == "__main__":
    main()