import os
import shutil
import torch

from dynaweb_config import DynaWebConfig
from models.agent import AgentPolicy
from training.mixer import TrajectoryMixer
from data.types import WebInteraction, WebState, WebAction, StateChange


CHECKPOINT_DIR = "tmp_tiny_loop_ckpt"


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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.train()

    mixer = TrajectoryMixer(
        sft_dataset=None,
        config=config,
        tokenizer=tokenizer,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
    )

    print("Starting tiny training loop...\n")

    for step in range(3):
        mixed = [make_fake_interaction(i) for i in range(5)]

        batch = mixer.to_gspo_batch(
            mixed=mixed,
            device=device,
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        print(f"Step {step} loss: {loss.item()}")

        assert torch.isfinite(loss), f"Loss exploded at step {step}"

        loss.backward()

        # quick gradient sanity check
        grad_norm = 0.0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                grad_norm += p.grad.detach().float().norm().item()

        print(f"Step {step} grad norm: {grad_norm}")

        optimizer.step()

        # save checkpoint each step
        step_dir = os.path.join(CHECKPOINT_DIR, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)

        model.save_pretrained(step_dir)
        tokenizer.save_pretrained(step_dir)

        print(f"Saved checkpoint at {step_dir}\n")

    print("\nTiny training loop test passed.")
    print("Multiple steps completed without crash.")
    print("Loss remained finite.")
    print("Checkpoints saved successfully.")

    shutil.rmtree(CHECKPOINT_DIR)


if __name__ == "__main__":
    main()