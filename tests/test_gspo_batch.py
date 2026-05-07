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


def main():
    config = DynaWebConfig()
    config.use_stub_models = False
    config.device = "auto"

    print("Loading AgentPolicy...")
    policy = AgentPolicy(config)

    tokenizer = policy.tokenizer

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
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    print("input_ids shape:", input_ids.shape)
    print("labels shape:", labels.shape)
    print("attention_mask shape:", attention_mask.shape)

    for i in range(len(mixed)):
        seq_input_ids = input_ids[i]
        seq_labels = labels[i]
        seq_attention = attention_mask[i]

        scored_mask = seq_labels != -100
        prompt_mask = (seq_labels == -100) & (seq_attention == 1)

        scored_label_ids = seq_labels[scored_mask].detach().cpu().tolist()
        scored_input_ids = seq_input_ids[scored_mask].detach().cpu().tolist()

        decoded_full = tokenizer.decode(
            seq_input_ids[seq_attention == 1],
            skip_special_tokens=False,
        )

        decoded_scored_labels = tokenizer.decode(
            scored_label_ids,
            skip_special_tokens=False,
        )

        decoded_scored_input_ids = tokenizer.decode(
            scored_input_ids,
            skip_special_tokens=False,
        )

        print("\nExample", i)
        print("Full decoded prompt + response:")
        print(decoded_full)

        print("\nDecoded scored labels:")
        print(decoded_scored_labels)

        print("\nDecoded scored input ids:")
        print(decoded_scored_input_ids)

        print("Prompt tokens masked:", prompt_mask.sum().item())
        print("Response/action tokens scored:", scored_mask.sum().item())

        assert prompt_mask.sum().item() > 0, \
            f"No prompt tokens were masked in example {i}"

        assert scored_mask.sum().item() > 0, \
            f"No response/action tokens were scored in example {i}"

        assert scored_label_ids == scored_input_ids, \
            f"Scored labels do not match scored input ids in example {i}"

        assert "click" in decoded_scored_labels or "type" in decoded_scored_labels, \
            f"Decoded scored response does not look like an action in example {i}"

    print("\nGSPO batch test passed.")
    print("Prompt tokens are masked with -100.")
    print("Response/action tokens contribute to loss.")


if __name__ == "__main__":
    main()