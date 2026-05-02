import torch

from dynaweb_config import DynaWebConfig
from models.agent import AgentPolicy
from training.mixer import TrajectoryMixer


def main():
    config = DynaWebConfig()
    config.use_stub_models = False
    config.device = "auto"

    print("Loading AgentPolicy...")
    policy = AgentPolicy(config)

    tokenizer = policy.tokenizer

    interactions = [
        {
            "prompt": "Observation: You are on a search page.\nTask: Find the contact page.\nAction:",
            "response": " click [3]",
        },
        {
            "prompt": "Observation: You are on a product page.\nTask: Add item to cart.\nAction:",
            "response": " click [7]",
        },
        {
            "prompt": "Observation: A form is visible.\nTask: Enter the username.\nAction:",
            "response": " type [2] mihir",
        },
        {
            "prompt": "Observation: You are on a login page.\nTask: Submit the form.\nAction:",
            "response": " click [5]",
        },
        {
            "prompt": "Observation: Search results are visible.\nTask: Open the first result.\nAction:",
            "response": " click [1]",
        },
    ]

    mixer = mixer = TrajectoryMixer(
        sft_dataset=None,
        config=config,
        tokenizer=policy.tokenizer,
    )

    batch = mixer.to_gspo_batch(
        mixed=interactions,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    print("input_ids shape:", input_ids.shape)
    print("labels shape:", labels.shape)
    print("attention_mask shape:", attention_mask.shape)

    assert input_ids.shape == labels.shape
    assert input_ids.shape == attention_mask.shape
    assert input_ids.shape[0] == 5

    for i, item in enumerate(interactions):
        prompt_ids = tokenizer(
            item["prompt"],
            add_special_tokens=True,
        )["input_ids"]

        response_ids = tokenizer(
            item["response"],
            add_special_tokens=False,
        )["input_ids"]

        seq_labels = labels[i].detach().cpu().tolist()

        # Remove left padding if tokenizer pads left
        non_pad_positions = attention_mask[i].detach().cpu().nonzero().flatten().tolist()
        first_real_token = non_pad_positions[0]

        prompt_label_slice = seq_labels[
            first_real_token : first_real_token + len(prompt_ids)
        ]

        response_label_slice = seq_labels[
            first_real_token + len(prompt_ids) :
            first_real_token + len(prompt_ids) + len(response_ids)
        ]

        print("\nExample", i)
        print("Decoded prompt:")
        print(tokenizer.decode(prompt_ids))

        print("Decoded response:")
        print(tokenizer.decode(response_ids))

        print("Prompt labels:", prompt_label_slice)
        print("Response labels:", response_label_slice)

        assert all(x == -100 for x in prompt_label_slice), \
            f"Prompt tokens are contributing to loss in example {i}"

        assert response_label_slice == response_ids, \
            f"Response labels do not match response ids in example {i}"

    print("\nGSPO batch test passed.")
    print("Prompt tokens are masked with -100.")
    print("Response/action tokens contribute to loss.")


if __name__ == "__main__":
    main()