import re
import torch

from dynaweb_config import DynaWebConfig
from models.agent import AgentPolicy


ACTION_PATTERNS = [
    r"click \[\d+\]",
    r"type \[\d+\] \[.+?\] \[[01]\]",
    r"select \[\d+\] \[.+?\]",
    r"go_back",
    r"scroll \[(down|up)\]",
    r"stop \[.*?\]",
]


def extract_action(text: str):
    for pattern in ACTION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    prompt = """You are an AI assistant performing tasks on a web browser. You will be provided with a task objective, the current web page observations, and interaction history. You need to issue an action for this step.

Generate the response in the following format:
REASON:
Provide your rationale for the action you will take.

ACTION:
One of the following (issue exactly one):
- click [id]               — click element with that id. E.g., click [7]
- type [id] [text] [1]     — type text into field, press Enter. E.g., type [3] [hello] [1]
- type [id] [text] [0]     — type text into field, no Enter.
- select [id] [value]      — select option in dropdown.
- go_back                  — navigate to previous page.
- scroll [down|up]         — scroll the page.
- stop [answer]            — task complete. E.g., stop [Found item at $9.99]

TASK: Find the contact page.

CURRENT PAGE:
URL: https://example.com
[link id=1 'Home']
[link id=2 'Products']
[link id=3 'Contact']
[button id=4 'Search']

Provide REASON then ACTION:"""

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()
    }

    print("Generating action...")

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=False,
    )

    generated_only = tokenizer.decode(
        output_ids[0][encoded["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )

    print("\nFull decoded output:")
    print(decoded)

    print("\nGenerated only:")
    print(generated_only)

    action = extract_action(generated_only)

    print("\nExtracted action:", action)

    assert action is not None, "No valid web action found in generated output."

    print("\nEnd-to-end action generation test passed.")
    print("Agent generated a valid browser action.")


if __name__ == "__main__":
    main()