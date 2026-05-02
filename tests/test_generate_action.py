import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from types import SimpleNamespace

from models.agent import AgentPolicy


# =========================
# Fake HuggingFace Outputs
# =========================

class FakeGenerateOutput:
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class FakeBatch(dict):
    def to(self, device):
        return self


# =========================
# Fake Tokenizer
# =========================

class FakeTokenizer:
    def __call__(self, prompt, return_tensors="pt", truncation=True, max_length=None):
        return FakeBatch({
            "input_ids": torch.tensor([[1, 2, 3]])
        })

    def decode(self, token_ids, skip_special_tokens=True):
        # Always returns a valid parsed action
        return "ACTION: click [1]"


# =========================
# Fake Model
# =========================

class FakeGenerateModel(nn.Module):
    def generate(self, **kwargs):
        # prompt tokens = [1,2,3]
        # generated tokens = [20,21]
        sequences = torch.tensor([[1, 2, 3, 20, 21]])

        vocab_size = 100

        # First generated token logits
        score_0 = torch.zeros(1, vocab_size)
        score_0[0, 20] = 5.0

        # Second generated token logits
        score_1 = torch.zeros(1, vocab_size)
        score_1[0, 21] = 4.0

        scores = (score_0, score_1)

        return FakeGenerateOutput(sequences, scores)


# =========================
# Test Function
# =========================

def test_generate_action_log_probs():
    agent = AgentPolicy.__new__(AgentPolicy)
    nn.Module.__init__(agent)

    agent.config = SimpleNamespace(
        use_stub_models=False,
        device="cpu",
        max_prompt_len=32,
        max_response_len=8,
        rollout_temp=1.0,
        rollout_top_p=1.0,
    )

    agent.tokenizer = FakeTokenizer()
    agent.model = FakeGenerateModel()

    action, token_log_probs = agent._real_generate("fake prompt")

    print("\n=== OUTPUT ===")
    print("Action:", action)
    print("Token log probs:", token_log_probs)
    print("Sequence log prob:", token_log_probs.sum())

    # =========================
    # Assertions
    # =========================
    print("Returned action repr:", repr(action))
    assert action == "click [1]", "Parsed action incorrect"
    assert token_log_probs.shape[0] == 2, "Wrong number of tokens"

    # Compute expected values
    vocab = torch.zeros(100)

    logits_0 = vocab.clone()
    logits_0[20] = 5.0
    expected_0 = torch.log_softmax(logits_0, dim=-1)[20]

    logits_1 = vocab.clone()
    logits_1[21] = 4.0
    expected_1 = torch.log_softmax(logits_1, dim=-1)[21]

    assert torch.allclose(token_log_probs[0], expected_0), "First token log prob incorrect"
    assert torch.allclose(token_log_probs[1], expected_1), "Second token log prob incorrect"

    print("\n✅ TEST PASSED: generate_action is correct")


# =========================
# Run directly
# =========================

if __name__ == "__main__":
    test_generate_action_log_probs()