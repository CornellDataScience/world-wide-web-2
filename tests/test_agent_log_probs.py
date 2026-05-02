import torch
import torch.nn as nn
from types import SimpleNamespace
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.agent import AgentPolicy


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        logits = self.lm_head(x)
        return FakeOutput(logits)


def make_fake_agent():
    from models.agent import AgentPolicy
    import torch.nn as nn

    agent = AgentPolicy.__new__(AgentPolicy)

    # Required because AgentPolicy inherits from nn.Module
    nn.Module.__init__(agent)

    agent.config = SimpleNamespace(use_stub_models=False)
    agent.model = FakeModel(vocab_size=100)

    return agent


def test_get_token_log_probs_shape_and_mask():
    agent = make_fake_agent()

    input_ids = torch.tensor([[10, 11, 12, 20, 21]])

    # first 3 tokens are prompt, last 2 are response/action
    labels = torch.tensor([[-100, -100, -100, 20, 21]])

    log_probs = agent.get_token_log_probs(input_ids, labels)

    print("log_probs:", log_probs)

    assert log_probs.shape == labels.shape

    # prompt positions must contribute exactly 0
    assert torch.all(log_probs[labels == -100] == 0.0)

    # response positions should be finite numbers
    assert torch.all(torch.isfinite(log_probs[labels != -100]))


def test_get_token_log_probs_allows_gradients():
    agent = make_fake_agent()

    input_ids = torch.tensor([
        [10, 11, 12, 20, 21],
        [13, 14, 15, 22, 23],
    ])

    labels = torch.tensor([
        [-100, -100, -100, 20, 21],
        [-100, -100, -100, 22, 23],
    ])

    log_probs = agent.get_token_log_probs(input_ids, labels)

    loss = -log_probs.sum()
    loss.backward()

    grads_exist = any(
        p.grad is not None
        for p in agent.model.parameters()
        if p.requires_grad
    )

    assert grads_exist, "No gradients are flowing through get_token_log_probs"
    
    
if __name__ == "__main__":
    test_get_token_log_probs_shape_and_mask()
    test_get_token_log_probs_allows_gradients()
    print("All tests passed!")