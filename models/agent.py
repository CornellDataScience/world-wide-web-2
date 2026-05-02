"""
models/agent.py

Agent policy: LLaMA-3.1-8B-Instruct (or stub for testing).

The agent takes a formatted prompt string and returns:
  - a generated action string (e.g. "click [7]")
  - per-token log probabilities of the response (for GSPO)

In stub mode, returns deterministic fake actions and log probs so the
full training loop can be tested without a GPU or model weights.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from dynaweb_config import DynaWebConfig
from data.types import WebInteraction, WebState, WebAction
from utils.prompt import format_agent_prompt, parse_agent_output


class AgentPolicy(nn.Module):
    """
    Wraps LLaMA-3.1-8B-Instruct for web agent action generation.

    Key methods:
        generate_action()       — produce one action string + log probs
        get_token_log_probs()   — score a full sequence (for GSPO ratio)
    """

    def __init__(self, config: DynaWebConfig):
        super().__init__()
        self.config = config

        if config.use_stub_models:
            print("[AgentPolicy] Running in STUB mode — no real model loaded.")
            self.model = None
            self.tokenizer = None
        else:
            self._load_real_model()

    def _load_real_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import torch

        model_name = getattr(
            self.config,
            "agent_model_name",
            "stanfordnlp/llama8b-nnetnav-live",
        )

        print(f"[AgentPolicy] Loading {model_name} with 4-bit quantization + LoRA ...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        quantization_config = None

        if getattr(self.config, "use_4bit", True):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=getattr(self.config, "device", "auto"),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.model.config.use_cache = False

        if getattr(self.config, "use_lora", True):
            if getattr(self.config, "use_4bit", True):
                self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=getattr(self.config, "lora_r", 16),
                lora_alpha=getattr(self.config, "lora_alpha", 32),
                lora_dropout=getattr(self.config, "lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.model.train()

        print("[AgentPolicy] Loaded NNetNav with 4-bit quantization + LoRA.")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Standard forward pass. Returns CausalLMOutput."""
        assert self.model is not None, "forward() called in stub mode"
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def generate_action(
        self,
        prompt: str,
        step: int = 0,
    ) -> Tuple[str, torch.Tensor]:
        """
        Generate one action from a prompt string.

        Input:
            prompt: str   — formatted agent prompt
            step:   int   — current step index (used by stub for variety)
        Output:
            action_text:     str            — e.g. "click [7]"
            token_log_probs: Tensor[L]      — log probs of response tokens
        """
        if self.config.use_stub_models:
            return self._stub_generate(step)

        return self._real_generate(prompt)

    def _real_generate(self, prompt: str) -> Tuple[str, torch.Tensor]:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_len,
        ).to(self.config.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_response_len,
                temperature=self.config.rollout_temp,
                top_p=self.config.rollout_top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode generated tokens only (exclude prompt)
        prompt_len = inputs["input_ids"].shape[1]
        response_ids = out.sequences[0, prompt_len:]
        action_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Compute per-token log probs from scores
        # out.scores is a tuple of Tensors [1, vocab_size], one per generated token
        log_probs = []
        for i, score in enumerate(out.scores):
            token_id = response_ids[i]
            lp = torch.log_softmax(score[0], dim=-1)[token_id]
            log_probs.append(lp)
        token_log_probs = torch.stack(log_probs) if log_probs else torch.tensor([0.0])

        # Parse out just the ACTION line
        parsed = parse_agent_output(action_text)
        return parsed["action"], token_log_probs

    def _stub_generate(self, step: int = 0) -> Tuple[str, torch.Tensor]:
        """
        Deterministic fake generation for testing.
        Cycles through a small set of plausible actions.
        Token log probs are small negative values (like a real model).
        """
        stub_actions = [
            "click [1]",
            "type [3] [search term] [1]",
            "click [11]",
            "go_back",
            "stop [Task completed]",
        ]
        action = stub_actions[step % len(stub_actions)]
        # Fake log probs: uniform ~log(1/vocab) ≈ -10, length = num tokens in action
        n_tokens = len(action.split()) + 2
        token_log_probs = torch.full((n_tokens,), -10.0)
        return action, token_log_probs

    def get_token_log_probs(
        self,
        input_ids: torch.Tensor,       # [B, L]
        labels: torch.Tensor,          # [B, L]  (-100 for prompt positions)
    ) -> torch.Tensor:
        """
        Compute per-token log probs for full sequences.
        Used for GSPO ratio computation.

        Input:
            input_ids: Tensor[B, L]
            labels:    Tensor[B, L]  — -100 for tokens we don't score
        Output:
            Tensor[B, L]  — log probs, 0.0 where label == -100
        """
        if self.config.use_stub_models:
            return self._stub_log_probs(input_ids, labels)

        out = self.model(input_ids=input_ids)
        logits = out.logits

        log_probs = torch.log_softmax(logits, dim=-1)  # [B, L, vocab]

        # Gather log prob for each actual token
        shifted_labels = labels[:, 1:].clone()             # [B, L-1]
        shifted_log_probs = log_probs[:, :-1, :]           # [B, L-1, vocab]

        # Mask out -100 positions
        mask = (shifted_labels != -100)
        safe_labels = shifted_labels.clone()
        safe_labels[~mask] = 0

        gathered = shifted_log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        gathered = gathered * mask.float()

        # Pad back to [B, L]
        pad = torch.zeros(input_ids.shape[0], 1, device=input_ids.device)
        return torch.cat([pad, gathered], dim=1)

    def _stub_log_probs(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Stub: return small uniform log probs where labels != -100."""
        mask = (labels != -100).float()
        return torch.full_like(mask, -10.0) * mask

    def score_reward_prompt(self, prompt: str) -> float:
        """
        Run the reward assessment prompt through the model and parse YES/NO.
        Used by SelfAssessReward.

        Input:  prompt: str
        Output: 1.0 (YES) or 0.0 (NO)
        """
        if self.config.use_stub_models:
            # Stub: reward 1.0 for interactions that have 'stop' actions
            return 1.0 if "stop [" in prompt.lower() else 0.0

        from utils.prompt import parse_reward_output
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_len,
        ).to(self.config.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        return parse_reward_output(response)
    
    def save_adapter(self, save_dir: str):
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"[AgentPolicy] Saved LoRA adapter to {save_dir}")