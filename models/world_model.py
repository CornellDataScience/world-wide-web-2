"""
models/world_model.py

Web World Model: given (WebState, WebAction) → predicted next WebState.

Implements Section 3.2 of the paper:
  1. Predict state changes Δ(o_t, o_{t+1}) in natural language
  2. Apply Δ to current state to get o_{t+1}

In stub mode, returns a minimally modified next state so the rollout
engine can run end-to-end without any GPU.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from config import DynaWebConfig
from data.datatypes import WebState, WebAction, StateChange, ActionType
from utils.prompt import format_world_model_prompt, parse_wm_output


class WebWorldModel(nn.Module):
    """
    LLM-based web world model.

    Given current WebState + WebAction, predicts next WebState.
    Trained on NNetNav transition pairs (StateChange objects).
    """

    def __init__(self, config: DynaWebConfig):
        super().__init__()
        self.config = config

        if config.use_stub_models:
            print("[WorldModel] Running in STUB mode — no real model loaded.")
            self.model = None
            self.tokenizer = None
        else:
            self._load_real_model()

    def _load_real_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[WorldModel] Loading {self.config.wm_model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.wm_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.config.precision == "bfloat16" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.wm_model_name,
            torch_dtype=dtype,
            device_map=self.config.device,
        )
        self.model.eval()
        print(f"[WorldModel] Loaded.")

    def predict_next_state(
        self,
        task: str,
        current_state: WebState,
        action: WebAction,
        prev_action: Optional[WebAction] = None,
    ) -> Tuple[str, WebState]:
        """
        Predict next web state given current state and action.

        Input:
            task:          str       — natural language task
            current_state: WebState  — current page state
            action:        WebAction — action just taken
            prev_action:   WebAction — previous action (optional)
        Output:
            state_changes: str      — natural language description of changes
            next_state:    WebState — predicted next page state
        """
        if self.config.use_stub_models:
            return self._stub_predict(current_state, action)

        return self._real_predict(task, current_state, action, prev_action)

    def _real_predict(
        self,
        task: str,
        current_state: WebState,
        action: WebAction,
        prev_action: Optional[WebAction],
    ) -> Tuple[str, WebState]:
        prompt = format_world_model_prompt(
            task=task,
            current_obs=current_state.to_model_input(),
            current_action=action.to_string(),
            prev_action=prev_action.to_string() if prev_action else None,
            url=current_state.url,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_len,
        ).to(self.config.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.wm_max_tokens,
                temperature=self.config.wm_temp,
                top_p=self.config.wm_top_p,
                do_sample=True,
            )

        prompt_len = inputs["input_ids"].shape[1]
        raw_output = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)

        state_changes, next_tree = parse_wm_output(raw_output)

        # Build next WebState from predicted tree
        next_state = WebState.from_acc_tree(
            acc_tree=next_tree if next_tree else current_state.acc_tree,
            url=self._predict_url(current_state.url, action),
        )

        return state_changes, next_state

    def _stub_predict(
        self,
        current_state: WebState,
        action: WebAction,
    ) -> Tuple[str, WebState]:
        """
        Stub world model: applies minimal deterministic state changes.
        Simulates realistic transitions for each action type.
        """
        action_type = action.action_type

        if action_type == ActionType.CLICK:
            # Simulate clicking: modify heading to indicate navigation
            new_tree = (
                f"[heading id=100 'Page after clicking id={action.target_id}']\n"
                f"[text id=101 'Content loaded']\n"
                f"[button id=102 'Back']\n"
                f"[link id=103 'Continue']"
            )
            changes = f"Clicked element {action.target_id}. Page content updated."

        elif action_type == ActionType.TYPE:
            # Simulate typing: add a result element
            new_tree = (
                current_state.acc_tree + "\n"
                f"[text id=200 'You typed: {action.value}']\n"
                f"[button id=201 'Submit']"
            )
            changes = f"Typed '{action.value}' into element {action.target_id}."

        elif action_type == ActionType.GO_BACK:
            # Simulate going back: return a generic previous page
            new_tree = (
                "[heading id=300 'Previous Page']\n"
                "[button id=301 'Home']\n"
                "[textbox id=302 'Search']\n"
                "[button id=303 'Submit']"
            )
            changes = "Navigated back to previous page."

        elif action_type == ActionType.STOP:
            # Terminal: keep current state
            new_tree = current_state.acc_tree
            changes = "Task stopped. No page change."

        else:
            new_tree = current_state.acc_tree
            changes = "No significant change detected."

        next_state = WebState.from_acc_tree(
            acc_tree=new_tree,
            url=self._predict_url(current_state.url, action),
        )
        return changes, next_state

    def _predict_url(self, current_url: str, action: WebAction) -> str:
        """Simple URL update heuristic."""
        if action.action_type == ActionType.GO_BACK:
            # Strip last path segment
            parts = current_url.rstrip("/").rsplit("/", 1)
            return parts[0] if len(parts) > 1 else current_url
        if action.action_type == ActionType.TYPE and action.value:
            base = current_url.split("?")[0]
            return f"{base}?q={action.value.replace(' ', '+')}"
        return current_url

    def is_terminal_state(self, action: WebAction) -> bool:
        """True if the action signals end of episode."""
        return action.is_terminal