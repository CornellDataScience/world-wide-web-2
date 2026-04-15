"""
data/types.py

Core data classes for DynaWeb.

WebInteraction is the top-level object that flows through the whole pipeline.
The conversion chain is:

  raw dict (JSONL)
      → WebInteraction.from_raw_data()
      → used by NNetNavDataset
      → consumed by rollout engine (obs = interaction.state_changes[t].pre_state)
      → consumed by GSPO (reward = interaction.reward)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional
from xml.dom import minidom


# ---------------------------------------------------------------------------
# ActionType
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    CLICK = "CLICK"
    SELECT = "SELECT"
    TYPE = "TYPE"
    GO_BACK = "GO_BACK"
    SCROLL = "SCROLL"
    STOP = "STOP"

    @classmethod
    def from_string(cls, s: str) -> "ActionType":
        """
        Parse action type from raw string.
        Accepts both 'CLICK' and 'click [7]' style strings.
        """
        s = s.strip().upper()
        for member in cls:
            if s.startswith(member.value):
                return member
        # Fallback: try to infer from NNetNav/WebArena action format
        lowered = s.lower()
        if lowered.startswith("click"):
            return cls.CLICK
        if lowered.startswith("type"):
            return cls.TYPE
        if lowered.startswith("select"):
            return cls.SELECT
        if lowered.startswith("go_back"):
            return cls.GO_BACK
        if lowered.startswith("scroll"):
            return cls.SCROLL
        if lowered.startswith("stop"):
            return cls.STOP
        return cls.CLICK  # default fallback


# ---------------------------------------------------------------------------
# WebState
# ---------------------------------------------------------------------------

@dataclass
class WebState:
    """
    A single state of the website at a point in time.
    dom_tree is the primary representation; acc_tree is the text form
    used as input to the LLM (accessibility tree string).
    """
    dom_tree: Optional[minidom.Document] = None
    acc_tree: str = ""          # text accessibility tree — model input
    url: str = ""

    def dom_to_string(self) -> str:
        """Serialize DOM to XML string for diffing."""
        if self.dom_tree is not None:
            return self.dom_tree.toxml()
        return self.acc_tree   # fallback to acc_tree if no DOM

    def to_model_input(self) -> str:
        """
        Return the string representation fed into the LLM.
        URL header + accessibility tree text.
        """
        parts = []
        if self.url:
            parts.append(f"URL: {self.url}")
        parts.append(self.acc_tree)
        return "\n".join(parts)

    @classmethod
    def from_acc_tree(cls, acc_tree: str, url: str = "") -> "WebState":
        """Build a WebState directly from an accessibility tree string (no DOM)."""
        return cls(dom_tree=None, acc_tree=acc_tree, url=url)

    @classmethod
    def from_dom_string(cls, dom_string: str, url: str = "") -> "WebState":
        """Build a WebState by parsing a raw XML/DOM string."""
        try:
            dom = minidom.parseString(dom_string)
        except Exception:
            dom = None
        # Extract text content as acc_tree fallback
        acc = dom_string if dom is None else _dom_to_acc_tree(dom)
        return cls(dom_tree=dom, acc_tree=acc, url=url)


def _dom_to_acc_tree(dom: minidom.Document) -> str:
    """
    Minimal DOM → accessibility tree text conversion.
    Walks interactive elements and formats them as [type id=N 'text'].
    """
    lines = []
    _id_counter = [0]

    INTERACTIVE_TAGS = {"button", "input", "a", "select", "textarea", "h1", "h2", "h3", "p"}

    def walk(node, depth=0):
        if node.nodeType == minidom.Node.ELEMENT_NODE:
            tag = node.tagName.lower()
            if tag in INTERACTIVE_TAGS:
                _id_counter[0] += 1
                node_id = _id_counter[0]
                text = (node.getAttribute("aria-label") or
                        node.getAttribute("placeholder") or
                        node.getAttribute("value") or
                        (node.firstChild.nodeValue.strip()
                         if node.firstChild and node.firstChild.nodeType == minidom.Node.TEXT_NODE
                         else ""))
                text = text.strip().replace("'", "")[:60]
                lines.append(f"[{tag} id={node_id} '{text}']")
        for child in node.childNodes:
            walk(child, depth + 1)

    walk(dom.documentElement)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WebAction
# ---------------------------------------------------------------------------

@dataclass
class WebAction:
    """
    A single action by the agent.

    target_id:   integer element id (from accessibility tree)
    target_elem: actual DOM element (optional, only available during live eval)
    value:       string value for TYPE actions
    raw:         original action string, e.g. "type [15] [hello] [1]"
    """
    action_type: ActionType
    target_id: int = -1
    target_elem: Optional[Any] = None   # minidom.Element when available
    value: str = ""
    raw: str = ""

    def to_string(self) -> str:
        """Serialize back to WebArena action string format."""
        if self.action_type == ActionType.CLICK:
            return f"click [{self.target_id}]"
        if self.action_type == ActionType.TYPE:
            return f"type [{self.target_id}] [{self.value}] [1]"
        if self.action_type == ActionType.SELECT:
            return f"select [{self.target_id}] [{self.value}]"
        if self.action_type == ActionType.GO_BACK:
            return "go_back"
        if self.action_type == ActionType.SCROLL:
            return f"scroll [{self.value}]"
        if self.action_type == ActionType.STOP:
            return f"stop [{self.value}]"
        return self.raw

    @classmethod
    def from_string(cls, action_str: str) -> "WebAction":
        """
        Parse a WebArena/NNetNav action string into a WebAction.

        Handles:
            click [7]
            type [15] [search term] [1]
            type [15] [search term] [0]
            select [3] [option value]
            go_back
            scroll [down]
            stop [answer text]
        """
        s = action_str.strip()
        action_type = ActionType.from_string(s)

        if action_type == ActionType.CLICK:
            target_id = _extract_first_int(s)
            return cls(action_type=action_type, target_id=target_id, raw=s)

        if action_type == ActionType.TYPE:
            # type [id] [content] [press_enter]
            parts = _extract_bracketed_parts(s)
            target_id = int(parts[0]) if parts and parts[0].isdigit() else -1
            value = parts[1] if len(parts) > 1 else ""
            return cls(action_type=action_type, target_id=target_id, value=value, raw=s)

        if action_type == ActionType.SELECT:
            parts = _extract_bracketed_parts(s)
            target_id = int(parts[0]) if parts and parts[0].isdigit() else -1
            value = parts[1] if len(parts) > 1 else ""
            return cls(action_type=action_type, target_id=target_id, value=value, raw=s)

        if action_type == ActionType.GO_BACK:
            return cls(action_type=action_type, raw=s)

        if action_type == ActionType.SCROLL:
            parts = _extract_bracketed_parts(s)
            value = parts[0] if parts else "down"
            return cls(action_type=action_type, value=value, raw=s)

        if action_type == ActionType.STOP:
            parts = _extract_bracketed_parts(s)
            value = parts[0] if parts else ""
            return cls(action_type=action_type, value=value, raw=s)

        return cls(action_type=action_type, raw=s)

    @property
    def is_terminal(self) -> bool:
        return self.action_type == ActionType.STOP


# ---------------------------------------------------------------------------
# StateChange
# ---------------------------------------------------------------------------

@dataclass
class StateChange:
    """
    Records the result/change after an agent applied an action.

    pre_state  → action → post_state

    This is the atomic unit used to:
      - train the world model (predict post_state from pre_state + action)
      - build trajectories for GSPO (sequence of StateChanges = one episode)
    """
    pre_state: WebState
    action: WebAction
    post_state: WebState

    def dom_change(self) -> bool:
        """True if the DOM changed as a result of this action."""
        return self.pre_state.dom_to_string() != self.post_state.dom_to_string()

    def to_wm_training_dict(self) -> dict:
        """
        Serialize to dict suitable for world model training.
        Keys match format_world_model_prompt() inputs.
        """
        return {
            "pre_obs": self.pre_state.to_model_input(),
            "action": self.action.to_string(),
            "post_obs": self.post_state.to_model_input(),
            "dom_changed": self.dom_change(),
        }


# ---------------------------------------------------------------------------
# WebInteraction
# ---------------------------------------------------------------------------

@dataclass
class WebInteraction:
    """
    A complete episode: task + website + sequence of state changes + reward.

    This is the top-level object that:
      - NNetNavDataset yields
      - TrajectoryMixer consumes (for real expert trajectories)
      - DreamingRolloutEngine produces (imagined trajectories)
      - GSPO trainer trains on

    state_changes[t].pre_state  = observation at step t
    state_changes[t].action     = action taken at step t
    state_changes[t].post_state = observation at step t+1
    final_state                 = last observation (terminal state)
    """
    task: str
    website: str
    state_changes: List[StateChange]
    reward: float
    final_state: WebState
    is_real: bool = True          # False for imagined (dreamed) trajectories

    @classmethod
    def from_raw_data(cls, raw: dict, reward: float = 1.0) -> "WebInteraction":
        """
        Build a WebInteraction from a raw JSONL record.

        Expected raw format (NNetNav-compatible):
        {
          "task": "...",
          "website": "...",          # optional, defaults to ""
          "actions": [
            {
              "action": "click [7]",
              "pre_obs": "[button id=7 'Submit']...",
              "post_obs": "[heading id=10 'Result']...",
              "url": "https://example.com"   # optional
            },
            ...
          ]
        }

        Also accepts the simpler format:
        {
          "task": "...",
          "trajectory": [
            {"obs": "...", "action": "..."},
            ...
          ],
          "reward": 1.0
        }
        """
        task = raw.get("task", "")
        website = raw.get("website", "")

        state_changes = []

        # --- Format A: "actions" list with pre/post obs ---
        if "actions" in raw:
            for step in raw["actions"]:
                pre_state = WebState.from_acc_tree(
                    acc_tree=step.get("pre_obs", ""),
                    url=step.get("url", ""),
                )
                post_state = WebState.from_acc_tree(
                    acc_tree=step.get("post_obs", ""),
                    url=step.get("next_url", step.get("url", "")),
                )
                action = WebAction.from_string(step.get("action", "stop []"))
                state_changes.append(StateChange(
                    pre_state=pre_state,
                    action=action,
                    post_state=post_state,
                ))

        # --- Format B: "trajectory" list with obs+action pairs ---
        elif "trajectory" in raw:
            traj = raw["trajectory"]
            for i, step in enumerate(traj):
                pre_obs = step.get("obs", "")
                action_str = step.get("action", "stop []")
                # post_obs is next step's obs, or same if last
                if i + 1 < len(traj):
                    post_obs = traj[i + 1].get("obs", pre_obs)
                else:
                    post_obs = pre_obs
                state_changes.append(StateChange(
                    pre_state=WebState.from_acc_tree(pre_obs),
                    action=WebAction.from_string(action_str),
                    post_state=WebState.from_acc_tree(post_obs),
                ))
            reward = float(raw.get("reward", reward))

        final_state = (
            state_changes[-1].post_state
            if state_changes
            else WebState.from_acc_tree("")
        )

        return cls(
            task=task,
            website=website,
            state_changes=state_changes,
            reward=reward,
            final_state=final_state,
            is_real=True,
        )

    # ------------------------------------------------------------------
    # Convenience accessors used throughout the pipeline
    # ------------------------------------------------------------------

    def obs_list(self) -> List[str]:
        """All observations as strings. Length = len(state_changes) + 1."""
        if not self.state_changes:
            return []
        obs = [sc.pre_state.to_model_input() for sc in self.state_changes]
        obs.append(self.final_state.to_model_input())
        return obs

    def action_list(self) -> List[str]:
        """All actions as strings. Length = len(state_changes)."""
        return [sc.action.to_string() for sc in self.state_changes]

    def action_objects(self) -> List[WebAction]:
        return [sc.action for sc in self.state_changes]

    def num_steps(self) -> int:
        return len(self.state_changes)

    def was_successful(self) -> bool:
        return self.reward >= 1.0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_first_int(s: str) -> int:
    """Extract first integer from a string like 'click [7]'."""
    import re
    m = re.search(r"\[(\d+)\]", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1


def _extract_bracketed_parts(s: str) -> List[str]:
    """
    Extract all [...] bracketed contents from string.
    'type [15] [hello world] [1]' → ['15', 'hello world', '1']
    """
    import re
    return re.findall(r"\[([^\]]*)\]", s)