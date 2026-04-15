"""
utils/prompt.py

Prompt formatting for agent and world model.
All functions accept data.types objects directly.
"""

from typing import List, Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# System prompts (Appendix A.1 and A.2 of the paper)
# ---------------------------------------------------------------------------

WEBARENA_SYSTEM_PROMPT = """\
You are an AI assistant performing tasks on a web browser. \
You will be provided with a task objective, the current web page observations, \
and interaction history. You need to issue an action for this step.

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
- stop [answer]            — task complete. E.g., stop [Found item at $9.99]\
"""

WM_SYSTEM_PROMPT = """\
You are an intelligent agent that predicts the next web page state \
from a given current action in a web environment.

Given the user objective, current accessibility tree, and action taken, \
first predict what changes occurred, then output the full next accessibility tree.

Generate your answer in exactly this format:
[Web state changes]
<describe what elements were added, removed, or modified>

[Next page accessibility tree]
<full accessibility tree of the next page state>\
"""


# ---------------------------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------------------------

def format_agent_prompt(
    task: str,
    obs_history: List[str],
    action_history: List[str],
) -> str:
    """
    Build agent prompt from task + observation/action history.

    Input:
        task:           str           — natural language task
        obs_history:    List[str]     — observations, most recent last
        action_history: List[str]     — actions taken so far (len = len(obs_history)-1)
    Output:
        str — full prompt string
    """
    lines = [WEBARENA_SYSTEM_PROMPT, "", f"TASK: {task}", ""]

    if len(obs_history) > 1:
        lines.append("INTERACTION HISTORY:")
        for i, (obs, act) in enumerate(zip(obs_history[:-1], action_history)):
            lines.append(f"Step {i+1} observation: {obs[:200].strip()}...")
            lines.append(f"Step {i+1} action: {act}")
        lines.append("")

    lines.append("CURRENT PAGE:")
    lines.append(obs_history[-1])
    lines.append("")
    lines.append("Provide REASON then ACTION:")

    return "\n".join(lines)


def format_agent_prompt_from_interaction(interaction) -> str:
    """
    Convenience: build agent prompt directly from a WebInteraction.
    Uses the interaction's obs_list() and action_list().
    """
    return format_agent_prompt(
        task=interaction.task,
        obs_history=interaction.obs_list(),
        action_history=interaction.action_list(),
    )


# ---------------------------------------------------------------------------
# World model prompt
# ---------------------------------------------------------------------------

def format_world_model_prompt(
    task: str,
    current_obs: str,
    current_action: str,
    prev_action: Optional[str] = None,
    url: str = "",
) -> str:
    """
    Build world model prompt for next-state prediction.

    Input:
        task:           str           — natural language task
        current_obs:    str           — current accessibility tree text
        current_action: str           — action just taken
        prev_action:    Optional[str] — previous action (for context)
        url:            str           — current page URL
    Output:
        str — full WM prompt string
    """
    lines = [WM_SYSTEM_PROMPT, ""]
    lines.append(f"User objective: {task}")
    if url:
        lines.append(f"Current URL: {url}")
    if prev_action:
        lines.append(f"Previous action: {prev_action}")
    lines.append(f"Current action: {current_action}")
    lines.append("")
    lines.append("Current accessibility tree:")
    lines.append(current_obs)
    lines.append("")
    lines.append("Predict the state changes and next accessibility tree:")

    return "\n".join(lines)


def format_wm_prompt_from_state_change(sc) -> str:
    """
    Convenience: build WM prompt directly from a StateChange object.
    Used during world model training.
    """
    return format_world_model_prompt(
        task="",
        current_obs=sc.pre_state.to_model_input(),
        current_action=sc.action.to_string(),
        url=sc.pre_state.url,
    )


# ---------------------------------------------------------------------------
# Reward prompt
# ---------------------------------------------------------------------------

def format_reward_prompt(
    task: str,
    obs_list: List[str],
    action_list: List[str],
) -> str:
    """
    Build self-assessment prompt for reward computation.

    Input:
        task:        str        — task description
        obs_list:    List[str]  — observations at each step
        action_list: List[str]  — actions taken
    Output:
        str — prompt asking model to judge task completion
    """
    lines = [
        "You are evaluating whether a web agent successfully completed a task.",
        "",
        f"Task: {task}",
        "",
        "Agent actions:",
    ]
    for i, action in enumerate(action_list):
        lines.append(f"  Step {i+1}: {action}")

    lines += [
        "",
        "Final page state (truncated):",
        obs_list[-1][:400] if obs_list else "(no observations)",
        "",
        "Did the agent successfully complete the task?",
        "Answer YES or NO on the first line, then give a one-sentence reason.",
        "Answer:",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def parse_agent_output(raw_output: str) -> Dict[str, str]:
    """
    Extract REASON and ACTION from agent's raw generated text.

    Input:  raw_output: str
    Output: {"reason": str, "action": str}
    """
    result = {"reason": "", "action": "stop [parse_error]"}

    if "REASON:" in raw_output:
        after_reason = raw_output.split("REASON:", 1)[1]
        if "ACTION:" in after_reason:
            result["reason"] = after_reason.split("ACTION:", 1)[0].strip()
        else:
            result["reason"] = after_reason.strip()[:300]

    if "ACTION:" in raw_output:
        after_action = raw_output.split("ACTION:", 1)[1].strip()
        for line in after_action.split("\n"):
            line = line.strip()
            if line:
                result["action"] = line
                break

    return result


def parse_wm_output(raw_output: str) -> Tuple[str, str]:
    """
    Split world model output into (state_changes, next_acc_tree).

    Input:  raw_output: str
    Output: (state_changes: str, next_obs_tree: str)
    """
    if "[Web state changes]" in raw_output and "[Next page accessibility tree]" in raw_output:
        after_changes = raw_output.split("[Web state changes]", 1)[1]
        parts = after_changes.split("[Next page accessibility tree]", 1)
        state_changes = parts[0].strip()
        next_tree = parts[1].strip() if len(parts) > 1 else ""
        return state_changes, next_tree

    # Fallback: treat full output as the next tree
    return "", raw_output.strip()


def parse_reward_output(raw_output: str) -> float:
    """
    Parse YES/NO reward from assessment output.

    Input:  raw_output: str
    Output: 1.0 for YES, 0.0 for NO
    """
    first_line = raw_output.strip().split("\n")[0].upper()
    if first_line.startswith("YES"):
        return 1.0
    return 0.0