"""
normalize.py

Reads output.jsonl (NNetNav/openweb format) and writes train.jsonl.

Data shape reality check
------------------------
Each JSONL record is ONE step of an episode, not a full trajectory:

  {
    "dataset": "openweb_NNetNav_Bof3",
    "id": "openweb_2603",
    "task_name": "openweb_2603",
    "n_tokens": "5092",
    "messages": [
      {"role": "system",    "content": "<system prompt>"},
      {"role": "user",      "content": "OBSERVATION:\n<acc_tree>\nURL: ...\nOBJECTIVE: ...\nPREVIOUS ACTIONS:\n1: None\n..."},
      {"role": "assistant", "content": "Let's think ... ```type [89] [water chemistry] [1]```"}
    ],
    "output":  "<|start_header_id|>assistant<|end_header_id|>\n<assistant content>",
    "prompt":  "<|begin_of_text|><|start_header_id|>system<|end_header_id|>..."
  }

Because each record is a single step, we group records by task_name to rebuild
full trajectories. Each task_name group becomes one WebInteraction:
  - step 0 → StateChange(pre_state_0, action_0, post_state_0=pre_state_1)
  - step 1 → StateChange(pre_state_1, action_1, post_state_1=pre_state_2)
  - ...
  - last step: post_state = same as pre_state (no next obs available)

Output: train.jsonl with one line per WebInteraction (Format A of from_raw_data):
  {
    "task":    "...",
    "website": "...",
    "actions": [
      {"pre_obs": "...", "action": "...", "post_obs": "...", "url": "..."},
      ...
    ]
  }
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_step(record: dict) -> dict | None:
    """
    Extract (task, website, step_number, url, acc_tree, action_str)
    from one raw JSONL record. Returns None if the record is malformed.
    """
    try:
        messages = record.get("messages", [])
        # Expect system / user / assistant
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
        asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)
        if not user_msg or not asst_msg:
            return None

        # --- acc_tree: everything between "OBSERVATION:\n" and "\nURL:" ---
        obs_match = re.search(r"^OBSERVATION:\n(.*?)(?=\nURL:)", user_msg, re.DOTALL)
        acc_tree = obs_match.group(1).strip() if obs_match else ""

        # --- URL ---
        url_match = re.search(r"^URL: (.+)$", user_msg, re.MULTILINE)
        url = url_match.group(1).strip() if url_match else ""

        # --- OBJECTIVE (= task) ---
        obj_match = re.search(r"^OBJECTIVE: (.+)$", user_msg, re.MULTILINE)
        objective = obj_match.group(1).strip() if obj_match else ""

        # --- Step number from PREVIOUS ACTIONS ---
        # "PREVIOUS ACTIONS:\n1: None\n2: click [7]\n..."
        # The step index of THIS record = number of listed previous actions
        prev_match = re.search(r"^PREVIOUS ACTIONS:\n(.*)", user_msg, re.MULTILINE | re.DOTALL)
        prev_raw = prev_match.group(1).strip() if prev_match else ""
        prev_lines = [l for l in prev_raw.splitlines() if l.strip() and l.strip() != "None"]
        # Count non-None previous action lines to get step index
        step_index = len([l for l in prev_lines if not l.strip().endswith(": None")])

        # --- Action from assistant backticks ---
        action_match = re.search(r"```(.+?)```", asst_msg, re.DOTALL)
        action_str = action_match.group(1).strip() if action_match else ""
        if not action_str:
            return None

        # --- Website: infer from URL or dataset field ---
        task_name = record.get("task_name", "")
        dataset = record.get("dataset", "")
        try:
            from urllib.parse import urlparse
            website = urlparse(url).netloc.replace("www.", "")
        except Exception:
            website = dataset

        return {
            "task_name":   task_name,
            "task":        objective,
            "website":     website,
            "step_index":  step_index,
            "url":         url,
            "acc_tree":    acc_tree,
            "action_str":  action_str,
        }
    except Exception as e:
        print(f"  [WARN] Failed to parse record id={record.get('id')}: {e}", file=sys.stderr)
        return None


def group_into_trajectories(steps: list[dict]) -> dict[str, list[dict]]:
    """
    Group parsed steps by task_name, sort each group by step_index.
    Returns {task_name: [step_0, step_1, ...]}.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for step in steps:
        groups[step["task_name"]].append(step)
    # Sort each group by step_index
    for name in groups:
        groups[name].sort(key=lambda s: s["step_index"])
    return dict(groups)


def trajectory_to_interaction(task_name: str, steps: list[dict]) -> dict:
    """
    Convert an ordered list of steps into a WebInteraction-compatible dict
    (Format A: actions[] with pre_obs/post_obs pairs).

    post_obs for step i = pre_obs of step i+1.
    post_obs for the final step = same as pre_obs (no next observation).
    """
    actions = []
    for i, step in enumerate(steps):
        pre_obs  = step["acc_tree"]
        # post_obs = next step's acc_tree, or same obs if this is the last step
        post_obs = steps[i + 1]["acc_tree"] if i + 1 < len(steps) else step["acc_tree"]
        next_url = steps[i + 1]["url"]      if i + 1 < len(steps) else step["url"]

        actions.append({
            "pre_obs":  pre_obs,
            "action":   step["action_str"],
            "post_obs": post_obs,
            "url":      step["url"],
            "next_url": next_url,
        })

    # Reward heuristic: 1.0 if final action is a stop[], else 0.0
    final_action = steps[-1]["action_str"].lower().strip() if steps else ""
    reward = 1.0 if final_action.startswith("stop") else 0.0

    return {
        "task":    steps[0]["task"] if steps else "",
        "website": steps[0]["website"] if steps else "",
        "actions": actions,
        "reward":  reward,
    }


# ---------------------------------------------------------------------------
# Validation: call every WebInteraction method and assert invariants
# ---------------------------------------------------------------------------

def validate_interaction(raw: dict, idx: int) -> list[str]:
    """
    Build a WebInteraction from the normalized dict and exercise every method.
    Returns a list of error strings (empty = all good).
    """
    errors = []
    try:
        # Import from the working-directory types.py
        sys.path.insert(0, str(Path(__file__).parent))
        from datatypes import WebInteraction  # noqa: PLC0415

        wi = WebInteraction.from_raw_data(raw, reward=raw.get("reward", 0.0))

        # obs_list length must be len(state_changes) + 1
        obs = wi.obs_list()
        n = wi.num_steps()
        if len(obs) != n + 1 and n > 0:
            errors.append(f"obs_list length {len(obs)} != num_steps+1 {n+1}")

        # No empty obs strings
        for j, o in enumerate(obs):
            if not o.strip():
                errors.append(f"Empty obs at index {j}")

        # action_list length must equal num_steps
        acts = wi.action_list()
        if len(acts) != n:
            errors.append(f"action_list length {len(acts)} != num_steps {n}")

        # action_objects must be parseable
        objs = wi.action_objects()
        if len(objs) != n:
            errors.append(f"action_objects length {len(objs)} != num_steps {n}")

        # was_successful must not raise
        _ = wi.was_successful()

        # to_wm_training_dict on every StateChange
        for sc in wi.state_changes:
            d = sc.to_wm_training_dict()
            for key in ("pre_obs", "action", "post_obs", "dom_changed"):
                if key not in d:
                    errors.append(f"Missing key '{key}' in to_wm_training_dict")

    except Exception as e:
        errors.append(f"Exception during validation: {e}")

    return errors


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(interactions: list[dict], action_type_counts: dict[str, int]) -> None:
    n = len(interactions)
    if n == 0:
        print("No interactions to report.")
        return

    lengths = [len(i["actions"]) for i in interactions]
    avg_len = sum(lengths) / n
    successes = sum(1 for i in interactions if i.get("reward", 0) >= 1.0)

    print(f"\n{'='*50}")
    print(f"  DATASET STATISTICS")
    print(f"{'='*50}")
    print(f"  Total trajectories:  {n}")
    print(f"  Total steps:         {sum(lengths)}")
    print(f"  Avg steps/episode:   {avg_len:.2f}")
    print(f"  Min/Max steps:       {min(lengths)} / {max(lengths)}")
    print(f"  Success rate:        {successes}/{n} ({100*successes/n:.1f}%)")
    print(f"\n  Action type distribution:")
    total_acts = sum(action_type_counts.values())
    for atype, count in sorted(action_type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_acts if total_acts else 0
        print(f"    {atype:<20} {count:>5}  ({pct:.1f}%)")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    input_path  = Path("data/raw/train_wa.jsonl")
    output_path = Path("data/raw/clean_train_wa.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Read all raw records ---
    raw_records = []
    with open(input_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {lineno}: JSON parse error: {e}", file=sys.stderr)

    print(f"Read {len(raw_records)} raw records from {input_path.name}")

    # --- 2. Parse each record into a step dict ---
    steps = []
    failed = 0
    for record in raw_records:
        step = parse_step(record)
        if step:
            steps.append(step)
        else:
            failed += 1

    print(f"Parsed {len(steps)} valid steps ({failed} failed)")

    # --- 3. Group steps by task_name → trajectories ---
    trajectories = group_into_trajectories(steps)
    print(f"Grouped into {len(trajectories)} unique trajectories")

    # --- 4. Convert each trajectory to a WebInteraction dict ---
    interactions = []
    action_type_counts: dict[str, int] = defaultdict(int)

    for task_name, traj_steps in trajectories.items():
        interaction = trajectory_to_interaction(task_name, traj_steps)
        interactions.append(interaction)

        # Count action types
        for act in interaction["actions"]:
            action_str = act["action"].strip().lower()
            # Extract the verb (first word / bracket-prefix)
            parts = action_str.split("[")[0].strip().split()
            verb = parts[0] if parts else action_str or "unknown"
            action_type_counts[verb] += 1

    # --- 5. Validate every interaction ---
    print(f"\nValidating {len(interactions)} interactions...")
    total_errors = 0
    for i, interaction in enumerate(interactions):
        errors = validate_interaction(interaction, i)
        if errors:
            total_errors += len(errors)
            print(f"  [ERROR] interaction {i} ({interaction.get('task','')[:50]})")
            for e in errors:
                print(f"    - {e}")

    if total_errors == 0:
        print(f"  All interactions passed validation.")
    else:
        print(f"  {total_errors} total validation errors.")

    # --- 6. Write output ---
    with open(output_path, "w") as f:
        for interaction in interactions:
            f.write(json.dumps(interaction) + "\n")

    print(f"\nWrote {len(interactions)} interactions to {output_path}")

    # --- 7. Print statistics ---
    print_statistics(interactions, dict(action_type_counts))


if __name__ == "__main__":
    main()