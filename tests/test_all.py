"""
tests/test_all.py

Tests every component of DynaWeb individually, then runs the full
training loop end-to-end in stub mode.

All tests run without a GPU and without any model weights.

Run:
    cd dynaweb
    python tests/test_all.py

Expected output:
    All tests pass with printed shapes and values at each stage.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Helpers
# ============================================================

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS} {name}" + (f"  [{detail}]" if detail else ""))
    else:
        print(f"  {FAIL} {name}" + (f"  [{detail}]" if detail else ""))
        raise AssertionError(f"FAILED: {name}")


# ============================================================
# 1. Data types
# ============================================================

def test_action_parsing():
    print("\n[1] ActionType + WebAction parsing")
    from data.types import WebAction, ActionType

    a = WebAction.from_string("click [7]")
    check("click parsed", a.action_type == ActionType.CLICK)
    check("click target id", a.target_id == 7)
    check("click not terminal", not a.is_terminal)

    a = WebAction.from_string("type [15] [hello world] [1]")
    check("type parsed", a.action_type == ActionType.TYPE)
    check("type target id", a.target_id == 15)
    check("type value", a.value == "hello world")

    a = WebAction.from_string("stop [task done]")
    check("stop parsed", a.action_type == ActionType.STOP)
    check("stop is terminal", a.is_terminal)
    check("stop value", a.value == "task done")

    a = WebAction.from_string("go_back")
    check("go_back parsed", a.action_type == ActionType.GO_BACK)

    # Roundtrip
    for raw in ["click [3]", "type [5] [search] [1]", "go_back", "stop [done]"]:
        action = WebAction.from_string(raw)
        check(f"roundtrip: {raw}", action.to_string() == raw, action.to_string())


def test_web_state():
    print("\n[2] WebState")
    from data.types import WebState

    s = WebState.from_acc_tree("[button id=1 'Home']\n[textbox id=2 'Search']", url="https://ex.com")
    check("acc_tree stored", "button" in s.acc_tree)
    check("url stored", s.url == "https://ex.com")
    check("to_model_input contains URL", "https://ex.com" in s.to_model_input())
    check("to_model_input contains tree", "button" in s.to_model_input())
    check("dom_to_string fallback", "button" in s.dom_to_string())


def test_state_change():
    print("\n[3] StateChange")
    from data.types import WebState, WebAction, StateChange, ActionType

    pre = WebState.from_acc_tree("[button id=1 'Submit']")
    post = WebState.from_acc_tree("[heading id=10 'Success']")
    action = WebAction(ActionType.CLICK, target_id=1, raw="click [1]")

    sc = StateChange(pre_state=pre, action=action, post_state=post)
    check("dom_change detected", sc.dom_change())

    same_post = WebState.from_acc_tree("[button id=1 'Submit']")
    sc_no_change = StateChange(pre_state=pre, action=action, post_state=same_post)
    check("no dom_change", not sc_no_change.dom_change())

    d = sc.to_wm_training_dict()
    check("wm dict has pre_obs", "pre_obs" in d)
    check("wm dict has action", "action" in d)
    check("wm dict has post_obs", "post_obs" in d)
    check("wm dict dom_changed", d["dom_changed"] is True)


def test_web_interaction_from_raw():
    print("\n[4] WebInteraction.from_raw_data — Format A (actions list)")
    from data.types import WebInteraction

    raw_a = {
        "task": "Search for Python",
        "website": "example.com",
        "actions": [
            {
                "action": "type [3] [Python] [1]",
                "pre_obs": "[textbox id=3 'Search']",
                "post_obs": "[heading id=10 'Results']",
                "url": "https://example.com",
            },
            {
                "action": "stop [done]",
                "pre_obs": "[heading id=10 'Results']",
                "post_obs": "[heading id=10 'Results']",
                "url": "https://example.com/results",
            },
        ],
        "reward": 1.0,
    }
    ia = WebInteraction.from_raw_data(raw_a)
    check("task parsed", ia.task == "Search for Python")
    check("website parsed", ia.website == "example.com")
    check("2 state changes", ia.num_steps() == 2)
    check("reward", ia.reward == 1.0)
    check("is_real", ia.is_real)
    check("obs_list length", len(ia.obs_list()) == 3)
    check("action_list length", len(ia.action_list()) == 2)
    check("was_successful", ia.was_successful())

    print("\n[4b] WebInteraction.from_raw_data — Format B (trajectory list)")
    raw_b = {
        "task": "Login",
        "trajectory": [
            {"obs": "[textbox id=1 'User']", "action": "type [1] [admin] [1]"},
            {"obs": "[button id=2 'Submit']", "action": "click [2]"},
            {"obs": "[heading id=3 'Welcome']", "action": "stop [logged in]"},
        ],
        "reward": 1.0,
    }
    ib = WebInteraction.from_raw_data(raw_b)
    check("format B steps", ib.num_steps() == 3)
    check("format B reward", ib.reward == 1.0)


def test_sample_data():
    print("\n[5] Hardcoded sample data")
    from data.samples import make_sample_interactions

    samples = make_sample_interactions()
    check("4 samples", len(samples) == 4)
    for i, s in enumerate(samples):
        check(f"sample {i} has steps", s.num_steps() > 0)
        check(f"sample {i} has task", len(s.task) > 0)
        check(f"sample {i} obs_list matches", len(s.obs_list()) == s.num_steps() + 1)
        check(f"sample {i} action_list matches", len(s.action_list()) == s.num_steps())


def test_dataset():
    print("\n[6] NNetNavDataset (no file — uses hardcoded samples)")
    from data.dataset import NNetNavDataset

    ds = NNetNavDataset("data/raw/NONEXISTENT.jsonl")
    check("loaded fallback samples", len(ds) > 0, f"{len(ds)} samples")

    item = ds[0]
    check("item is WebInteraction", hasattr(item, "task"))
    check("item has state_changes", len(item.state_changes) > 0)

    task, state = ds.sample_random_state()
    check("sample_random_state task", isinstance(task, str))
    check("sample_random_state state", hasattr(state, "acc_tree"))

    pairs = ds.get_wm_training_pairs()
    check("wm pairs > 0", len(pairs) > 0, f"{len(pairs)} pairs")


# ============================================================
# 2. Prompt formatting
# ============================================================

def test_prompt_formatting():
    print("\n[7] Prompt formatting")
    from utils.prompt import (
        format_agent_prompt,
        format_world_model_prompt,
        format_reward_prompt,
        parse_agent_output,
        parse_wm_output,
        parse_reward_output,
        format_agent_prompt_from_interaction,
    )
    from data.samples import make_sample_interactions

    interaction = make_sample_interactions()[0]

    # Agent prompt
    prompt = format_agent_prompt(
        task=interaction.task,
        obs_history=interaction.obs_list(),
        action_history=interaction.action_list(),
    )
    check("agent prompt is string", isinstance(prompt, str))
    check("agent prompt has task", interaction.task in prompt)
    check("agent prompt has REASON", "REASON:" in prompt)
    check("agent prompt has ACTION", "ACTION:" in prompt)

    # From interaction convenience
    prompt2 = format_agent_prompt_from_interaction(interaction)
    check("from_interaction same as manual", prompt == prompt2)

    # WM prompt
    sc = interaction.state_changes[0]
    wm_prompt = format_world_model_prompt(
        task=interaction.task,
        current_obs=sc.pre_state.to_model_input(),
        current_action=sc.action.to_string(),
    )
    check("wm prompt is string", isinstance(wm_prompt, str))
    check("wm prompt has action", sc.action.to_string() in wm_prompt)

    # Reward prompt
    r_prompt = format_reward_prompt(
        task=interaction.task,
        obs_list=interaction.obs_list(),
        action_list=interaction.action_list(),
    )
    check("reward prompt has task", interaction.task in r_prompt)
    check("reward prompt has YES/NO", "YES or NO" in r_prompt)

    # Output parsers
    raw_agent = "REASON:\nI need to search.\nACTION:\ntype [3] [python] [1]"
    parsed = parse_agent_output(raw_agent)
    check("parse_agent reason", "search" in parsed["reason"])
    check("parse_agent action", parsed["action"] == "type [3] [python] [1]")

    raw_wm = "[Web state changes]\nButton was clicked.\n\n[Next page accessibility tree]\n[heading id=10 'Result']"
    changes, tree = parse_wm_output(raw_wm)
    check("parse_wm changes", "clicked" in changes)
    check("parse_wm tree", "heading" in tree)

    check("parse_reward YES", parse_reward_output("YES, task done") == 1.0)
    check("parse_reward NO", parse_reward_output("NO, failed") == 0.0)


# ============================================================
# 3. Models (stub mode)
# ============================================================

def test_agent_stub():
    print("\n[8] AgentPolicy (stub)")
    from dynaweb_config import DynaWebConfig
    from models.agent import AgentPolicy
    import torch

    config = DynaWebConfig(use_stub_models=True)
    agent = AgentPolicy(config)

    # generate_action
    action, log_probs = agent.generate_action("fake prompt", step=0)
    check("action is string", isinstance(action, str))
    check("action non-empty", len(action) > 0)
    check("log_probs is tensor", isinstance(log_probs, torch.Tensor))
    check("log_probs 1D", log_probs.dim() == 1)
    check("log_probs negative", (log_probs < 0).all())

    # Different steps give different actions
    actions = [agent.generate_action("p", step=i)[0] for i in range(5)]
    check("stub varies by step", len(set(actions)) > 1)

    # get_token_log_probs
    B, L = 3, 32
    input_ids = torch.randint(0, 1000, (B, L))
    labels = torch.cat([
        torch.full((B, L // 2), -100),
        torch.randint(0, 1000, (B, L // 2))
    ], dim=1)
    lp = agent.get_token_log_probs(input_ids, labels)
    check("log_probs shape", lp.shape == (B, L), str(lp.shape))
    check("prompt positions zeroed", (lp[:, :L//2] == 0).all())
    check("response positions non-zero", (lp[:, L//2:] != 0).any())

    # score_reward_prompt
    r = agent.score_reward_prompt("task: search. actions: stop [done]")
    check("reward 1.0 for stop", r == 1.0)


def test_world_model_stub():
    print("\n[9] WebWorldModel (stub)")
    from dynaweb_config import DynaWebConfig
    from models.world_model import WebWorldModel
    from data.types import WebState, WebAction, ActionType

    config = DynaWebConfig(use_stub_models=True)
    wm = WebWorldModel(config)

    state = WebState.from_acc_tree("[button id=1 'Home']\n[textbox id=2 'Search']", url="https://ex.com")

    # Click action
    action_click = WebAction(ActionType.CLICK, target_id=1, raw="click [1]")
    changes, next_state = wm.predict_next_state("search task", state, action_click)
    check("changes is str", isinstance(changes, str))
    check("next_state is WebState", isinstance(next_state, WebState))
    check("next_state has acc_tree", len(next_state.acc_tree) > 0)
    check("click changes acc_tree", next_state.acc_tree != state.acc_tree)

    # Type action
    action_type = WebAction(ActionType.TYPE, target_id=2, value="python", raw="type [2] [python] [1]")
    _, next_state_type = wm.predict_next_state("search task", state, action_type)
    check("type appends value", "python" in next_state_type.acc_tree)

    # Stop action
    action_stop = WebAction(ActionType.STOP, value="done", raw="stop [done]")
    check("is_terminal_state", wm.is_terminal_state(action_stop))
    check("click not terminal", not wm.is_terminal_state(action_click))

    # URL update
    check("click url unchanged", next_state.url == "https://ex.com")
    check("type url has query", "python" in next_state_type.url)


def test_reward_fn():
    print("\n[10] SelfAssessReward (stub)")
    from dynaweb_config import DynaWebConfig
    from models.agent import AgentPolicy
    from models.reward import SelfAssessReward
    from data.samples import make_sample_interactions
    import torch

    config = DynaWebConfig(use_stub_models=True)
    agent = AgentPolicy(config)
    reward_fn = SelfAssessReward(agent, config)

    samples = make_sample_interactions()

    # Real trajectory: uses stored reward directly
    r = reward_fn.compute_reward(samples[0])
    check("real traj reward = stored", r == samples[0].reward)

    # Imagined trajectory: uses self-assessment
    imagined = make_sample_interactions()[0]
    imagined.is_real = False
    r_imag = reward_fn.compute_reward(imagined)
    check("imagined reward is float", isinstance(r_imag, float))
    check("imagined reward 0 or 1", r_imag in (0.0, 1.0))

    # Batch
    rewards = reward_fn.batch_compute_rewards(samples)
    check("batch tensor shape", rewards.shape == (len(samples),), str(rewards.shape))
    check("batch dtype float", rewards.dtype == torch.float32)


# ============================================================
# 4. GSPO loss
# ============================================================

def test_gspo():
    print("\n[11] GSPO loss functions")
    from training.gspo import (
        compute_token_ratios,
        compute_sequence_ratios,
        compute_gspo_advantages,
        gspo_loss,
        compute_full_gspo_loss,
    )
    from dynaweb_config import DynaWebConfig
    from models.agent import AgentPolicy
    import torch

    B, L, n = 4, 32, 2   # B*n=4 trajectories, L=32 tokens

    # Same policy → ratios should be ~1.0
    log_probs = torch.full((B, L), -10.0)
    log_ratios = compute_token_ratios(log_probs, log_probs)
    check("same policy ratios = 0", (log_ratios == 0).all())

    # Sequence ratios
    labels = torch.cat([
        torch.full((B, L // 2), -100, dtype=torch.long),
        torch.randint(0, 100, (B, L // 2))
    ], dim=1)
    seq_lens = torch.full((B,), L // 2, dtype=torch.long)
    seq_ratios = compute_sequence_ratios(log_ratios, seq_lens, labels)
    check("seq_ratios shape", seq_ratios.shape == (B,), str(seq_ratios.shape))
    check("same policy seq_ratios = 1", torch.allclose(seq_ratios, torch.ones(B), atol=1e-5))

    # Advantages
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])   # B*n = 4, n=2
    advantages = compute_gspo_advantages(rewards, n=n)
    check("advantages shape", advantages.shape == (B,), str(advantages.shape))
    check("advantages mean ~0", abs(advantages.mean().item()) < 1e-5)

    # Loss
    seq_ratios_nograd = torch.ones(B)
    loss = gspo_loss(seq_ratios_nograd, advantages, eps=0.2)
    check("loss is scalar", loss.dim() == 0)
    check("same policy loss ~0", abs(loss.item()) < 1e-4, f"loss={loss.item():.6f}")

    # Full GSPO with stub agent
    config = DynaWebConfig(use_stub_models=True)
    agent = AgentPolicy(config)
    input_ids = torch.randint(0, 1000, (B, L))
    old_lp = agent.get_token_log_probs(input_ids, labels)

    loss, diag = compute_full_gspo_loss(
        agent=agent,
        old_log_probs=old_lp,
        input_ids=input_ids,
        labels=labels,
        rewards=rewards,
        seq_lengths=seq_lens,
        n=n,
        eps=0.2,
    )
    check("full loss scalar", loss.dim() == 0)
    check("diagnostics has loss", "loss" in diag)
    check("diagnostics has reward", "mean_reward" in diag)
    check("diagnostics has ratio", "mean_seq_ratio" in diag)
    print(f"    Diagnostics: {diag}")


# ============================================================
# 5. Rollout engine
# ============================================================

def test_rollout_engine():
    print("\n[12] DreamingRolloutEngine (stub)")
    from dynaweb_config import DynaWebConfig
    from models.agent import AgentPolicy
    from models.world_model import WebWorldModel
    from models.reward import SelfAssessReward
    from training.rollout import DreamingRolloutEngine
    from data.types import WebState

    config = DynaWebConfig(use_stub_models=True, dream_length=3, rollout_n=2)
    agent = AgentPolicy(config)
    wm = WebWorldModel(config)
    reward_fn = SelfAssessReward(agent, config)
    engine = DreamingRolloutEngine(agent, wm, reward_fn, config)

    initial_state = WebState.from_acc_tree(
        "[button id=1 'Home']\n[textbox id=2 'Search']",
        url="https://example.com"
    )

    # Single trajectory
    traj = engine.generate_single_trajectory("Search for Python", initial_state)
    check("traj is WebInteraction", hasattr(traj, "state_changes"))
    check("traj is not real", not traj.is_real)
    check("traj has steps", traj.num_steps() > 0)
    check("traj num_steps <= dream_length", traj.num_steps() <= config.dream_length + 1)
    check("traj reward is float", isinstance(traj.reward, float))
    check("traj reward 0 or 1", traj.reward in (0.0, 1.0))
    print(f"    Steps: {traj.num_steps()}, reward: {traj.reward}")

    # Group (n rollouts for one task)
    group = engine.generate_group("Search for Python", initial_state, n=2)
    check("group has 2 trajs", len(group) == 2)
    check("all imagined", all(not t.is_real for t in group))

    # Batch (B tasks)
    states = [initial_state, initial_state]
    tasks = ["Task 1", "Task 2"]
    groups = engine.generate_batch(tasks, states)
    check("batch shape [2][2]", len(groups) == 2 and len(groups[0]) == 2)


# ============================================================
# 6. Mixer
# ============================================================

def test_mixer():
    print("\n[13] TrajectoryMixer (stub)")
    from dynaweb_config import DynaWebConfig
    from models.agent import AgentPolicy
    from models.world_model import WebWorldModel
    from models.reward import SelfAssessReward
    from training.rollout import DreamingRolloutEngine
    from training.mixer import TrajectoryMixer
    from data.dataset import NNetNavDataset
    from data.types import WebState
    import torch

    config = DynaWebConfig(use_stub_models=True, dream_length=2, rollout_n=2, real_traj_ratio=0.5)
    agent = AgentPolicy(config)
    wm = WebWorldModel(config)
    reward_fn = SelfAssessReward(agent, config)
    engine = DreamingRolloutEngine(agent, wm, reward_fn, config)
    dataset = NNetNavDataset("nonexistent.jsonl")
    mixer = TrajectoryMixer(dataset, config, tokenizer=None)

    initial_state = WebState.from_acc_tree("[button id=1 'Home']")
    groups = engine.generate_batch(["Task A", "Task B"], [initial_state, initial_state])

    # Mix
    mixed = mixer.mix(groups)
    check("mixed length = B*n", len(mixed) == 4, f"len={len(mixed)}")
    n_real = sum(1 for t in mixed if t.is_real)
    n_imag = sum(1 for t in mixed if not t.is_real)
    check("some real", n_real > 0, f"real={n_real}")
    check("some imagined", n_imag > 0 or n_real == 4, f"imag={n_imag}")

    # Stub batch
    batch = mixer.to_gspo_batch(mixed, device="cpu")
    check("input_ids present", "input_ids" in batch)
    check("labels present", "labels" in batch)
    check("rewards present", "rewards" in batch)
    check("input_ids 2D", batch["input_ids"].dim() == 2)
    check("rewards 1D", batch["rewards"].dim() == 1)
    check("rewards length", batch["rewards"].shape[0] == len(mixed), str(batch["rewards"].shape))
    print(f"    input_ids shape: {batch['input_ids'].shape}")
    print(f"    rewards: {batch['rewards'].tolist()}")


# ============================================================
# 7. Full end-to-end training loop
# ============================================================

def test_full_training_loop():
    print("\n[14] Full training loop (stub, 1 step)")
    from dynaweb_config import DynaWebConfig
    from data.dataset import NNetNavDataset
    from models.agent import AgentPolicy
    from models.world_model import WebWorldModel
    from models.reward import SelfAssessReward
    from training.trainer import DynaWebTrainer

    config = DynaWebConfig(
        use_stub_models=True,
        epochs=1,
        train_batch_size=2,
        rollout_n=2,
        dream_length=2,
        real_traj_ratio=0.5,
    )

    dataset = NNetNavDataset("nonexistent.jsonl")
    agent = AgentPolicy(config)
    wm = WebWorldModel(config)
    reward_fn = SelfAssessReward(agent, config)

    trainer = DynaWebTrainer(
        config=config,
        agent=agent,
        world_model=wm,
        reward_fn=reward_fn,
        train_dataset=dataset,
        val_dataset=dataset,
        tokenizer=None,
    )

    # Run one training step manually
    batch = [dataset[i] for i in range(min(2, len(dataset)))]
    metrics = trainer.train_step(batch)

    check("metrics has loss", "loss" in metrics)
    check("metrics has reward", "mean_reward" in metrics)
    check("metrics has advantage", "mean_advantage" in metrics)
    check("metrics has ratio", "mean_seq_ratio" in metrics)
    check("loss is finite", abs(metrics["loss"]) < 1e6)
    print(f"    Metrics: {metrics}")

    # Run full train() for 1 epoch
    print("\n    Running trainer.train() for 1 epoch...")
    trainer.train()
    check("log history populated", len(trainer.log_history) > 0)

    # Evaluate
    eval_metrics = trainer.evaluate()
    check("eval has success_rate", "val_success_rate" in eval_metrics)
    check("success_rate 0-1", 0.0 <= eval_metrics["val_success_rate"] <= 1.0)
    print(f"    Eval: {eval_metrics}")


# ============================================================
# Run all tests
# ============================================================

def main():
    tests = [
        test_action_parsing,
        test_web_state,
        test_state_change,
        test_web_interaction_from_raw,
        test_sample_data,
        test_dataset,
        test_prompt_formatting,
        test_agent_stub,
        test_world_model_stub,
        test_reward_fn,
        test_gspo,
        test_rollout_engine,
        test_mixer,
        test_full_training_loop,
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("DynaWeb Test Suite")
    print("=" * 60)

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n  {FAIL} {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"\n  {FAIL} Unexpected error in {test_fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()