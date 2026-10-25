"""
Phase 6: Experiment runner.

Runs repeated experiments comparing:
- Random policy
- PPO policy
- PPO + LLM brainstormer

Results are persisted in SQLite and aggregated after each run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from agent.train_ppo import build_env, build_model
from agent.random_policy import RandomPolicy
from agent.greedy_policy import CoverageGreedyPolicy
from agent.bandit_policy import UCB1BanditPolicy
from experiment.tracker import ExperimentTracker
from experiment.metrics import aggregate_metrics, compare_policies
from env.cross_api_env import DEFAULT_MAX_SLOTS
from executor.planted_bugs import classify_planted_bug


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOCK_SPEC = REPO_ROOT / "tests" / "fixtures" / "sample_api.yaml"
DEFAULT_DB = REPO_ROOT / "runs" / "phase6_experiments.db"

ALL_POLICIES = [
    "random",
    "random_budget_matched",
    "greedy",
    "bandit",
    "ppo",
    "ppo_llm",
    "cross_api_ppo",
]

EVALUATION_SEED_OFFSET = 10_000


def _evaluate_policy(
    env,
    predict,
    tracker: ExperimentTracker,
    exp_id: int,
    n_episodes: int,
    max_steps_per_episode: int,
    reset_seed_base: int,
    use_planted_bugs: bool,
    on_step=None,
) -> None:
    """Roll out `predict` for n_episodes and persist per-episode stats AND
    every anomaly the env reports.

    - Anomalies are written to the `anomalies` table at the step they occur
      (first_seen_step = 1-based request index within this evaluation run,
      so MIN(first_seen_step) is a true "requests to first bug").
    - With `use_planted_bugs`, an anomaly whose (operation, status) matches a
      planted bug is tagged is_planted_bug=1 / bug_name; every other anomaly
      is stored as is_planted_bug=0 (a false positive w.r.t. the planted set).
    - Coverage is read from `env.unwrapped`: gymnasium >=1.0 wrappers (e.g.
      ScenarioSeededEnv for ppo_llm) no longer forward attributes.
    """
    base_env = env.unwrapped
    global_step = 0
    experiment_seen_anomalies: set[tuple] = set()

    for ep_num in range(n_episodes):
        obs, info = env.reset(seed=reset_seed_base + ep_num)

        ep_reward = 0.0
        step_count = 0
        new_anomalies_this_ep = 0

        while step_count < max_steps_per_episode:
            action = predict(obs)

            obs, reward, done, truncated, info = env.step(action)

            if on_step is not None:
                on_step(action, reward)

            ep_reward += reward
            step_count += 1
            global_step += 1

            signature = info.get("anomaly_signature")
            if signature is not None:
                if signature not in experiment_seen_anomalies:
                    new_anomalies_this_ep += 1
                    experiment_seen_anomalies.add(signature)

                bug_key = (
                    classify_planted_bug(info["operation_id"], info["status_code"])
                    if use_planted_bugs else None
                )
                tracker.record_anomaly_at_step(
                    exp_id,
                    info["operation_id"],
                    info["status_code"],
                    signature,
                    episode=ep_num,
                    step=global_step,
                    is_planted_bug=int(bug_key is not None),
                    bug_name=bug_key,
                )

            if done or truncated:
                break

        tracker.record_episode(
            exp_id,
            ep_num,
            ep_reward,
            step_count,
            base_env.coverage.hits,
            base_env.coverage.anomaly_signatures,
            new_anomalies_this_ep,
        )


def _run_random_policy(
    spec_path: Path,
    n_episodes: int,
    max_steps_per_episode: int,
    use_planted_bugs: bool,
    trial_seed: int,
    db_path: Path,
    trial_id: int,
) -> int:
    """Evaluate Random policy using the paired evaluation seed schedule."""
    eval_seed_base = trial_seed + EVALUATION_SEED_OFFSET

    env = build_env(
        spec_path,
        use_mock=True,
        max_steps_per_episode=max_steps_per_episode,
        seed=eval_seed_base,
        enable_brainstormer=False,
        use_planted_bugs=use_planted_bugs,
    )

    tracker = ExperimentTracker(db_path)

    exp_id = tracker.start_experiment(
        name=f"random_trial_{trial_id}",
        policy="random",
        trial_id=trial_id,
        seed=trial_seed,
        max_steps_per_episode=max_steps_per_episode,
        timesteps_total=0,
        n_episodes=n_episodes,
    )

    policy = RandomPolicy(env.action_space.n, seed=trial_seed)

    _evaluate_policy(
        env,
        lambda obs: policy.predict(obs)[0],
        tracker,
        exp_id,
        n_episodes,
        max_steps_per_episode,
        reset_seed_base=eval_seed_base,
        use_planted_bugs=use_planted_bugs,
    )

    eval_requests = env.unwrapped.runner.request_count
    tracker.record_request_counts(
        exp_id,
        training_requests=0,
        prep_requests=0,
        eval_requests=eval_requests,
        total_requests=eval_requests,
        brainstorm_calls=0,
        seeded_episodes=0,
    )

    env.close()

    return exp_id


def _run_budget_matched_random_policy(
    spec_path: Path,
    n_episodes: int,
    max_steps_per_episode: int,
    timesteps_total: int,
    use_planted_bugs: bool,
    trial_seed: int,
    db_path: Path,
    trial_id: int,
) -> int:
    """Evaluate Random policy with request budget matched to PPO's total budget.

    Budget B_target = timesteps_total + (n_episodes * max_steps_per_episode).
    Runs random episodes under paired seeds until B_target requests are executed.
    """
    target_budget = (
        timesteps_total + (n_episodes * max_steps_per_episode)
        if timesteps_total > 0
        else n_episodes * max_steps_per_episode
    )
    eval_seed_base = trial_seed + EVALUATION_SEED_OFFSET

    env = build_env(
        spec_path,
        use_mock=True,
        max_steps_per_episode=max_steps_per_episode,
        seed=eval_seed_base,
        enable_brainstormer=False,
        use_planted_bugs=use_planted_bugs,
    )

    tracker = ExperimentTracker(db_path)

    est_episodes = max(
        n_episodes,
        (target_budget + max_steps_per_episode - 1) // max_steps_per_episode,
    )

    exp_id = tracker.start_experiment(
        name=f"random_budget_matched_trial_{trial_id}",
        policy="random_budget_matched",
        trial_id=trial_id,
        seed=trial_seed,
        max_steps_per_episode=max_steps_per_episode,
        timesteps_total=0,
        n_episodes=est_episodes,
    )

    policy = RandomPolicy(env.action_space.n, seed=trial_seed)
    base_env = env.unwrapped
    global_step = 0
    experiment_seen_anomalies: set[tuple] = set()
    ep_num = 0

    while base_env.runner.request_count < target_budget:
        obs, info = env.reset(seed=eval_seed_base + ep_num)

        ep_reward = 0.0
        step_count = 0
        new_anomalies_this_ep = 0

        while step_count < max_steps_per_episode and base_env.runner.request_count < target_budget:
            action = policy.predict(obs)[0]

            obs, reward, done, truncated, info = env.step(action)

            ep_reward += reward
            step_count += 1
            global_step += 1

            signature = info.get("anomaly_signature")
            if signature is not None:
                if signature not in experiment_seen_anomalies:
                    new_anomalies_this_ep += 1
                    experiment_seen_anomalies.add(signature)

                bug_key = (
                    classify_planted_bug(info["operation_id"], info["status_code"])
                    if use_planted_bugs else None
                )
                tracker.record_anomaly_at_step(
                    exp_id,
                    info["operation_id"],
                    info["status_code"],
                    signature,
                    episode=ep_num,
                    step=global_step,
                    is_planted_bug=int(bug_key is not None),
                    bug_name=bug_key,
                )

            if done or truncated:
                break

        tracker.record_episode(
            exp_id,
            ep_num,
            ep_reward,
            step_count,
            base_env.coverage.hits,
            base_env.coverage.anomaly_signatures,
            new_anomalies_this_ep,
        )
        ep_num += 1

    eval_requests = base_env.runner.request_count
    tracker.record_request_counts(
        exp_id,
        training_requests=0,
        prep_requests=0,
        eval_requests=eval_requests,
        total_requests=eval_requests,
        brainstorm_calls=0,
        seeded_episodes=0,
    )

    env.close()
    return exp_id


def _run_greedy_policy(
    spec_path: Path,
    n_episodes: int,
    max_steps_per_episode: int,
    use_planted_bugs: bool,
    trial_seed: int,
    db_path: Path,
    trial_id: int,
) -> int:
    """Evaluate CoverageGreedyPolicy using the paired evaluation seed schedule."""
    eval_seed_base = trial_seed + EVALUATION_SEED_OFFSET

    env = build_env(
        spec_path,
        use_mock=True,
        max_steps_per_episode=max_steps_per_episode,
        seed=eval_seed_base,
        enable_brainstormer=False,
        use_planted_bugs=use_planted_bugs,
    )

    tracker = ExperimentTracker(db_path)

    exp_id = tracker.start_experiment(
        name=f"greedy_trial_{trial_id}",
        policy="greedy",
        trial_id=trial_id,
        seed=trial_seed,
        max_steps_per_episode=max_steps_per_episode,
        timesteps_total=0,
        n_episodes=n_episodes,
    )

    policy = CoverageGreedyPolicy(env.action_space.n, seed=trial_seed)

    _evaluate_policy(
        env,
        lambda obs: policy.predict(obs)[0],
        tracker,
        exp_id,
        n_episodes,
        max_steps_per_episode,
        reset_seed_base=eval_seed_base,
        use_planted_bugs=use_planted_bugs,
    )

    eval_requests = env.unwrapped.runner.request_count
    tracker.record_request_counts(
        exp_id,
        training_requests=0,
        prep_requests=0,
        eval_requests=eval_requests,
        total_requests=eval_requests,
        brainstorm_calls=0,
        seeded_episodes=0,
    )

    env.close()
    return exp_id


def _run_bandit_policy(
    spec_path: Path,
    n_episodes: int,
    max_steps_per_episode: int,
    use_planted_bugs: bool,
    trial_seed: int,
    db_path: Path,
    trial_id: int,
) -> int:
    """Evaluate UCB1BanditPolicy using the paired evaluation seed schedule."""
    eval_seed_base = trial_seed + EVALUATION_SEED_OFFSET

    env = build_env(
        spec_path,
        use_mock=True,
        max_steps_per_episode=max_steps_per_episode,
        seed=eval_seed_base,
        enable_brainstormer=False,
        use_planted_bugs=use_planted_bugs,
    )

    tracker = ExperimentTracker(db_path)

    exp_id = tracker.start_experiment(
        name=f"bandit_trial_{trial_id}",
        policy="bandit",
        trial_id=trial_id,
        seed=trial_seed,
        max_steps_per_episode=max_steps_per_episode,
        timesteps_total=0,
        n_episodes=n_episodes,
    )

    policy = UCB1BanditPolicy(env.action_space.n, seed=trial_seed)

    _evaluate_policy(
        env,
        lambda obs: policy.predict(obs)[0],
        tracker,
        exp_id,
        n_episodes,
        max_steps_per_episode,
        reset_seed_base=eval_seed_base,
        use_planted_bugs=use_planted_bugs,
        on_step=policy.update,
    )

    eval_requests = env.unwrapped.runner.request_count
    tracker.record_request_counts(
        exp_id,
        training_requests=0,
        prep_requests=0,
        eval_requests=eval_requests,
        total_requests=eval_requests,
        brainstorm_calls=0,
        seeded_episodes=0,
    )

    env.close()
    return exp_id


def _run_ppo_policy(
    spec_path: Path,
    n_episodes: int,
    max_steps_per_episode: int,
    timesteps_total: int,
    trial_seed: int,
    db_path: Path,
    trial_id: int,
    enable_brainstormer: bool = False,
    use_planted_bugs: bool = False,
    brainstorm_interval_episodes: int = 50,
    llm_client: object | None = None,
) -> int:
    """Train PPO and evaluate the resulting policy."""

    train_env = build_env(
        spec_path,
        use_mock=True,
        max_steps_per_episode=max_steps_per_episode,
        seed=trial_seed,
        enable_brainstormer=enable_brainstormer,
        use_planted_bugs=use_planted_bugs,
        brainstorm_interval_episodes=brainstorm_interval_episodes,
        llm_client=llm_client,
    )

    monitored_train = Monitor(train_env)
    vec_env = DummyVecEnv([lambda: monitored_train])

    # Create args object with PPO hyperparameters
    ppo_args = argparse.Namespace(
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        seed=trial_seed,
        verbose=1,
    )

    model = build_model(vec_env, ppo_args)

    model.learn(total_timesteps=timesteps_total)

    # Extract training request metrics from train_env before closing
    prep_requests = getattr(train_env, "prep_request_count", 0)
    train_total_requests = train_env.unwrapped.runner.request_count
    training_requests = max(0, train_total_requests - prep_requests)
    brainstorm_calls = getattr(train_env, "brainstorm_calls", 0)
    seeded_episodes = getattr(train_env, "seeded_episodes", 0)

    vec_env.close()

    policy_name = "ppo_llm" if enable_brainstormer else "ppo"

    tracker = ExperimentTracker(db_path)

    exp_id = tracker.start_experiment(
        name=f"{policy_name}_trial_{trial_id}",
        policy=policy_name,
        trial_id=trial_id,
        seed=trial_seed,
        max_steps_per_episode=max_steps_per_episode,
        timesteps_total=timesteps_total,
        n_episodes=n_episodes,
    )

    eval_seed_base = trial_seed + EVALUATION_SEED_OFFSET

    # For evaluation, always disable brainstormer so that evaluation strictly
    # evaluates the learned policy, without scripted scenario action overrides.
    eval_env = build_env(
        spec_path,
        use_mock=True,
        max_steps_per_episode=max_steps_per_episode,
        seed=eval_seed_base,
        enable_brainstormer=False,
        use_planted_bugs=use_planted_bugs,
    )

    _evaluate_policy(
        eval_env,
        lambda obs: model.predict(obs, deterministic=True)[0],
        tracker,
        exp_id,
        n_episodes,
        max_steps_per_episode,
        reset_seed_base=eval_seed_base,
        use_planted_bugs=use_planted_bugs,
    )

    eval_requests = eval_env.unwrapped.runner.request_count
    total_requests = train_total_requests + eval_requests

    tracker.record_request_counts(
        exp_id,
        training_requests=training_requests,
        prep_requests=prep_requests,
        eval_requests=eval_requests,
        total_requests=total_requests,
        brainstorm_calls=brainstorm_calls,
        seeded_episodes=seeded_episodes,
    )

    eval_env.close()

    return exp_id


def _run_cross_api_ppo_with_exploration(
    spec_path: str | Path,
    n_episodes: int = 5,
    max_steps_per_episode: int = 20,
    timesteps_total: int = 0,
    total_budget: int | None = None,
    explore_budget: int | None = None,
    explore_ratio: float = 0.2,
    model_path: str | Path | None = None,
    use_planted_bugs: bool = False,
    trial_seed: int = 42,
    db_path: Path = DEFAULT_DB,
    trial_id: int = 0,
    on_progress=None,
    max_slots: int = DEFAULT_MAX_SLOTS,
) -> int:
    """Run an API execution with an explicit, bounded exploration phase followed by
    frozen MaskablePPO policy testing under strict action masking.

    1. Validates the OpenAPI specification and initializes clean, isolated API state.
    2. Performs bounded exploration up to explore_budget or (explore_ratio * total_budget),
       recording coverage, anomalies, and dependency values.
    3. Seamlessly transitions to pre-trained MaskablePPO using exploration state without retraining.
    4. Strictly enforces total request budget across both phases.
    5. Persists exploration and PPO metrics separately in SQLite.
    """
    from openapi_spec_validator import validate as validate_openapi
    from spec.parser import load_spec, parse_endpoints
    from corpus.seed_dict import build_seed_corpus
    from executor.request_runner import RequestRunner, BudgetExceededError
    from env.cross_api_env import (
        CrossApiFuzzEnv,
        DEFAULT_MAX_SLOTS,
        resolve_mock_transport_for_spec,
    )

    # 1. Spec validation & clean API state
    spec_file = Path(spec_path)
    if not spec_file.is_absolute():
        spec_file = REPO_ROOT / spec_file
    if not spec_file.exists() or not spec_file.is_file():
        raise FileNotFoundError(f"OpenAPI specification not found: {spec_file}")

    try:
        raw_spec = load_spec(spec_file)
        validate_openapi(raw_spec)
    except Exception as e:
        raise ValueError(f"Invalid OpenAPI specification at {spec_file}: {e}")

    endpoints = parse_endpoints(raw_spec)
    if not endpoints:
        raise ValueError(f"Specification {spec_file.name} contains no valid endpoints.")
    if len(endpoints) > max_slots:
        raise ValueError(
            f"Specification {spec_file.name} has {len(endpoints)} endpoints, "
            f"exceeding maximum slot capacity {max_slots}."
        )

    # 2. Budget calculation and enforcement
    if total_budget is not None and total_budget > 0:
        b_total = total_budget
    elif timesteps_total is not None and timesteps_total > 0:
        b_total = timesteps_total
    else:
        b_total = n_episodes * max_steps_per_episode
    b_total = max(1, b_total)

    if explore_budget is not None:
        b_explore = max(0, min(explore_budget, b_total))
    else:
        b_explore = max(0, min(int(b_total * explore_ratio), b_total))

    b_ppo = b_total - b_explore

    # Clean isolated transport and runner
    transport = resolve_mock_transport_for_spec(spec_file, use_planted_bugs=use_planted_bugs)
    if hasattr(transport, "reset_state"):
        try:
            transport.reset_state(reset_base=True)
        except TypeError:
            transport.reset_state()

    runner = RequestRunner(
        base_url="http://mock-api.local",
        rate_limit_per_sec=0,
        max_requests_per_run=b_total,
        transport=transport,
    )
    corpus = build_seed_corpus(endpoints)

    env = CrossApiFuzzEnv(
        endpoints=endpoints,
        corpus=corpus,
        runner=runner,
        max_slots=max_slots,
        max_steps_per_episode=max_steps_per_episode,
        seed=trial_seed,
    )

    tracker = ExperimentTracker(db_path)
    exp_id = tracker.start_experiment(
        name=f"cross_api_ppo_trial_{trial_id}",
        policy="cross_api_ppo",
        trial_id=trial_id,
        seed=trial_seed,
        max_steps_per_episode=max_steps_per_episode,
        timesteps_total=b_total,
        n_episodes=n_episodes,
    )

    seen_anomalies: set[tuple] = set()
    exploration_requests = 0
    exploration_reward = 0.0
    active_count = len(endpoints)
    visited_counts = [0] * active_count
    producer_indices = [i for i, ep in enumerate(endpoints) if ep.operation_id in env.producer_ops]

    if on_progress:
        on_progress({
            "stage": "exploring",
            "phase": "exploration",
            "requests_made": 0,
            "exploration_requests": 0,
            "ppo_requests": 0,
            "budget_total": b_total,
            "coverage_count": 0,
            "anomalies_found": 0,
        })

    # =========================================================================
    # Phase 1: Explicit Bounded Exploration
    # =========================================================================
    obs, info = env.reset(seed=trial_seed)

    while exploration_requests < b_explore and runner.request_count < b_total:
        # Choose action: prioritize unvisited producers, then unvisited with ready deps,
        # then unvisited, then uncovered, then least-visited active slot.
        chosen_action = None

        # 1. Unvisited producers
        unvisited_producers = [p for p in producer_indices if visited_counts[p] == 0]
        if unvisited_producers:
            chosen_action = unvisited_producers[0]

        # 2. Unvisited with ready dependencies
        if chosen_action is None:
            unvisited = [i for i in range(active_count) if visited_counts[i] == 0]
            ready_unvisited = [i for i in unvisited if env._dependencies_ready(endpoints[i]) > 0.5]
            if ready_unvisited:
                chosen_action = ready_unvisited[0]
            elif unvisited:
                chosen_action = unvisited[0]

        # 3. Uncovered endpoints
        if chosen_action is None:
            uncovered = [
                i for i in range(active_count)
                if not env.coverage.endpoint_covered(endpoints[i].operation_id)
            ]
            ready_uncovered = [i for i in uncovered if env._dependencies_ready(endpoints[i]) > 0.5]
            if ready_uncovered:
                chosen_action = ready_uncovered[0]
            elif uncovered:
                chosen_action = uncovered[0]

        # 4. Fallback: least-visited active endpoint (round robin)
        if chosen_action is None:
            min_visits = min(visited_counts)
            candidates = [i for i in range(active_count) if visited_counts[i] == min_visits]
            chosen_action = candidates[0]

        visited_counts[chosen_action] += 1

        obs, reward, done, truncated, step_info = env.step(chosen_action)
        exploration_requests += 1
        exploration_reward += reward

        sig = step_info.get("anomaly_signature")
        if sig is not None:
            seen_anomalies.add(sig)
            bug_key = (
                classify_planted_bug(step_info["operation_id"], step_info["status_code"])
                if use_planted_bugs else None
            )
            tracker.record_anomaly_at_step(
                exp_id,
                step_info["operation_id"],
                step_info["status_code"],
                sig,
                episode=0,
                step=runner.request_count,
                is_planted_bug=int(bug_key is not None),
                bug_name=bug_key,
            )

        if on_progress:
            on_progress({
                "stage": "exploring",
                "phase": "exploration",
                "requests_made": runner.request_count,
                "exploration_requests": exploration_requests,
                "ppo_requests": 0,
                "budget_total": b_total,
                "coverage_count": len(env.coverage.hits),
                "anomalies_found": len(seen_anomalies),
            })

        if step_info.get("budget_exceeded") or runner.request_count >= b_total:
            break

        if truncated:
            obs, _ = env.reset(preserve_state=True)

    # Persist exploration episode if any requests were made
    if exploration_requests > 0:
        tracker.record_episode(
            exp_id,
            episode_number=0,
            total_reward=exploration_reward,
            steps_taken=exploration_requests,
            coverage_hits=env.coverage.hits,
            anomaly_signatures=env.coverage.anomaly_signatures,
            new_anomaly_count=len(seen_anomalies),
        )

    # =========================================================================
    # Phase 2: Frozen MaskablePPO Policy Execution with Strict Action Masking
    # =========================================================================
    remaining_budget = max(0, b_total - runner.request_count)
    ppo_requests = 0

    if remaining_budget > 0:
        # Resolve saved model
        resolved_model_path: Path | None = None
        if model_path:
            p = Path(model_path)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if p.exists():
                resolved_model_path = p
            elif p.with_suffix(".zip").exists():
                resolved_model_path = p.with_suffix(".zip")
            else:
                raise FileNotFoundError(f"MaskablePPO model file not found at: {model_path}")
        else:
            default_candidate = REPO_ROOT / "runs" / "cross_api_ppo"
            if default_candidate.with_suffix(".zip").exists():
                resolved_model_path = default_candidate.with_suffix(".zip")
            elif default_candidate.exists():
                resolved_model_path = default_candidate
            else:
                raise FileNotFoundError(
                    f"No MaskablePPO model specified and default model not found at {default_candidate}. "
                    "A pre-trained MaskablePPO model is required to run the cross_api_ppo policy."
                )

        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(str(resolved_model_path))

        # Reset env preserving exploration coverage, dependency values, and mock state
        obs, _ = env.reset(preserve_state=True)

        if on_progress:
            on_progress({
                "stage": "policy_testing",
                "phase": "ppo",
                "requests_made": runner.request_count,
                "exploration_requests": exploration_requests,
                "ppo_requests": 0,
                "budget_total": b_total,
                "coverage_count": len(env.coverage.hits),
                "anomalies_found": len(seen_anomalies),
            })

        ep_num = 1
        while ppo_requests < remaining_budget and runner.request_count < b_total:
            ep_reward = 0.0
            step_count = 0
            new_anomalies_this_ep = 0

            while step_count < max_steps_per_episode and ppo_requests < remaining_budget and runner.request_count < b_total:
                action_mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
                action = int(action)

                obs, reward, done, truncated, step_info = env.step(action)
                if step_info.get("budget_exceeded"):
                    break

                ppo_requests += 1
                step_count += 1
                ep_reward += reward

                sig = step_info.get("anomaly_signature")
                if sig is not None:
                    if sig not in seen_anomalies:
                        new_anomalies_this_ep += 1
                        seen_anomalies.add(sig)
                    bug_key = (
                        classify_planted_bug(step_info["operation_id"], step_info["status_code"])
                        if use_planted_bugs else None
                    )
                    tracker.record_anomaly_at_step(
                        exp_id,
                        step_info["operation_id"],
                        step_info["status_code"],
                        sig,
                        episode=ep_num,
                        step=runner.request_count,
                        is_planted_bug=int(bug_key is not None),
                        bug_name=bug_key,
                    )

                if on_progress:
                    on_progress({
                        "stage": "policy_testing",
                        "phase": "ppo",
                        "requests_made": runner.request_count,
                        "exploration_requests": exploration_requests,
                        "ppo_requests": ppo_requests,
                        "budget_total": b_total,
                        "coverage_count": len(env.coverage.hits),
                        "anomalies_found": len(seen_anomalies),
                    })

                if done or truncated:
                    break

            tracker.record_episode(
                exp_id,
                ep_num,
                ep_reward,
                step_count,
                env.coverage.hits,
                env.coverage.anomaly_signatures,
                new_anomalies_this_ep,
            )
            ep_num += 1

            if ppo_requests < remaining_budget and runner.request_count < b_total:
                obs, _ = env.reset(preserve_state=True)

    # 3. Final metrics and request accounting
    actual_total = runner.request_count
    tracker.record_request_counts(
        exp_id,
        training_requests=0,
        prep_requests=exploration_requests,
        eval_requests=ppo_requests,
        total_requests=actual_total,
        brainstorm_calls=0,
        seeded_episodes=0,
        exploration_requests=exploration_requests,
        ppo_requests=ppo_requests,
    )

    if on_progress:
        on_progress({
            "stage": "completed",
            "phase": "completed",
            "requests_made": actual_total,
            "exploration_requests": exploration_requests,
            "ppo_requests": ppo_requests,
            "budget_total": b_total,
            "coverage_count": len(env.coverage.hits),
            "anomalies_found": len(seen_anomalies),
        })

    env.close()
    return exp_id


def run_experiments(
    spec_path: Path,
    db_path: Path,
    n_trials: int,
    policies: list[str],
    n_episodes_per_trial: int,
    max_steps_per_episode: int,
    timesteps_per_trial: int,
    seed_start: int,
    use_planted_bugs: bool = False,
    brainstorm_interval: int = 50,
    llm_client: object | None = None,
):
    """Run a full experiment suite: all policies × all trials."""

    print("[Phase 6] Starting experiments")
    print(f"  DB: {db_path}")
    print(f"  Spec: {spec_path}")
    print(f"  Policies: {', '.join(policies)}")
    print(f"  Trials: {n_trials}")
    print(f"  Episodes per trial: {n_episodes_per_trial}")
    print(f"  Seed range: {seed_start}-{seed_start + n_trials - 1}")
    print()

    experiment_ids = []

    for trial_id in range(n_trials):
        trial_seed = seed_start + trial_id

        print(f"\n{'=' * 70}")
        print(f"TRIAL {trial_id + 1}/{n_trials} (seed={trial_seed})")
        print(f"{'=' * 70}")

        for policy in policies:
            print(f"\n  Running {policy}...", end=" ", flush=True)

            if policy == "random":

                exp_id = _run_random_policy(
                    spec_path,
                    n_episodes=n_episodes_per_trial,
                    max_steps_per_episode=max_steps_per_episode,
                    use_planted_bugs=use_planted_bugs,
                    trial_seed=trial_seed,
                    db_path=db_path,
                    trial_id=trial_id,
                )

            elif policy == "random_budget_matched":

                exp_id = _run_budget_matched_random_policy(
                    spec_path,
                    n_episodes=n_episodes_per_trial,
                    max_steps_per_episode=max_steps_per_episode,
                    timesteps_total=timesteps_per_trial,
                    use_planted_bugs=use_planted_bugs,
                    trial_seed=trial_seed,
                    db_path=db_path,
                    trial_id=trial_id,
                )

            elif policy == "greedy":

                exp_id = _run_greedy_policy(
                    spec_path,
                    n_episodes=n_episodes_per_trial,
                    max_steps_per_episode=max_steps_per_episode,
                    use_planted_bugs=use_planted_bugs,
                    trial_seed=trial_seed,
                    db_path=db_path,
                    trial_id=trial_id,
                )

            elif policy == "bandit":

                exp_id = _run_bandit_policy(
                    spec_path,
                    n_episodes=n_episodes_per_trial,
                    max_steps_per_episode=max_steps_per_episode,
                    use_planted_bugs=use_planted_bugs,
                    trial_seed=trial_seed,
                    db_path=db_path,
                    trial_id=trial_id,
                )

            elif policy == "ppo":

                exp_id = _run_ppo_policy(
                    spec_path,
                    n_episodes=n_episodes_per_trial,
                    max_steps_per_episode=max_steps_per_episode,
                    timesteps_total=timesteps_per_trial,
                    trial_seed=trial_seed,
                    db_path=db_path,
                    trial_id=trial_id,
                    enable_brainstormer=False,
                    use_planted_bugs=use_planted_bugs,
                )

            elif policy == "ppo_llm":

                print("\n    (requires `ollama serve` running locally)")

                exp_id = _run_ppo_policy(
                    spec_path,
                    n_episodes=n_episodes_per_trial,
                    max_steps_per_episode=max_steps_per_episode,
                    timesteps_total=timesteps_per_trial,
                    trial_seed=trial_seed,
                    db_path=db_path,
                    trial_id=trial_id,
                    enable_brainstormer=True,
                    use_planted_bugs=use_planted_bugs,
                    brainstorm_interval_episodes=brainstorm_interval,
                    llm_client=llm_client,
                )

            elif policy == "cross_api_ppo":

                exp_id = _run_cross_api_ppo_with_exploration(
                    spec_path=spec_path,
                    n_episodes=n_episodes_per_trial,
                    max_steps_per_episode=max_steps_per_episode,
                    timesteps_total=timesteps_per_trial,
                    trial_seed=trial_seed,
                    db_path=db_path,
                    trial_id=trial_id,
                    use_planted_bugs=use_planted_bugs,
                )

            else:
                raise ValueError(f"Unknown policy: {policy}")

            experiment_ids.append(exp_id)

            print(f"[OK] (exp_id={exp_id})")

    return experiment_ids


def summarize_results(
    db_path: Path,
    policies: list[str],
    experiment_ids: list[int],
):
    """
    Summarize only the experiments created by the current run.

    This prevents historical experiments in the same SQLite database
    from contaminating the current experiment results.
    """

    tracker = ExperimentTracker(db_path)

    print(f"\n{'=' * 70}")
    print("AGGREGATING RESULTS")
    print(f"{'=' * 70}\n")

    policy_metrics_list = []

    for policy in policies:

        experiments = []

        # Only load experiments belonging to THIS run.
        for exp_id in experiment_ids:
            exp = tracker.get_experiment(exp_id)

            if exp and exp["policy"] == policy:
                experiments.append(exp)

        if not experiments:
            print(f"No results found for policy: {policy}")
            continue

        print(f"\n{policy.upper()}")
        print("-" * 70)

        metrics_list = []

        for exp in experiments:

            metrics = tracker.compute_final_metrics(exp["id"])

            if metrics:
                metrics_list.append(metrics)

                trial = exp["trial_id"]

                eval_steps_str = (
                    f"{metrics['eval_steps_to_first_bug']}"
                    if metrics.get("eval_steps_to_first_bug") is not None
                    else "N/A"
                )
                total_reqs_first_bug_str = (
                    f"{metrics['total_requests_to_first_bug']}"
                    if metrics.get("total_requests_to_first_bug") is not None
                    else "N/A"
                )
                extra_llm = ""
                if metrics.get("brainstorm_calls", 0) > 0 or metrics.get("seeded_episodes", 0) > 0:
                    extra_llm = f", brainstorm_calls={metrics.get('brainstorm_calls', 0)}, seeded_eps={metrics.get('seeded_episodes', 0)}"

                print(
                    f"  Trial {trial + 1}: "
                    f"reward={metrics['mean_episode_reward']:.3f}, "
                    f"ops={metrics['unique_operations_covered']}, "
                    f"states={metrics.get('distinct_coverage_states', 0)}, "
                    f"hits={metrics.get('total_coverage_hits', 0)}, "
                    f"anomalies={metrics['unique_anomalies_found']}, "
                    f"planted={metrics.get('planted_bugs_found', 0)}/3, "
                    f"false_pos={metrics.get('false_positives', 0)}, "
                    f"eval_steps_to_bug={eval_steps_str}, "
                    f"tot_reqs_to_bug={total_reqs_first_bug_str}{extra_llm}"
                )

        if metrics_list:

            agg = aggregate_metrics(metrics_list)

            policy_metrics_list.append(agg)

    if policy_metrics_list:
        print("\n" + compare_policies(policy_metrics_list))


def build_arg_parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(
        description="Phase 6: Experiment runner (Random vs PPO vs PPO+LLM)."
    )

    p.add_argument(
        "--spec",
        type=str,
        default=None,
        help="Path to OpenAPI spec (default: bundled sample_api.yaml).",
    )

    p.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB),
        help="SQLite DB path for experiment tracking.",
    )

    p.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="Number of repeated trials per policy.",
    )

    p.add_argument(
        "--n-episodes",
        type=int,
        default=20,
        help="Episodes to evaluate each policy.",
    )

    p.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max steps per episode.",
    )

    p.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Total timesteps for PPO training per trial.",
    )

    p.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=["random", "ppo"],
        choices=ALL_POLICIES,
        help=f"Policies to run (choices: {', '.join(ALL_POLICIES)}).",
    )

    p.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="Starting RNG seed (incremented per trial).",
    )

    p.add_argument(
        "--use-planted-bugs",
        action="store_true",
        help="Use planted-bug mock target (3 intentional bugs for discovery testing).",
    )

    p.add_argument(
        "--brainstorm-interval",
        type=int,
        default=50,
        help="Episodes between LLM brainstorm refreshes for ppo_llm (default: 500).",
    )

    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip experiments; just print aggregated results from DB.",
    )

    p.add_argument(
        "--analyze",
        action="store_true",
        help="Run statistical significance analysis (A12, Mann-Whitney U, Bootstrap CI) and print Markdown report.",
    )

    return p


def main(argv: list[str] | None = None):

    args = build_arg_parser().parse_args(argv)

    spec_path = Path(args.spec) if args.spec else DEFAULT_MOCK_SPEC
    db_path = Path(args.db_path)

    if not args.summarize_only:

        experiment_ids = run_experiments(
            spec_path=spec_path,
            db_path=db_path,
            n_trials=args.n_trials,
            policies=args.policies,
            n_episodes_per_trial=args.n_episodes,
            max_steps_per_episode=args.max_steps,
            timesteps_per_trial=args.timesteps,
            seed_start=args.seed_start,
            use_planted_bugs=args.use_planted_bugs,
            brainstorm_interval=args.brainstorm_interval,
        )

    else:

        # For summarize-only mode, use all matching experiments
        # currently present in the database.
        tracker = ExperimentTracker(db_path)

        experiment_ids = []

        for policy in args.policies:

            experiments = tracker.get_experiments(policy=policy)

            for exp in experiments:
                experiment_ids.append(exp["id"])

    summarize_results(
        db_path,
        args.policies,
        experiment_ids,
    )

    if args.analyze:
        from experiment.analysis import extract_policy_summaries, generate_markdown_report
        summaries = extract_policy_summaries(db_path)
        if summaries:
            print("\n" + generate_markdown_report(summaries))


if __name__ == "__main__":
    main()