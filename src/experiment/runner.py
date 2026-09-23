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
from experiment.tracker import ExperimentTracker
from experiment.metrics import aggregate_metrics, compare_policies
from executor.planted_bugs import PlantedBugsMockTarget


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOCK_SPEC = REPO_ROOT / "tests" / "fixtures" / "sample_api.yaml"
DEFAULT_DB = REPO_ROOT / "runs" / "phase6_experiments.db"


def _run_random_policy(
    spec_path: Path,
    n_episodes: int,
    max_steps_per_episode: int,
    use_planted_bugs: bool,
    trial_seed: int,
    db_path: Path,
    trial_id: int,
) -> int:
    """Run and evaluate the random policy."""

    env = build_env(
        spec_path,
        use_mock=True,
        max_steps_per_episode=max_steps_per_episode,
        seed=trial_seed,
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

    for ep_num in range(n_episodes):
        obs, info = env.reset(seed=trial_seed + ep_num)

        ep_reward = 0.0
        step_count = 0

        while step_count < max_steps_per_episode:
            action, _ = policy.predict(obs)

            obs, reward, done, truncated, info = env.step(action)

            ep_reward += reward
            step_count += 1

            if done or truncated:
                break

        tracker.record_episode(
            exp_id,
            ep_num,
            ep_reward,
            step_count,
            env.coverage.hits,
            env.coverage.anomaly_signatures,
            len(env.coverage.anomaly_signatures),
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
) -> int:
    """Train PPO and evaluate the resulting policy."""

    train_env = build_env(
        spec_path,
        use_mock=True,
        max_steps_per_episode=max_steps_per_episode,
        seed=trial_seed,
        enable_brainstormer=enable_brainstormer,
        use_planted_bugs=use_planted_bugs,
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

    eval_env = build_env(
        spec_path,
        use_mock=True,
        max_steps_per_episode=max_steps_per_episode,
        seed=trial_seed + 10000,
        enable_brainstormer=enable_brainstormer,
        use_planted_bugs=use_planted_bugs,
    )

    for ep_num in range(n_episodes):
        obs, info = eval_env.reset(seed=trial_seed + 10000 + ep_num)

        ep_reward = 0.0
        step_count = 0

        while step_count < max_steps_per_episode:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = eval_env.step(action)

            ep_reward += reward
            step_count += 1

            if done or truncated:
                break

        tracker.record_episode(
            exp_id,
            ep_num,
            ep_reward,
            step_count,
            eval_env.coverage.hits,
            eval_env.coverage.anomaly_signatures,
            len(eval_env.coverage.anomaly_signatures),
        )

    eval_env.close()

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
):
    """Run a full experiment suite: all policies × all trials."""

    print("[Phase 6] Starting experiments")
    print(f"  DB: {db_path}")
    print(f"  Spec: {spec_path}")
    print(f"  Policies: {', '.join(policies)}")
    print(f"  Trials: {n_trials}")
    print(f"  Episodes per trial: {n_episodes_per_trial}")
    print(f"  Seed range: {seed_start}–{seed_start + n_trials - 1}")
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
                )

            else:
                raise ValueError(f"Unknown policy: {policy}")

            experiment_ids.append(exp_id)

            print(f"✓ (exp_id={exp_id})")

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

                print(
                    f"  Trial {trial + 1}: "
                    f"reward={metrics['mean_episode_reward']:.3f}, "
                    f"ops={metrics['unique_operations_covered']}, "
                    f"anomalies={metrics['unique_anomalies_found']}"
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
        choices=["random", "ppo", "ppo_llm"],
        help="Policies to run.",
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
        "--summarize-only",
        action="store_true",
        help="Skip experiments; just print aggregated results from DB.",
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


if __name__ == "__main__":
    main()