"""
Tests for intelligent exploration baselines:
- CoverageGreedyPolicy
- UCB1BanditPolicy
- Budget-Matched Random policy runner
- Baseline integration into experiment runner
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from agent.greedy_policy import CoverageGreedyPolicy
from agent.bandit_policy import UCB1BanditPolicy
from experiment.tracker import ExperimentTracker
from experiment.runner import (
    _run_budget_matched_random_policy,
    _run_greedy_policy,
    _run_bandit_policy,
    EVALUATION_SEED_OFFSET,
    ALL_POLICIES,
)

FIXTURE_SPEC = Path(__file__).parent / "fixtures" / "sample_api.yaml"


def test_coverage_greedy_prioritizes_uncovered_endpoints():
    policy = CoverageGreedyPolicy(n_actions=4, seed=42)

    # Observation with 4 endpoints: 0, 1, 3 covered; 2 uncovered
    obs = np.array([1.0, 1.0, 0.0, 1.0, 0.5])
    action, _ = policy.predict(obs)
    assert action == 2

    # Multiple uncovered: 1 and 3 uncovered; selects one of them
    obs2 = np.array([1.0, 0.0, 1.0, 0.0, 0.5])
    action2, _ = policy.predict(obs2)
    assert action2 in (1, 3)


def test_coverage_greedy_falls_back_to_least_hit():
    policy = CoverageGreedyPolicy(n_actions=3, seed=42)
    policy.hit_counts = {0: 10, 1: 2, 2: 8}

    # All endpoints covered in current observation
    obs = np.array([1.0, 1.0, 1.0, 0.8])
    action, _ = policy.predict(obs)
    assert action == 1  # Least-hit endpoint


def test_ucb1_bandit_pulls_all_arms_first():
    bandit = UCB1BanditPolicy(n_actions=3, seed=42)
    pulled = set()

    for _ in range(3):
        action, _ = bandit.predict()
        pulled.add(action)
        bandit.update(action, 0.0)

    assert pulled == {0, 1, 2}


def test_ucb1_bandit_exploits_high_reward_arm():
    bandit = UCB1BanditPolicy(n_actions=3, seed=42)

    # Pull each arm once
    for a in range(3):
        bandit.predict()
        bandit.update(a, 0.0)

    # Give arm 1 strong positive rewards
    for _ in range(5):
        bandit.update(1, 10.0)

    # Subsequent predictions should exploit arm 1
    action, _ = bandit.predict()
    assert action == 1


def test_budget_matched_random_policy_runner():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_bm.db"
        target_timesteps = 25
        n_episodes = 2
        max_steps = 5
        # Total expected requests: 25 + (2 * 5) = 35

        exp_id = _run_budget_matched_random_policy(
            spec_path=FIXTURE_SPEC,
            n_episodes=n_episodes,
            max_steps_per_episode=max_steps,
            timesteps_total=target_timesteps,
            use_planted_bugs=False,
            trial_seed=42,
            db_path=db_path,
            trial_id=0,
        )

        tracker = ExperimentTracker(db_path)
        metrics = tracker.compute_final_metrics(exp_id)

        assert metrics is not None
        assert metrics["policy"] == "random_budget_matched"
        assert metrics["eval_requests"] == 35
        assert metrics["total_requests"] == 35
        assert metrics["training_requests"] == 0
        assert metrics["prep_requests"] == 0


def test_greedy_and_bandit_policy_runners():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_baselines.db"

        greedy_exp_id = _run_greedy_policy(
            spec_path=FIXTURE_SPEC,
            n_episodes=2,
            max_steps_per_episode=5,
            use_planted_bugs=True,
            trial_seed=42,
            db_path=db_path,
            trial_id=0,
        )

        bandit_exp_id = _run_bandit_policy(
            spec_path=FIXTURE_SPEC,
            n_episodes=2,
            max_steps_per_episode=5,
            use_planted_bugs=True,
            trial_seed=42,
            db_path=db_path,
            trial_id=0,
        )

        tracker = ExperimentTracker(db_path)

        greedy_m = tracker.compute_final_metrics(greedy_exp_id)
        assert greedy_m is not None
        assert greedy_m["policy"] == "greedy"
        assert greedy_m["eval_requests"] == 10
        assert greedy_m["total_requests"] == 10

        bandit_m = tracker.compute_final_metrics(bandit_exp_id)
        assert bandit_m is not None
        assert bandit_m["policy"] == "bandit"
        assert bandit_m["eval_requests"] == 10
        assert bandit_m["total_requests"] == 10


def test_all_policies_in_all_policies_list():
    assert "random" in ALL_POLICIES
    assert "random_budget_matched" in ALL_POLICIES
    assert "greedy" in ALL_POLICIES
    assert "bandit" in ALL_POLICIES
    assert "ppo" in ALL_POLICIES
    assert "ppo_llm" in ALL_POLICIES
