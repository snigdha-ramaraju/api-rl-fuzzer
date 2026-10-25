"""
Tests for statistical analysis module:
- Vargha-Delaney A12 effect size calculation & interpretation
- Mann-Whitney U test (non-parametric two-sample)
- Wilcoxon signed-rank test (paired samples)
- Bootstrap 95% confidence intervals
- Markdown comparative reporting from SQLite
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from experiment.analysis import (
    vargha_delaney_a12,
    interpret_a12,
    mann_whitney_u_test,
    wilcoxon_signed_rank_test,
    bootstrap_mean_ci,
    extract_policy_summaries,
    generate_markdown_report,
)
from experiment.tracker import ExperimentTracker


def test_vargha_delaney_a12_properties():
    # Exactly identical samples -> 0.5
    assert vargha_delaney_a12([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.5
    assert interpret_a12(0.5) == "negligible"

    # X strictly dominates Y -> 1.0
    assert vargha_delaney_a12([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]) == 1.0
    assert interpret_a12(1.0) == "large"

    # Y strictly dominates X -> 0.0
    assert vargha_delaney_a12([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == 0.0
    assert interpret_a12(0.0) == "large"

    # Intermediate known overlap
    x = [2.0, 4.0, 6.0]
    y = [1.0, 3.0, 5.0]
    # Pairs: (2>1, 2<3, 2<5), (4>1, 4>3, 4<5), (6>1, 6>3, 6>5) -> 1 + 2 + 3 = 6 wins out of 9
    assert abs(vargha_delaney_a12(x, y) - (6.0 / 9.0)) < 1e-6


def test_mann_whitney_u_test_significant():
    x = [100.0, 102.0, 105.0, 107.0, 110.0, 112.0]
    y = [10.0, 12.0, 14.0, 15.0, 17.0, 19.0]
    u, p_val = mann_whitney_u_test(x, y)
    assert u == 0.0
    assert p_val < 0.01

    # Identical distributions
    _, p_same = mann_whitney_u_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert p_same > 0.5


def test_wilcoxon_signed_rank_test_paired():
    # Strict positive improvements across all pairs (N=6 to reach p < 0.05)
    x = [10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    y = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    t, p_val = wilcoxon_signed_rank_test(x, y)
    assert t == 0.0
    assert p_val < 0.05


def test_bootstrap_mean_ci_coverage():
    diffs = [2.0, 2.5, 3.0, 3.5, 4.0]
    mean_val = sum(diffs) / len(diffs)  # 3.0
    low, high = bootstrap_mean_ci(diffs, n_resamples=1000)
    assert low <= mean_val <= high
    assert low >= 2.0
    assert high <= 4.0


def test_extract_and_generate_markdown_report():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_report.db"
        tracker = ExperimentTracker(db_path)

        # Record Random trial
        exp_r = tracker.start_experiment("random_0", "random", 0, 42, 10, 0, 2)
        tracker.record_episode(exp_r, 0, 5.0, 10, {("createOrder", 201): 5}, set(), 0)
        tracker.record_episode(exp_r, 1, 6.0, 10, {("createOrder", 201): 5}, set(), 0)
        tracker.record_request_counts(exp_r, 0, 0, 20, 20, 0, 0)

        # Record PPO trial
        exp_p = tracker.start_experiment("ppo_0", "ppo", 0, 42, 10, 100, 2)
        tracker.record_episode(exp_p, 0, 15.0, 10, {("createOrder", 201): 5, ("getOrder", 200): 5}, set(), 1)
        tracker.record_episode(exp_p, 1, 16.0, 10, {("createOrder", 201): 5, ("getOrder", 200): 5}, set(), 0)
        tracker.record_request_counts(exp_p, 100, 0, 20, 120, 0, 0)

        summaries = extract_policy_summaries(db_path)
        assert "random" in summaries
        assert "ppo" in summaries
        assert summaries["ppo"].mean_reward > summaries["random"].mean_reward

        report = generate_markdown_report(summaries)
        assert "# Research Evaluation" in report
        assert "`random`" in report
        assert "`ppo`" in report
        assert "Vargha-Delaney" in report
