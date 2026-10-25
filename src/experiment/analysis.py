"""
Phase 6: Statistical Analysis & Reporting Module.

Provides rigorous non-parametric statistical tests and effect size metrics
for comparing API fuzzing policies across repeated trials:
- Vargha-Delaney A12 effect size
- Mann-Whitney U test (independent samples)
- Wilcoxon signed-rank test (paired trial seeds)
- Bootstrap 95% confidence intervals for metric differences
- Formatted Markdown summary tables
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

from experiment.tracker import ExperimentTracker


def vargha_delaney_a12(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Compute Vargha-Delaney A12 non-parametric effect size.

    A12 measures the probability that a randomly selected observation from
    group X is greater than a randomly selected observation from group Y.
    - A12 = 0.50: No effect / stochastic equality
    - A12 > 0.50: Group X stochastically dominates Group Y
    - Thresholds: negligible (< 0.56), small (0.56-0.64), medium (0.64-0.71), large (>= 0.71)
    """
    n = len(x)
    m = len(y)
    if n == 0 or m == 0:
        return 0.5

    wins = 0.0
    for xi in x:
        for yj in y:
            if xi > yj:
                wins += 1.0
            elif xi == yj:
                wins += 0.5

    return wins / (n * m)


def interpret_a12(a12: float) -> str:
    """Classify Vargha-Delaney effect size magnitude."""
    diff = abs(a12 - 0.5)
    if diff < 0.06:
        return "negligible"
    elif diff < 0.14:
        return "small"
    elif diff < 0.21:
        return "medium"
    else:
        return "large"


def _norm_cdf(z: float) -> float:
    """Standard normal cumulative distribution function using math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_whitney_u_test(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """
    Two-tailed Mann-Whitney U test with normal approximation and continuity correction.
    Returns (U, p_value).
    """
    n1 = len(x)
    n2 = len(y)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Combine and rank
    combined = [(v, 0, i) for i, v in enumerate(x)] + [(v, 1, j) for j, v in enumerate(y)]
    combined.sort(key=lambda t: t[0])

    # Assign average ranks for ties
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum of ranks for sample 1
    r1 = sum(ranks[idx] for idx, item in enumerate(combined) if item[1] == 0)

    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - r1
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    # Normal approximation
    mean_u = (n1 * n2) / 2.0
    var_u = (n1 * n2 * (n1 + n2 + 1)) / 12.0

    if var_u <= 0:
        return u, 1.0

    z = (abs(u - mean_u) - 0.5) / math.sqrt(var_u)
    p_value = 2.0 * (1.0 - _norm_cdf(z))
    p_value = max(0.0, min(1.0, p_value))

    return u, p_value


def wilcoxon_signed_rank_test(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """
    Two-tailed Wilcoxon signed-rank test for paired samples.
    Returns (T, p_value).
    """
    if len(x) != len(y):
        raise ValueError("Paired samples must have identical lengths")

    diffs = [xi - yi for xi, yi in zip(x, y) if (xi - yi) != 0.0]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    # Rank absolute differences
    abs_diffs = [(abs(d), d, i) for i, d in enumerate(diffs)]
    abs_diffs.sort(key=lambda t: t[0])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    w_pos = sum(ranks[idx] for idx, item in enumerate(abs_diffs) if item[1] > 0)
    w_neg = sum(ranks[idx] for idx, item in enumerate(abs_diffs) if item[1] < 0)
    t = min(w_pos, w_neg)

    # Normal approximation for n >= 5
    mean_t = (n * (n + 1)) / 4.0
    var_t = (n * (n + 1) * (2 * n + 1)) / 24.0

    if var_t <= 0:
        return t, 1.0

    z = (abs(t - mean_t) - 0.5) / math.sqrt(var_t)
    p_value = 2.0 * (1.0 - _norm_cdf(z))
    p_value = max(0.0, min(1.0, p_value))

    return t, p_value


def bootstrap_mean_ci(
    diffs: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute non-parametric 95% bootstrap confidence interval for the mean.
    Returns (lower_bound, upper_bound).
    """
    if not diffs:
        return 0.0, 0.0
    if len(diffs) == 1:
        return diffs[0], diffs[0]

    rng = random.Random(seed)
    n = len(diffs)
    bootstrap_means = []

    for _ in range(n_resamples):
        sample = [rng.choice(diffs) for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    lower_idx = int((alpha / 2.0) * n_resamples)
    upper_idx = int((1.0 - alpha / 2.0) * n_resamples)

    return bootstrap_means[lower_idx], bootstrap_means[min(upper_idx, n_resamples - 1)]


@dataclass
class PolicyEvaluationSummary:
    policy: str
    n_trials: int
    mean_reward: float
    std_reward: float
    mean_ops_covered: float
    std_ops_covered: float
    mean_distinct_states: float
    std_distinct_states: float
    mean_planted_bugs: float
    std_planted_bugs: float
    mean_anomalies: float
    std_anomalies: float
    mean_total_requests: float
    raw_rewards: list[float]
    raw_ops: list[float]
    raw_distinct_states: list[float]
    raw_planted_bugs: list[float]
    raw_total_requests: list[float]


def extract_policy_summaries(db_path: Path | str) -> dict[str, PolicyEvaluationSummary]:
    """Extract and aggregate trial-level metrics per policy from SQLite."""
    tracker = ExperimentTracker(db_path)
    experiments = tracker.get_experiments()

    by_policy: dict[str, list[dict]] = {}
    for exp in experiments:
        pol = exp["policy"]
        m = tracker.compute_final_metrics(exp["id"])
        if m:
            by_policy.setdefault(pol, []).append(m)

    summaries = {}
    for pol, metrics_list in by_policy.items():
        n = len(metrics_list)
        rewards = [m["mean_episode_reward"] for m in metrics_list]
        ops = [m["unique_operations_covered"] for m in metrics_list]
        states = [m.get("distinct_coverage_states", 0) for m in metrics_list]
        planted = [m.get("planted_bugs_found", 0) for m in metrics_list]
        anomalies = [m["unique_anomalies_found"] for m in metrics_list]
        reqs = [m.get("total_requests", 0) for m in metrics_list]

        def _mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        def _std(vals):
            if len(vals) <= 1:
                return 0.0
            m = _mean(vals)
            return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

        summaries[pol] = PolicyEvaluationSummary(
            policy=pol,
            n_trials=n,
            mean_reward=_mean(rewards),
            std_reward=_std(rewards),
            mean_ops_covered=_mean(ops),
            std_ops_covered=_std(ops),
            mean_distinct_states=_mean(states),
            std_distinct_states=_std(states),
            mean_planted_bugs=_mean(planted),
            std_planted_bugs=_std(planted),
            mean_anomalies=_mean(anomalies),
            std_anomalies=_std(anomalies),
            mean_total_requests=_mean(reqs),
            raw_rewards=rewards,
            raw_ops=ops,
            raw_distinct_states=states,
            raw_planted_bugs=planted,
            raw_total_requests=reqs,
        )

    return summaries


def generate_markdown_report(summaries: dict[str, PolicyEvaluationSummary]) -> str:
    """Generate a GitHub-flavored Markdown comparative analysis table."""
    lines = [
        "# Research Evaluation: Policy Benchmark Comparison",
        "",
        "| Policy | Trials | Mean Reward +/- Std | Ops Covered +/- Std | Coverage States +/- Std | Planted Bugs (Ground Truth) | Total Requests |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for pol, s in summaries.items():
        lines.append(
            f"| `{pol}` | {s.n_trials} | {s.mean_reward:.3f} +/- {s.std_reward:.3f} | "
            f"{s.mean_ops_covered:.2f} +/- {s.std_ops_covered:.2f} | "
            f"{s.mean_distinct_states:.1f} +/- {s.std_distinct_states:.1f} | "
            f"{s.mean_planted_bugs:.2f} +/- {s.std_planted_bugs:.2f} | "
            f"{s.mean_total_requests:.0f} |"
        )

    lines.append("")
    lines.append("## Statistical Significance vs Baselines")
    lines.append("")
    lines.append("| Comparison | Metric | Vargha-Delaney $\\hat{A}_{12}$ | Magnitude | Mann-Whitney U ($p$) | 95% Bootstrap CI |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    # Comparisons against random and random_budget_matched
    baseline_keys = [k for k in ["random", "random_budget_matched"] if k in summaries]
    treatment_keys = [k for k in ["greedy", "bandit", "ppo", "ppo_llm"] if k in summaries]

    for b_key in baseline_keys:
        b_summary = summaries[b_key]
        for t_key in treatment_keys:
            t_summary = summaries[t_key]

            # Metric 1: Coverage States
            a12_states = vargha_delaney_a12(t_summary.raw_distinct_states, b_summary.raw_distinct_states)
            _, p_states = mann_whitney_u_test(t_summary.raw_distinct_states, b_summary.raw_distinct_states)
            diff_states = [
                ti - bi for ti, bi in zip(t_summary.raw_distinct_states, b_summary.raw_distinct_states)
            ]
            ci_s_low, ci_s_high = bootstrap_mean_ci(diff_states) if diff_states else (0.0, 0.0)

            lines.append(
                f"| `{t_key}` vs `{b_key}` | Coverage States | {a12_states:.3f} | {interpret_a12(a12_states)} | $p={p_states:.4f}$ | [{ci_s_low:+.1f}, {ci_s_high:+.1f}] |"
            )

            # Metric 2: Planted Bugs
            a12_bugs = vargha_delaney_a12(t_summary.raw_planted_bugs, b_summary.raw_planted_bugs)
            _, p_bugs = mann_whitney_u_test(t_summary.raw_planted_bugs, b_summary.raw_planted_bugs)
            diff_bugs = [
                ti - bi for ti, bi in zip(t_summary.raw_planted_bugs, b_summary.raw_planted_bugs)
            ]
            ci_b_low, ci_b_high = bootstrap_mean_ci(diff_bugs) if diff_bugs else (0.0, 0.0)

            lines.append(
                f"| `{t_key}` vs `{b_key}` | Planted Bugs | {a12_bugs:.3f} | {interpret_a12(a12_bugs)} | $p={p_bugs:.4f}$ | [{ci_b_low:+.2f}, {ci_b_high:+.2f}] |"
            )

    return "\n".join(lines)
