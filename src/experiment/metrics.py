"""
Phase 6: Evaluation metrics for comparing policies.
Provides fair, multi-trial aggregation of coverage, anomalies, and reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class PolicyMetrics:
    """Aggregated metrics for a single policy across multiple trials."""
    policy: str
    n_trials: int
    
    # Reward metrics
    mean_episode_reward: float
    std_episode_reward: float
    mean_total_reward: float
    std_total_reward: float
    
    # Coverage metrics
    mean_unique_ops_covered: float
    std_unique_ops_covered: float
    mean_total_coverage_hits: float
    std_total_coverage_hits: float
    
    # Anomaly discovery metrics
    mean_unique_anomalies: float
    std_unique_anomalies: float

    # Ground-truth planted-bug metrics (defaults ensure backward compatibility)
    mean_planted_bugs: float = 0.0
    std_planted_bugs: float = 0.0
    mean_false_positives: float = 0.0
    std_false_positives: float = 0.0
    mean_steps_to_first_bug: float | None = None
    
    def __str__(self) -> str:
        steps_str = (
            f"{self.mean_steps_to_first_bug:.1f}"
            if self.mean_steps_to_first_bug is not None
            else "N/A"
        )
        return f"""Policy: {self.policy}
  Trials: {self.n_trials}
  Mean episode reward: {self.mean_episode_reward:.3f} ± {self.std_episode_reward:.3f}
  Mean total reward/run: {self.mean_total_reward:.3f} ± {self.std_total_reward:.3f}
  Mean ops covered: {self.mean_unique_ops_covered:.2f} ± {self.std_unique_ops_covered:.2f}
  Mean coverage hits: {self.mean_total_coverage_hits:.1f} ± {self.std_total_coverage_hits:.1f}
  Mean anomalies found: {self.mean_unique_anomalies:.2f} ± {self.std_unique_anomalies:.2f}
  Planted bugs found: {self.mean_planted_bugs:.2f} ± {self.std_planted_bugs:.2f}
  False positives: {self.mean_false_positives:.2f} ± {self.std_false_positives:.2f}
  Mean steps to first bug: {steps_str}"""


def aggregate_metrics(experiment_metrics: list[dict]) -> PolicyMetrics:
    """Aggregate metrics from multiple trials of the same policy."""
    if not experiment_metrics:
        raise ValueError("No experiments to aggregate")
    
    policy = experiment_metrics[0]["policy"]
    
    # Extract arrays for each metric
    episode_rewards = [m["mean_episode_reward"] for m in experiment_metrics]
    total_rewards = [m["total_reward"] for m in experiment_metrics]
    unique_ops = [m["unique_operations_covered"] for m in experiment_metrics]
    total_hits = [m["total_coverage_hits"] for m in experiment_metrics]
    unique_anomalies = [m["unique_anomalies_found"] for m in experiment_metrics]
    planted_bugs = [m.get("planted_bugs_found", 0) for m in experiment_metrics]
    false_positives = [m.get("false_positives", 0) for m in experiment_metrics]
    steps_vals = [
        m["steps_to_first_bug"]
        for m in experiment_metrics
        if m.get("steps_to_first_bug") is not None
    ]
    mean_steps = mean(steps_vals) if steps_vals else None

    # Compute statistics (handle single trial case)
    def safe_stdev(data):
        return stdev(data) if len(data) > 1 else 0.0
    
    return PolicyMetrics(
        policy=policy,
        n_trials=len(experiment_metrics),
        mean_episode_reward=mean(episode_rewards),
        std_episode_reward=safe_stdev(episode_rewards),
        mean_total_reward=mean(total_rewards),
        std_total_reward=safe_stdev(total_rewards),
        mean_unique_ops_covered=mean(unique_ops),
        std_unique_ops_covered=safe_stdev(unique_ops),
        mean_total_coverage_hits=mean(total_hits),
        std_total_coverage_hits=safe_stdev(total_hits),
        mean_unique_anomalies=mean(unique_anomalies),
        std_unique_anomalies=safe_stdev(unique_anomalies),
        mean_planted_bugs=mean(planted_bugs),
        std_planted_bugs=safe_stdev(planted_bugs),
        mean_false_positives=mean(false_positives),
        std_false_positives=safe_stdev(false_positives),
        mean_steps_to_first_bug=mean_steps,
    )


def compare_policies(policy_metrics_list: list[PolicyMetrics]) -> str:
    """Format a readable comparison of multiple policies."""
    lines = ["=" * 70]
    lines.append("EXPERIMENT COMPARISON")
    lines.append("=" * 70)
    
    for pm in policy_metrics_list:
        lines.append(str(pm))
        lines.append("")
    
    # Rank by mean_episode_reward
    sorted_policies = sorted(
        policy_metrics_list,
        key=lambda m: m.mean_episode_reward,
        reverse=True
    )
    
    lines.append("=" * 70)
    lines.append("RANKING (by mean episode reward)")
    lines.append("=" * 70)
    for i, pm in enumerate(sorted_policies, 1):
        lines.append(f"{i}. {pm.policy}: {pm.mean_episode_reward:.3f}")
    
    lines.append("=" * 70)
    return "\n".join(lines)
