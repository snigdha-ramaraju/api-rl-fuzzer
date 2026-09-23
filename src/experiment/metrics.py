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
    
    def __str__(self) -> str:
        return f"""Policy: {self.policy}
  Trials: {self.n_trials}
  Mean episode reward: {self.mean_episode_reward:.3f} ± {self.std_episode_reward:.3f}
  Mean total reward/run: {self.mean_total_reward:.3f} ± {self.std_total_reward:.3f}
  Mean ops covered: {self.mean_unique_ops_covered:.2f} ± {self.std_unique_ops_covered:.2f}
  Mean coverage hits: {self.mean_total_coverage_hits:.1f} ± {self.std_total_coverage_hits:.1f}
  Mean anomalies found: {self.mean_unique_anomalies:.2f} ± {self.std_unique_anomalies:.2f}"""


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
