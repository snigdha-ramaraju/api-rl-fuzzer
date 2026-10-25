"""
Intelligent baseline: Multi-Armed Bandit (UCB1) policy.

Selects API operations based on Upper Confidence Bound (UCB1) balancing
empirical reward exploration and exploitation.
"""

from __future__ import annotations

import math
import random
import numpy as np


class UCB1BanditPolicy:
    """
    Multi-Armed Bandit using UCB1 algorithm for API endpoint selection.

    Formula:
        score(a) = Q(a) + c * sqrt(ln(t) / N(a))
    where:
        Q(a) is the average reward received from arm a.
        N(a) is the pull count for arm a.
        t is the total steps across all arms.
        c is the exploration bonus factor (default sqrt(2)).
    """

    def __init__(
        self,
        n_actions: int,
        c: float = 1.41421356,  # sqrt(2)
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.c = c
        self.rng = random.Random(seed)
        self.counts: list[int] = [0] * n_actions
        self.total_rewards: list[float] = [0.0] * n_actions
        self.total_steps: int = 0

    def predict(
        self,
        observation: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[int, None]:
        # If any action has never been pulled, pull it first
        unpulled = [i for i, cnt in enumerate(self.counts) if cnt == 0]
        if unpulled:
            action = self.rng.choice(unpulled)
            return action, None

        # UCB1 score calculation
        log_total = math.log(max(1, self.total_steps))
        best_score = -float("inf")
        best_actions = []

        for i in range(self.n_actions):
            mean_reward = self.total_rewards[i] / self.counts[i]
            exploration = self.c * math.sqrt(log_total / self.counts[i])
            score = mean_reward + exploration

            if score > best_score + 1e-9:
                best_score = score
                best_actions = [i]
            elif abs(score - best_score) <= 1e-9:
                best_actions.append(i)

        action = self.rng.choice(best_actions)
        return action, None

    def update(self, action: int, reward: float) -> None:
        """Update empirical reward statistics after step execution."""
        if 0 <= action < self.n_actions:
            self.counts[action] += 1
            self.total_rewards[action] += reward
            self.total_steps += 1

    def set_random_seed(self, seed: int) -> None:
        self.rng.seed(seed)
