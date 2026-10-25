"""
Intelligent baseline: Coverage-Greedy policy.

Greedily prioritizes uncovered endpoints in the current episode based on
the environment observation vector. When all endpoints are covered within
the episode, falls back to the least-hit endpoints to maximize diversity.
"""

from __future__ import annotations

import random
import numpy as np


class CoverageGreedyPolicy:
    """
    A heuristic fuzzer policy that prioritizes uncovered endpoints.
    """

    def __init__(self, n_actions: int, seed: int | None = None):
        self.n_actions = n_actions
        self.rng = random.Random(seed)
        self.hit_counts: dict[int, int] = {i: 0 for i in range(n_actions)}

    def predict(self, observation: np.ndarray | None = None, deterministic: bool = True) -> tuple[int, None]:
        if observation is not None and len(observation) >= self.n_actions:
            # Observation format: [cov_0, cov_1, ..., cov_{n-1}, progress]
            cov = observation[:self.n_actions]
            uncovered = [i for i, c in enumerate(cov) if c < 0.5]
            if uncovered:
                # Prioritize uncovered endpoints with tie-breaking on total historical hits
                min_hits = min(self.hit_counts[i] for i in uncovered)
                candidates = [i for i in uncovered if self.hit_counts[i] == min_hits]
                action = self.rng.choice(candidates)
                self.hit_counts[action] += 1
                return action, None

        # All covered in this episode: choose historically least-hit endpoint
        min_hits = min(self.hit_counts.values()) if self.hit_counts else 0
        candidates = [i for i, count in self.hit_counts.items() if count == min_hits]
        action = self.rng.choice(candidates) if candidates else self.rng.randint(0, self.n_actions - 1)
        self.hit_counts[action] += 1
        return action, None

    def update(self, action: int, reward: float) -> None:
        """Optional reward update (tracked via hit_counts in predict)."""
        pass

    def set_random_seed(self, seed: int) -> None:
        self.rng.seed(seed)
