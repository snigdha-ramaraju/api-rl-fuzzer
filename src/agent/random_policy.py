"""
Phase 6: Random baseline policy.
A simple, fair baseline that takes uniformly random actions.
Used to evaluate improvement of learned policies (PPO, PPO+LLM).
"""

from __future__ import annotations

import random


class RandomPolicy:
    """
    A random action policy that samples uniformly from the action space.
    Implements a minimal interface compatible with the experiment runner.
    """

    def __init__(self, n_actions: int, seed: int | None = None):
        """
        Args:
            n_actions: Number of possible actions (endpoints).
            seed: RNG seed for reproducibility.
        """
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    def predict(self, observation, deterministic: bool = True) -> tuple[int, None]:
        """
        Sample a random action.
        Args:
            observation: Ignored (random policy doesn't use observation).
            deterministic: Ignored (randomness is inherent to this policy).

        Returns:
            (action, state) where action is a uniform random int in [0, n_actions).
        """
        action = self.rng.randint(0, self.n_actions - 1)
        return action, None

    def set_random_seed(self, seed: int):
        """Reset the RNG with a new seed."""
        self.rng.seed(seed)
