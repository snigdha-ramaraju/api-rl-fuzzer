"""
Wraps ApiFuzzEnv (unmodified, per Phase 3's design) to implement episode
seeding: at each reset(), with probability `seed_episode_probability`, an
episode is initialized to play out an LLM-proposed scenario's action
sequence instead of the agent's own choices, for exactly the length of
that sequence. After the scripted prefix is exhausted, control reverts to
the agent for the rest of the episode. PPO experiences the resulting
reward exactly like any other rollout -- there is no special-cased reward
path and no change to the action space, only which actions get taken
during the scripted prefix.

This deliberately keeps the LLM off the step-level decision path: the
wrapper only ever consults `scenario_pool.pop_scenario()`, which returns
data already grounded against the real spec (brainstormer/grounding.py).
"""

from __future__ import annotations

import random

import gymnasium as gym

from .api_env import ApiFuzzEnv


class ScenarioSeededEnv(gym.Wrapper):
    def __init__(self, env: ApiFuzzEnv, scenario_pool, seed_episode_probability: float = 0.15,
                 rng: random.Random | None = None):
        super().__init__(env)
        self.scenario_pool = scenario_pool
        self.seed_episode_probability = seed_episode_probability
        self.rng = rng or random.Random()

        self._scripted_actions: list[int] = []
        self._scripted_pos = 0
        self._current_scenario = None

    def _coverage_summary(self) -> dict:
        return {
            "covered_endpoints": int(sum(self.env.coverage.coverage_vector())),
            "total_endpoints": len(self.env.endpoints),
            "distinct_status_count": self.env.coverage.distinct_status_count(),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Periodic brainstorm cadence lives in the pool; this just triggers
        # the per-episode check. A no-op most of the time (see pool.py).
        self.scenario_pool.maybe_refresh(self._coverage_summary())

        self._scripted_actions = []
        self._scripted_pos = 0
        self._current_scenario = None

        if self.rng.random() < self.seed_episode_probability:
            scenario = self.scenario_pool.pop_scenario()
            if scenario is not None:
                op_to_idx = {e.operation_id: i for i, e in enumerate(self.env.endpoints)}
                actions = [op_to_idx[step.operation_id] for step in scenario.sequence
                           if step.operation_id in op_to_idx]
                if actions:
                    self._scripted_actions = actions
                    self._current_scenario = scenario

        info = dict(info)
        info["scenario_seeded"] = self._current_scenario is not None
        info["scenario_tag"] = self._current_scenario.tag if self._current_scenario else None
        return obs, info

    def step(self, action):
        scripted_step = self._scripted_pos < len(self._scripted_actions)
        if scripted_step:
            action = self._scripted_actions[self._scripted_pos]
            self._scripted_pos += 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["scenario_seeded_step"] = scripted_step
        return obs, reward, terminated, truncated, info
