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
import numpy as np

from .action_space import resolve_request
from .api_env import ApiFuzzEnv
from executor.request_runner import BudgetExceededError


class ScenarioSeededEnv(gym.Wrapper):
    def __init__(self, env: ApiFuzzEnv, scenario_pool, seed_episode_probability: float = 0.15,
                 rng: random.Random | None = None, prepare_at_reset: bool = False):
        super().__init__(env)
        self.scenario_pool = scenario_pool
        self.seed_episode_probability = seed_episode_probability
        self.rng = rng or random.Random()
        self.prepare_at_reset = prepare_at_reset

        self._scripted_actions: list[int] = []
        self._scripted_pos = 0
        self._current_scenario = None
        self._prepared_ops: set[str] = set()
        self._prep_requests_count = 0
        self._seeded_episodes_count = 0

    @property
    def prep_request_count(self) -> int:
        return self._prep_requests_count

    @property
    def seeded_episodes(self) -> int:
        return self._seeded_episodes_count

    @property
    def brainstorm_calls(self) -> int:
        return getattr(self.scenario_pool, "brainstorm_calls", 0)

    def _coverage_summary(self) -> dict:
        return {
            "covered_endpoints": int(sum(self.env.coverage.coverage_vector())),
            "total_endpoints": len(self.env.endpoints),
            "distinct_status_count": self.env.coverage.distinct_status_count(),
        }

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = random.Random(seed)

        obs, info = self.env.reset(**kwargs)

        # Periodic brainstorm cadence lives in the pool; this just triggers
        # the per-episode check. A no-op most of the time (see pool.py).
        self.scenario_pool.maybe_refresh(self._coverage_summary())

        self._scripted_actions = []
        self._scripted_pos = 0
        self._current_scenario = None
        self._prepared_ops = set()

        if self.rng.random() < self.seed_episode_probability:
            scenario = self.scenario_pool.pop_scenario()
            if scenario is not None:
                self._current_scenario = scenario
                self._seeded_episodes_count += 1
                op_to_idx = {e.operation_id: i for i, e in enumerate(self.env.endpoints)}
                actions = [op_to_idx[step.operation_id] for step in scenario.sequence
                           if step.operation_id in op_to_idx]
                if actions:
                    self._scripted_actions = actions

                if self.prepare_at_reset:
                    # Execute scenario sequence during reset to prime target state
                    # and known_values without overriding PPO training actions in step().
                    op_to_endpoint = {e.operation_id: e for e in self.env.endpoints}
                    for step in scenario.sequence:
                        endpoint = op_to_endpoint.get(step.operation_id)
                        if endpoint is None:
                            continue
                        resolved = resolve_request(
                            endpoint,
                            self.env.corpus,
                            self.env.known_values,
                            self.env.dependency_index,
                            self.rng,
                            self.env.seed_probability,
                        )
                        try:
                            self._prep_requests_count += 1
                            result = self.env.runner.execute(
                                endpoint,
                                resolved.path_values,
                                resolved.query_values,
                                resolved.header_values,
                                resolved.body_values,
                            )
                            if result and result.body and isinstance(result.body, dict):
                                for field_name, value in result.body.items():
                                    self.env.known_values[(endpoint.operation_id, field_name)] = value
                            self._prepared_ops.add(step.operation_id)
                        except BudgetExceededError:
                            break
                        except Exception:
                            pass

        info = dict(info)
        info["scenario_seeded"] = self._current_scenario is not None
        info["scenario_tag"] = self._current_scenario.tag if self._current_scenario else None
        info["prepared_operations"] = sorted(list(self._prepared_ops))

        if self.prepare_at_reset and self._prepared_ops:
            cov = [1.0 if (self.env.coverage.endpoint_covered(e.operation_id) or e.operation_id in self._prepared_ops) else 0.0 for e in self.env.endpoints]
            progress = [self.env._step_count / self.env.max_steps]
            obs = np.array(cov + progress, dtype=np.float32)

        return obs, info

    def step(self, action):
        if not self.prepare_at_reset:
            # Legacy step-interception mode (Phase 5 compatibility)
            scripted_step = self._scripted_pos < len(self._scripted_actions)
            original_action = action
            if scripted_step:
                action = self._scripted_actions[self._scripted_pos]
                self._scripted_pos += 1

            obs, reward, terminated, truncated, info = self.env.step(action)
            info = dict(info)
            info["scenario_seeded_step"] = scripted_step
            info["action_mismatch"] = bool(scripted_step and original_action != action)
            return obs, reward, terminated, truncated, info

        # Reset-preparation mode (100% on-policy PPO control):
        # Action is never replaced; every action executed is the action sampled by PPO.
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["scenario_seeded_step"] = False
        info["action_mismatch"] = False

        if self._prepared_ops:
            cov = [1.0 if (self.env.coverage.endpoint_covered(e.operation_id) or e.operation_id in self._prepared_ops) else 0.0 for e in self.env.endpoints]
            progress = [self.env._step_count / self.env.max_steps]
            obs = np.array(cov + progress, dtype=np.float32)

        return obs, reward, terminated, truncated, info
