"""
Gymnasium environment wrapping a target API for RL-driven exploration.
Wires together everything built so far:
  - spec/parser.py       -> action space (one action per endpoint)
  - spec/graph.py         -> dependency-aware value resolution across a rollout
  - spec/oracle.py        -> anomaly signal feeding the reward
  - corpus/*               -> seed/fallback value resolution (Phase 2)
  - executor/request_runner.py -> real HTTP execution with budget/rate limit

No LLM involvement here -- this environment is fully usable standalone.
The mandatory local-only LLM brainstormer (Phase 5) only ever *seeds*
episodes later; it never participates in stepping and this module has
no dependency on it.
"""

from __future__ import annotations

import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from spec.parser import Endpoint
from spec.graph import build_dependency_graph
from spec.oracle import check_response
from corpus.seed_dict import SeedCorpus
from executor.request_runner import RequestRunner, BudgetExceededError
from .action_space import build_action_space, build_dependency_index, resolve_request
from .coverage import CoverageTracker
from .reward import compute_reward


class ApiFuzzEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, endpoints: list[Endpoint], corpus: SeedCorpus,
                 runner: RequestRunner, max_steps_per_episode: int = 50,
                 seed_probability: float = 0.7, seed: int | None = None):
        super().__init__()
        self.endpoints = build_action_space(endpoints)
        self.corpus = corpus
        self.runner = runner
        self.max_steps = max_steps_per_episode
        self.seed_probability = seed_probability
        self.rng = random.Random(seed)

        edges, self.unresolved = build_dependency_graph(endpoints)
        self.dependency_index = build_dependency_index(edges)

        self.coverage = CoverageTracker(endpoint_ids=[e.operation_id for e in self.endpoints])
        self.known_values: dict[tuple[str, str], object] = {}
        self._step_count = 0

        self.action_space = spaces.Discrete(len(self.endpoints))
        # observation: one coverage flag per endpoint + normalized episode progress
        obs_len = len(self.endpoints) + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

    def _observation(self) -> np.ndarray:
        cov = self.coverage.coverage_vector()
        progress = [self._step_count / self.max_steps]
        return np.array(cov + progress, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        self.coverage.reset()
        self.known_values.clear()
        self._step_count = 0
        return self._observation(), {}

    def step(self, action: int):
        endpoint = self.endpoints[action]
        resolved = resolve_request(
            endpoint, self.corpus, self.known_values, self.dependency_index,
            self.rng, self.seed_probability,
        )

        try:
            result = self.runner.execute(
                endpoint, resolved.path_values, resolved.query_values,
                resolved.header_values, resolved.body_values,
            )
        except BudgetExceededError:
            self._step_count = self.max_steps
            return self._observation(), 0.0, False, True, {"budget_exceeded": True}

        transport_error = result.error is not None
        oracle_result = None
        status_for_coverage = -1 if transport_error else result.status_code

        if not transport_error:
            oracle_result = check_response(endpoint, result.status_code, result.body)
            # feed this call's response fields into known_values so LATER
            # steps this episode can chain off of it via the dependency graph
            if result.body:
                for field_name, value in result.body.items():
                    self.known_values[(endpoint.operation_id, field_name)] = value

        is_new = self.coverage.record(endpoint.operation_id, status_for_coverage)

        is_new_anomaly = False
        if oracle_result is not None and oracle_result.is_anomalous:
            is_new_anomaly = self.coverage.record_anomaly(
                endpoint.operation_id, status_for_coverage, oracle_result
            )

        reward = compute_reward(is_new, oracle_result, resolved.schema_valid,
                                 transport_error, is_new_anomaly=is_new_anomaly)

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "operation_id": endpoint.operation_id,
            "status_code": None if transport_error else result.status_code,
            "is_new_coverage": is_new,
            "anomalous": bool(oracle_result and oracle_result.is_anomalous),
            "is_new_anomaly": is_new_anomaly,
            "schema_valid_request": resolved.schema_valid,
        }
        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self.runner.close()
