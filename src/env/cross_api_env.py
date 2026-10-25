"""
Cross-API Environment and Multi-API Environment for RL Fuzzing.

Provides:
- CrossApiFuzzEnv: A Gymnasium environment with a fixed-size semantic observation
  space (max_slots * 6 + 2) and fixed-size action space Discrete(max_slots).
  Supports strict action masking (inactive slots are masked False and rejected on step).
- MultiApiFuzzEnv: An environment supporting multiple OpenAPI specifications with
  randomized or deterministic API selection across episodes, cleanly resetting all
  API-specific state, coverage, dependency graphs, and known values.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from spec.parser import load_spec, parse_endpoints, Endpoint, Field
from spec.graph import build_dependency_graph, DependencyEdge
from spec.oracle import check_response
from corpus.seed_dict import SeedCorpus, build_seed_corpus, load_sample_inputs
from executor.request_runner import RequestRunner, BudgetExceededError
from executor.mock_target import make_mock_transport
from executor.mock_ecommerce_target import make_ecommerce_mock_transport
from .action_space import build_action_space, build_dependency_index, resolve_request
from .coverage import CoverageTracker
from .reward import compute_reward

DEFAULT_MAX_SLOTS = 64
FEATURES_PER_SLOT = 6
GLOBAL_FEATURES = 2
OBSERVATION_DIM = DEFAULT_MAX_SLOTS * FEATURES_PER_SLOT + GLOBAL_FEATURES  # 386

METHOD_ENCODINGS = {
    "GET": 0.2,
    "POST": 0.4,
    "PUT": 0.6,
    "DELETE": 0.8,
    "PATCH": 1.0,
}


def make_tiny_mock_transport():
    """Mock transport for tiny 2-endpoint auth API (tests/fixtures/tiny_api.yaml)."""
    import httpx

    tokens: set[str] = set()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method.upper()

        if method == "POST" and path == "/auth/token":
            tok = "tok-12345"
            tokens.add(tok)
            return httpx.Response(200, json={"token": tok})

        if method == "GET" and path == "/auth/profile":
            token_param = request.url.params.get("token")
            if token_param in tokens:
                return httpx.Response(200, json={"username": "admin"})
            return httpx.Response(401, json={"error": "unauthorized"})

        return httpx.Response(404, json={"error": "not found"})

    def reset_state():
        tokens.clear()

    transport = httpx.MockTransport(handler)
    transport.reset_state = reset_state
    return transport


def make_crapi_mock_transport():
    """Mock transport for OWASP crAPI benchmark (datasets/openapi/crapi-openapi-spec.json)."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method.upper()
        if "auth" in path or "signup" in path or "login" in path:
            return httpx.Response(200, json={"token": "crapi-mock-jwt-token", "status": "success"})
        return httpx.Response(200, json={"status": "ok", "id": 1, "message": "success"})

    def reset_state():
        pass

    transport = httpx.MockTransport(handler)
    transport.reset_state = reset_state
    return transport


def resolve_mock_transport_for_spec(spec_path: str | Path, use_planted_bugs: bool = False):
    """Factory helper resolving the proper mock transport for known specs."""
    name = Path(spec_path).name.lower()
    if "ecommerce" in name:
        return make_ecommerce_mock_transport(use_planted_bugs=use_planted_bugs)
    elif "tiny" in name:
        return make_tiny_mock_transport()
    elif "crapi" in name:
        return make_crapi_mock_transport()
    else:
        # Default Orders API mock target
        return make_mock_transport()


class CrossApiFuzzEnv(gym.Env):
    """
    Gymnasium environment wrapping an API with a fixed-size semantic interface.

    Fixed Spaces:
    - Action Space: spaces.Discrete(max_slots)
    - Observation Space: spaces.Box(shape=(max_slots * 6 + 2,))
    - Strict Action Masking: action_masks() provides boolean mask of length max_slots.
    - Never remaps invalid actions: selecting an inactive slot raises ValueError.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        endpoints: list[Endpoint],
        corpus: SeedCorpus,
        runner: RequestRunner,
        max_slots: int = DEFAULT_MAX_SLOTS,
        max_steps_per_episode: int = 50,
        seed_probability: float = 0.7,
        seed: int | None = None,
    ):
        super().__init__()
        self.max_slots = max_slots
        if len(endpoints) > self.max_slots:
            raise ValueError(
                f"API has {len(endpoints)} endpoints, exceeding maximum slot capacity {self.max_slots}."
            )

        self.endpoints = list(endpoints)
        self.corpus = corpus
        self.runner = runner
        self.max_steps = max_steps_per_episode
        self.seed_probability = seed_probability
        self.rng = random.Random(seed)

        edges, self.unresolved = build_dependency_graph(endpoints)
        self.dependency_index = build_dependency_index(edges)
        self.producer_ops = {edge.producer_op for edge in edges}

        self.coverage = CoverageTracker(endpoint_ids=[e.operation_id for e in self.endpoints])
        self.known_values: dict[tuple[str, str], object] = {}
        self._step_count = 0

        self.action_space = spaces.Discrete(self.max_slots)
        obs_len = self.max_slots * FEATURES_PER_SLOT + GLOBAL_FEATURES
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        """Returns boolean mask where True indicates an active endpoint slot."""
        mask = np.zeros(self.max_slots, dtype=bool)
        mask[: len(self.endpoints)] = True
        return mask

    def _dependencies_ready(self, endpoint: Endpoint) -> float:
        """Check if all required input fields have resolved producer values available."""
        required_fields = [f for f in endpoint.all_input_fields if f.required]
        if not required_fields:
            return 1.0

        for f in required_fields:
            dep_edges = self.dependency_index.get((endpoint.operation_id, f.name), [])
            if dep_edges:
                has_value = any((e.producer_op, e.producer_field) in self.known_values for e in dep_edges)
                if not has_value:
                    return 0.0
        return 1.0

    def _observation(self) -> np.ndarray:
        slot_features = []
        n_endpoints = len(self.endpoints)
        covered_count = 0

        for i in range(self.max_slots):
            if i < n_endpoints:
                ep = self.endpoints[i]
                is_active = 1.0
                is_covered = 1.0 if self.coverage.endpoint_covered(ep.operation_id) else 0.0
                if is_covered > 0.5:
                    covered_count += 1
                method_code = METHOD_ENCODINGS.get(ep.method.upper(), 0.1)
                deps_ready = self._dependencies_ready(ep)
                is_producer = 1.0 if ep.operation_id in self.producer_ops else 0.0

                # Compute hit count for this endpoint in current episode
                hits = sum(count for (op, _status), count in self.coverage.hits.items() if op == ep.operation_id)
                hit_freq = min(1.0, hits / max(1, self.max_steps))

                slot_features.extend([is_active, is_covered, method_code, deps_ready, is_producer, hit_freq])
            else:
                # Inactive padding slot
                slot_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        progress = self._step_count / max(1, self.max_steps)
        coverage_ratio = covered_count / max(1, n_endpoints)
        global_features = [progress, coverage_ratio]

        return np.array(slot_features + global_features, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None, preserve_state: bool = False):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        if not preserve_state:
            self.coverage.reset()
            self.known_values.clear()
        self._step_count = 0
        if not preserve_state and hasattr(self.runner, "reset_state"):
            self.runner.reset_state()
        return self._observation(), {"action_mask": self.action_masks()}

    def step(self, action: int):
        # Strict validation: Never remap invalid actions!
        if action < 0 or action >= len(self.endpoints):
            raise ValueError(
                f"Action {action} is inactive. Valid action range for current API "
                f"({len(self.endpoints)} endpoints) is 0..{len(self.endpoints) - 1}. "
                f"Never remap invalid actions."
            )

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
            return self._observation(), 0.0, False, True, {
                "budget_exceeded": True,
                "action_mask": self.action_masks(),
            }

        transport_error = result.error is not None
        oracle_result = None
        status_for_coverage = -1 if transport_error else result.status_code

        if not transport_error:
            oracle_result = check_response(endpoint, result.status_code, result.body)
            if result.body and isinstance(result.body, dict):
                for field_name, value in result.body.items():
                    self.known_values[(endpoint.operation_id, field_name)] = value

        is_new = self.coverage.record(endpoint.operation_id, status_for_coverage)

        is_new_anomaly = False
        anomaly_signature = None
        if oracle_result is not None and oracle_result.is_anomalous:
            is_new_anomaly = self.coverage.record_anomaly(
                endpoint.operation_id, status_for_coverage, oracle_result
            )
            anomaly_signature = self.coverage.anomaly_signature(
                endpoint.operation_id, status_for_coverage, oracle_result
            )

        reward = compute_reward(
            is_new, oracle_result, resolved.schema_valid,
            transport_error, is_new_anomaly=is_new_anomaly,
        )

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "operation_id": endpoint.operation_id,
            "status_code": None if transport_error else result.status_code,
            "is_new_coverage": is_new,
            "anomalous": bool(oracle_result and oracle_result.is_anomalous),
            "is_new_anomaly": is_new_anomaly,
            "anomaly_signature": anomaly_signature,
            "schema_valid_request": resolved.schema_valid,
            "action_mask": self.action_masks(),
        }
        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self.runner.close()


class MultiApiFuzzEnv(gym.Env):
    """
    Multi-API Environment supporting a collection of OpenAPI specifications.

    Between episodes (on reset), an API is selected either randomly or via options['spec_idx'].
    All API-specific state (endpoints, corpus, dependencies, coverage, known values,
    and mock transports) are strictly isolated and reset when switching APIs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        spec_paths: list[str | Path],
        max_slots: int = DEFAULT_MAX_SLOTS,
        max_steps_per_episode: int = 50,
        seed_probability: float = 0.7,
        seed: int | None = None,
        use_mock: bool = True,
        use_planted_bugs: bool = False,
    ):
        super().__init__()
        if not spec_paths:
            raise ValueError("spec_paths must contain at least one specification.")

        self.max_slots = max_slots
        self.max_steps = max_steps_per_episode
        self.seed_probability = seed_probability
        self.rng = random.Random(seed)
        self.use_mock = use_mock
        self.use_planted_bugs = use_planted_bugs

        # Pre-initialize configs for all specifications
        self.api_configs: list[dict[str, Any]] = []
        for path in spec_paths:
            p = Path(path)
            spec = load_spec(p)
            endpoints = parse_endpoints(spec)
            if len(endpoints) > self.max_slots:
                raise ValueError(
                    f"Specification {p.name} has {len(endpoints)} endpoints, "
                    f"exceeding max_slots ({self.max_slots})."
                )
            corpus = build_seed_corpus(endpoints)
            edges, unresolved = build_dependency_graph(endpoints)
            dep_index = build_dependency_index(edges)
            producer_ops = {e.producer_op for e in edges}

            transport = resolve_mock_transport_for_spec(p, use_planted_bugs=self.use_planted_bugs)
            runner = RequestRunner(
                base_url="http://mock",
                rate_limit_per_sec=0,
                max_requests_per_run=10_000_000,
                transport=transport,
            )

            self.api_configs.append({
                "spec_path": p,
                "endpoints": endpoints,
                "corpus": corpus,
                "edges": edges,
                "dependency_index": dep_index,
                "producer_ops": producer_ops,
                "runner": runner,
            })

        # Set default active API to the first one
        self.current_idx = 0
        self._activate_api(0)

        # Observation and Action spaces are constant across all APIs
        self.action_space = spaces.Discrete(self.max_slots)
        obs_len = self.max_slots * FEATURES_PER_SLOT + GLOBAL_FEATURES
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

    def _activate_api(self, idx: int) -> None:
        """Activate and cleanly reset state for the chosen API index."""
        self.current_idx = idx
        cfg = self.api_configs[idx]
        self.current_spec_path = cfg["spec_path"]
        self.endpoints = cfg["endpoints"]
        self.corpus = cfg["corpus"]
        self.dependency_index = cfg["dependency_index"]
        self.producer_ops = cfg["producer_ops"]
        self.runner = cfg["runner"]

        self.coverage = CoverageTracker(endpoint_ids=[e.operation_id for e in self.endpoints])
        self.known_values = {}
        self._step_count = 0
        if hasattr(self.runner, "reset_state"):
            self.runner.reset_state()

    def action_masks(self) -> np.ndarray:
        """Returns boolean mask for the currently active API."""
        mask = np.zeros(self.max_slots, dtype=bool)
        mask[: len(self.endpoints)] = True
        return mask

    def _dependencies_ready(self, endpoint: Endpoint) -> float:
        required_fields = [f for f in endpoint.all_input_fields if f.required]
        if not required_fields:
            return 1.0

        for f in required_fields:
            dep_edges = self.dependency_index.get((endpoint.operation_id, f.name), [])
            if dep_edges:
                has_value = any((e.producer_op, e.producer_field) in self.known_values for e in dep_edges)
                if not has_value:
                    return 0.0
        return 1.0

    def _observation(self) -> np.ndarray:
        slot_features = []
        n_endpoints = len(self.endpoints)
        covered_count = 0

        for i in range(self.max_slots):
            if i < n_endpoints:
                ep = self.endpoints[i]
                is_active = 1.0
                is_covered = 1.0 if self.coverage.endpoint_covered(ep.operation_id) else 0.0
                if is_covered > 0.5:
                    covered_count += 1
                method_code = METHOD_ENCODINGS.get(ep.method.upper(), 0.1)
                deps_ready = self._dependencies_ready(ep)
                is_producer = 1.0 if ep.operation_id in self.producer_ops else 0.0

                hits = sum(count for (op, _status), count in self.coverage.hits.items() if op == ep.operation_id)
                hit_freq = min(1.0, hits / max(1, self.max_steps))

                slot_features.extend([is_active, is_covered, method_code, deps_ready, is_producer, hit_freq])
            else:
                slot_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        progress = self._step_count / max(1, self.max_steps)
        coverage_ratio = covered_count / max(1, n_endpoints)
        global_features = [progress, coverage_ratio]

        return np.array(slot_features + global_features, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)

        # Select API: either forced via options or randomly selected
        if options and "spec_idx" in options:
            chosen_idx = int(options["spec_idx"]) % len(self.api_configs)
        else:
            chosen_idx = self.rng.randint(0, len(self.api_configs) - 1)

        self._activate_api(chosen_idx)

        return self._observation(), {
            "spec_path": str(self.current_spec_path),
            "n_endpoints": len(self.endpoints),
            "action_mask": self.action_masks(),
        }

    def step(self, action: int):
        # Strict validation: Never remap invalid actions!
        if action < 0 or action >= len(self.endpoints):
            raise ValueError(
                f"Action {action} is inactive. Valid action range for current API "
                f"'{self.current_spec_path.name}' ({len(self.endpoints)} endpoints) "
                f"is 0..{len(self.endpoints) - 1}. Never remap invalid actions."
            )

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
            return self._observation(), 0.0, False, True, {
                "budget_exceeded": True,
                "action_mask": self.action_masks(),
            }

        transport_error = result.error is not None
        oracle_result = None
        status_for_coverage = -1 if transport_error else result.status_code

        if not transport_error:
            oracle_result = check_response(endpoint, result.status_code, result.body)
            if result.body and isinstance(result.body, dict):
                for field_name, value in result.body.items():
                    self.known_values[(endpoint.operation_id, field_name)] = value

        is_new = self.coverage.record(endpoint.operation_id, status_for_coverage)

        is_new_anomaly = False
        anomaly_signature = None
        if oracle_result is not None and oracle_result.is_anomalous:
            is_new_anomaly = self.coverage.record_anomaly(
                endpoint.operation_id, status_for_coverage, oracle_result
            )
            anomaly_signature = self.coverage.anomaly_signature(
                endpoint.operation_id, status_for_coverage, oracle_result
            )

        reward = compute_reward(
            is_new, oracle_result, resolved.schema_valid,
            transport_error, is_new_anomaly=is_new_anomaly,
        )

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps

        info = {
            "operation_id": endpoint.operation_id,
            "status_code": None if transport_error else result.status_code,
            "is_new_coverage": is_new,
            "anomalous": bool(oracle_result and oracle_result.is_anomalous),
            "is_new_anomaly": is_new_anomaly,
            "anomaly_signature": anomaly_signature,
            "schema_valid_request": resolved.schema_valid,
            "action_mask": self.action_masks(),
            "spec_path": str(self.current_spec_path),
        }
        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        for cfg in self.api_configs:
            cfg["runner"].close()
