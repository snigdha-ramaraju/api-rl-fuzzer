import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from spec.parser import load_spec, parse_endpoints, Field
from spec.graph import build_dependency_graph
from spec.oracle import OracleResult
from corpus.seed_dict import SeedCorpus, build_seed_corpus
from executor.request_runner import RequestRunner, BudgetExceededError
from executor.config import load_target_config
from env.coverage import CoverageTracker
from env.reward import compute_reward
from env.action_space import build_dependency_index, resolve_request
from env.api_env import ApiFuzzEnv

FIXTURE_SPEC = Path(__file__).parent / "fixtures" / "sample_api.yaml"


def _endpoints():
    spec = load_spec(FIXTURE_SPEC)
    return parse_endpoints(spec)


def _op(endpoints, op_id):
    return next(e for e in endpoints if e.operation_id == op_id)


def make_mock_transport():
    """Simulates the sample Orders API in-memory. No network, no live server."""
    store = {}
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        if method == "POST" and path == "/orders":
            counter["n"] += 1
            oid = f"ord-{counter['n']}"
            store[oid] = "created"
            return httpx.Response(201, json={"id": oid, "status": "created"})

        if method == "GET" and path.startswith("/orders/"):
            oid = path.rsplit("/", 1)[-1]
            if oid in store:
                return httpx.Response(200, json={"id": oid, "status": store[oid]})
            return httpx.Response(404, json={"error": "not found"})

        if method == "POST" and path.startswith("/orders/"):
            oid = path.rsplit("/", 1)[-1]
            if oid in store:
                store[oid] = "cancelled"
                return httpx.Response(200, json={"id": oid, "status": "cancelled"})
            return httpx.Response(404, json={"error": "not found"})

        return httpx.Response(404, json={"error": "unmatched route"})

    return httpx.MockTransport(handler)


# ---------- executor ----------

def test_request_runner_executes_create_and_get():
    endpoints = _endpoints()
    runner = RequestRunner(base_url="http://mock", rate_limit_per_sec=0,
                            max_requests_per_run=100, transport=make_mock_transport())

    create = _op(endpoints, "createOrder")
    result = runner.execute(create, {}, {}, {}, {"item_id": "sku-1", "quantity": 2})
    assert result.status_code == 201
    assert result.body["status"] == "created"
    oid = result.body["id"]

    get = _op(endpoints, "getOrder")
    result2 = runner.execute(get, {"orderId": oid}, {}, {}, {})
    assert result2.status_code == 200
    assert result2.body["id"] == oid
    runner.close()


def test_request_runner_enforces_budget():
    endpoints = _endpoints()
    runner = RequestRunner(base_url="http://mock", rate_limit_per_sec=0,
                            max_requests_per_run=1, transport=make_mock_transport())
    create = _op(endpoints, "createOrder")

    runner.execute(create, {}, {}, {}, {"item_id": "x", "quantity": 1})
    try:
        runner.execute(create, {}, {}, {}, {"item_id": "y", "quantity": 1})
        assert False, "should have raised BudgetExceededError"
    except BudgetExceededError:
        pass
    runner.close()


def test_load_target_config():
    cfg = load_target_config(Path(__file__).resolve().parents[1] / "config" / "target.yaml")
    assert cfg.base_url == "http://localhost:8080"
    assert cfg.max_requests_per_run == 5000
    assert cfg.rate_limit_per_sec == 5
    assert cfg.auth_header is None


# ---------- coverage ----------

def test_coverage_tracker_new_vs_repeat():
    tracker = CoverageTracker(endpoint_ids=["createOrder", "getOrder"])
    assert tracker.record("createOrder", 201) is True
    assert tracker.record("createOrder", 201) is False  # repeat
    assert tracker.record("createOrder", 400) is True     # new status for same op
    assert tracker.endpoint_covered("getOrder") is False
    assert tracker.endpoint_covered("createOrder") is True
    vec = tracker.coverage_vector()
    assert vec == [1.0, 0.0]


def _anomaly(status_known=False, shape_valid=None, missing=None, bad_types=None) -> OracleResult:
    return OracleResult(status_known=status_known, shape_valid=shape_valid,
                         missing_fields=missing or [], unexpected_type_fields=bad_types or [])


def test_coverage_tracker_record_anomaly_gates_on_exact_signature():
    """Reward-hacking regression: the SAME anomaly (same op, status, and
    reason) must only be reported novel once per episode."""
    tracker = CoverageTracker(endpoint_ids=["getOrder"])
    undeclared_404 = _anomaly(status_known=False, shape_valid=None)

    assert tracker.record_anomaly("getOrder", 404, undeclared_404) is True
    # Identical anomaly, repeated arbitrarily many times -> never novel again.
    for _ in range(5):
        assert tracker.record_anomaly("getOrder", 404, undeclared_404) is False


def test_coverage_tracker_record_anomaly_distinguishes_different_anomaly_reasons():
    """A genuinely different anomaly on the SAME (op, status) pair -- e.g.
    a different missing field -- is still novel, even though plain (op,
    status) coverage would already consider that pair \"seen\"."""
    tracker = CoverageTracker(endpoint_ids=["createOrder"])
    missing_status = _anomaly(status_known=True, shape_valid=False, missing=["status"])
    missing_id = _anomaly(status_known=True, shape_valid=False, missing=["id"])

    assert tracker.record_anomaly("createOrder", 201, missing_status) is True
    assert tracker.record_anomaly("createOrder", 201, missing_status) is False  # same reason again
    assert tracker.record_anomaly("createOrder", 201, missing_id) is True       # different reason


def test_coverage_tracker_reset_clears_anomaly_signatures_too():
    tracker = CoverageTracker(endpoint_ids=["getOrder"])
    undeclared_404 = _anomaly()
    assert tracker.record_anomaly("getOrder", 404, undeclared_404) is True
    tracker.reset()
    # A fresh episode should be able to earn the anomaly-novelty bonus again.
    assert tracker.record_anomaly("getOrder", 404, undeclared_404) is True


# ---------- reward ----------

def test_reward_new_coverage_bonus():
    r = compute_reward(is_new_coverage=True, oracle_result=None,
                        request_was_schema_valid=True, transport_error=False)
    assert r == 1.0


def test_reward_transport_error_short_circuits():
    r = compute_reward(is_new_coverage=True, oracle_result=None,
                        request_was_schema_valid=True, transport_error=True)
    assert r == -0.2


def test_reward_schema_invalid_penalty_stacks_with_coverage():
    r = compute_reward(is_new_coverage=True, oracle_result=None,
                        request_was_schema_valid=False, transport_error=False)
    assert r == 1.0 - 0.5


def test_reward_anomaly_bonus_paid_when_novel():
    anomalous = _anomaly()
    r = compute_reward(is_new_coverage=True, oracle_result=anomalous,
                        request_was_schema_valid=True, transport_error=False,
                        is_new_anomaly=True)
    assert r == 1.0 + 2.0


def test_reward_anomaly_bonus_withheld_when_not_novel():
    """The core reward-hacking fix: an anomalous response that has already
    been seen this episode (is_new_anomaly=False) must NOT pay the anomaly
    bonus again, even though oracle_result.is_anomalous is still True."""
    anomalous = _anomaly()
    r = compute_reward(is_new_coverage=False, oracle_result=anomalous,
                        request_was_schema_valid=True, transport_error=False,
                        is_new_anomaly=False)
    assert r == 0.0  # no coverage bonus (not new), no anomaly bonus (not novel)


def test_reward_anomaly_gate_has_no_effect_when_not_anomalous():
    """is_new_anomaly is irrelevant when the response isn't anomalous at
    all -- confirms the default value is safe for non-anomalous call sites
    (e.g. the three tests above that never mention it)."""
    r_true = compute_reward(is_new_coverage=True, oracle_result=None,
                             request_was_schema_valid=True, transport_error=False,
                             is_new_anomaly=True)
    r_false = compute_reward(is_new_coverage=True, oracle_result=None,
                              request_was_schema_valid=True, transport_error=False,
                              is_new_anomaly=False)
    assert r_true == r_false == 1.0


# ---------- action space / dependency resolution ----------

def test_resolve_request_prefers_dependency_value_over_corpus():
    endpoints = _endpoints()
    edges, _ = build_dependency_graph(endpoints)
    dep_index = build_dependency_index(edges)
    corpus = SeedCorpus()  # empty on purpose -- dependency value must win

    known_values = {("createOrder", "id"): "known-order-id"}
    get_order = _op(endpoints, "getOrder")

    resolved = resolve_request(get_order, corpus, known_values, dep_index,
                                rng=random.Random(0), seed_probability=1.0)
    assert resolved.path_values["orderId"] == "known-order-id"


def test_resolve_request_falls_back_when_no_dependency_value_known():
    endpoints = _endpoints()
    edges, _ = build_dependency_graph(endpoints)
    dep_index = build_dependency_index(edges)
    corpus = SeedCorpus()

    get_order = _op(endpoints, "getOrder")
    resolved = resolve_request(get_order, corpus, {}, dep_index,
                                rng=random.Random(0), seed_probability=1.0)
    assert isinstance(resolved.path_values["orderId"], str)
    assert resolved.schema_valid is True


# ---------- full environment ----------

def _make_env(max_steps=10, seed_probability=1.0):
    endpoints = _endpoints()
    corpus = build_seed_corpus(endpoints)
    runner = RequestRunner(base_url="http://mock", rate_limit_per_sec=0,
                            max_requests_per_run=1000, transport=make_mock_transport())
    return ApiFuzzEnv(endpoints, corpus, runner, max_steps_per_episode=max_steps,
                       seed_probability=seed_probability, seed=42)


def test_env_spaces_shape():
    env = _make_env()
    assert env.action_space.n == 3  # createOrder, getOrder, cancelOrder
    assert env.observation_space.shape == (4,)  # 3 endpoints + progress scalar
    env.close()


def test_env_reset_returns_zero_coverage_observation():
    env = _make_env()
    obs, info = env.reset(seed=1)
    assert obs.shape == (4,)
    assert obs[:3].sum() == 0.0  # nothing covered yet
    env.close()


def test_env_chains_create_then_get_via_dependency_graph():
    """The key Phase 3 integration test: calling createOrder then getOrder
    should hit the SAME order id via the dependency graph, resulting in a
    200 (found), not a 404 (which would happen with an unrelated random id)."""
    env = _make_env(seed_probability=1.0)
    env.reset(seed=1)

    create_idx = next(i for i, e in enumerate(env.endpoints) if e.operation_id == "createOrder")
    get_idx = next(i for i, e in enumerate(env.endpoints) if e.operation_id == "getOrder")

    obs, reward, terminated, truncated, info = env.step(create_idx)
    assert info["status_code"] == 201
    assert info["is_new_coverage"] is True

    obs, reward, terminated, truncated, info = env.step(get_idx)
    assert info["status_code"] == 200          # proves dependency-graph chaining worked
    assert info["anomalous"] is False           # 200 is a declared status for getOrder
    env.close()


def test_env_getorder_without_prior_create_is_anomalous():
    """Calling getOrder cold (no known order id) hits a 404, which the spec
    never declared for this endpoint -- the oracle should flag it, and the
    reward should include the anomaly bonus."""
    env = _make_env(seed_probability=1.0)
    env.reset(seed=1)

    get_idx = next(i for i, e in enumerate(env.endpoints) if e.operation_id == "getOrder")
    obs, reward, terminated, truncated, info = env.step(get_idx)

    assert info["status_code"] == 404
    assert info["anomalous"] is True
    assert info["is_new_anomaly"] is True
    assert reward >= 1.0 + 2.0  # new-coverage bonus + anomaly bonus, at least
    env.close()


def test_env_repeating_the_identical_anomaly_does_not_farm_unlimited_reward():
    """Reward-hacking regression test, full stack: calling the same
    endpoint cold over and over produces the exact same undeclared-404
    anomaly every time. Only the FIRST occurrence may earn the anomaly
    bonus (or the coverage bonus); every repeat must earn nothing, so
    total episode reward from this alone stays bounded rather than
    growing without limit with the number of repeats."""
    env = _make_env(max_steps=6, seed_probability=1.0)
    env.reset(seed=1)
    get_idx = next(i for i, e in enumerate(env.endpoints) if e.operation_id == "getOrder")

    obs, first_reward, terminated, truncated, info = env.step(get_idx)
    assert info["status_code"] == 404
    assert info["is_new_coverage"] is True
    assert info["is_new_anomaly"] is True
    assert first_reward == 1.0 + 2.0

    repeat_rewards = []
    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(get_idx)
        assert info["status_code"] == 404
        assert info["anomalous"] is True          # still structurally anomalous...
        assert info["is_new_anomaly"] is False    # ...but no longer NOVEL...
        assert info["is_new_coverage"] is False
        repeat_rewards.append(reward)

    # ...so every repeat earns nothing further: the exploit is closed.
    assert repeat_rewards == [0.0, 0.0, 0.0, 0.0]
    env.close()


def test_env_truncates_after_max_steps():
    env = _make_env(max_steps=3, seed_probability=1.0)
    env.reset(seed=1)
    create_idx = next(i for i, e in enumerate(env.endpoints) if e.operation_id == "createOrder")

    truncated = False
    for _ in range(3):
        _, _, _, truncated, _ = env.step(create_idx)
    assert truncated is True
    env.close()


def test_env_budget_exceeded_ends_episode_gracefully():
    endpoints = _endpoints()
    corpus = build_seed_corpus(endpoints)
    runner = RequestRunner(base_url="http://mock", rate_limit_per_sec=0,
                            max_requests_per_run=1, transport=make_mock_transport())
    env = ApiFuzzEnv(endpoints, corpus, runner, max_steps_per_episode=10, seed=1)
    env.reset(seed=1)

    create_idx = next(i for i, e in enumerate(env.endpoints) if e.operation_id == "createOrder")
    obs, reward, terminated, truncated, info = env.step(create_idx)
    assert not truncated

    obs, reward, terminated, truncated, info = env.step(create_idx)
    assert truncated is True
    assert info.get("budget_exceeded") is True
    env.close()
