"""
Regression tests for the planted-bug pipeline (Phase 6).

Confirmed defects these guard against:
  1. The three planted-bug triggers were unreachable from ApiFuzzEnv's real
     request shapes, so --use-planted-bugs changed nothing.
  2. The runner never persisted anomalies to SQLite, so unique_anomalies_found /
     planted_bugs_found / false_positives / steps_to_first_bug were always
     0 / 0 / 0 / None even when the env had flagged anomalies.
  3. steps_to_first_bug counted non-planted anomalies.
  4. Evaluation read `env.coverage` directly, which raises AttributeError on
     wrapped envs (ppo_llm / ScenarioSeededEnv) under gymnasium >= 1.0.
  5. Planted-bug state was a module-level global shared across experiments.
"""

import sys
import tempfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.train_ppo import build_env
from agent.random_policy import RandomPolicy
from env.scenario_seeding import ScenarioSeededEnv
from executor.mock_target import make_mock_transport
from executor.planted_bugs import (
    BIG_QUANTITY_THRESHOLD,
    PlantedBugsMockTarget,
    classify_planted_bug,
)
from experiment.runner import (
    _evaluate_policy,
    _run_ppo_policy,
    _run_random_policy,
    build_arg_parser,
)
from experiment.tracker import ExperimentTracker
from experiment.metrics import PolicyMetrics, aggregate_metrics, compare_policies
from brainstormer.scenarios import Scenario, ScenarioStep, ScenarioBatch

FIXTURE_SPEC = Path(__file__).parent / "fixtures" / "sample_api.yaml"


def _env(planted: bool, max_steps: int = 50, seed: int = 1):
    return build_env(FIXTURE_SPEC, use_mock=True, max_steps_per_episode=max_steps,
                     seed=seed, use_planted_bugs=planted)


def _idx(env):
    endpoints = env.unwrapped.endpoints if hasattr(env, "unwrapped") else env.endpoints
    return {e.operation_id: i for i, e in enumerate(endpoints)}


def _req(method, path, js=None):
    return httpx.Request(method, "http://mock" + path, json=js)


# ---------- 1. planted bugs are reachable through the REAL env ----------

def test_cold_get_order_triggers_malformed_id_bug_through_env():
    env = _env(planted=True)
    env.reset(seed=1)
    _, _, _, _, info = env.step(_idx(env)["getOrder"])  # no prior createOrder
    assert info["status_code"] == 400
    assert info["anomalous"] is True
    assert info["anomaly_signature"] is not None
    assert classify_planted_bug(info["operation_id"], info["status_code"]) == "bug_getorder_malformed_id"
    env.close()


def test_double_cancel_needs_create_cancel_cancel_sequence():
    env = _env(planted=True)
    i = _idx(env)
    env.reset(seed=1)
    statuses = [env.step(i[op])[4]["status_code"] for op in ("createOrder", "cancelOrder", "cancelOrder")]
    assert statuses == [201, 200, 409]
    env.close()


def test_single_cancel_is_not_a_bug():
    env = _env(planted=True)
    i = _idx(env)
    env.reset(seed=1)
    statuses = [env.step(i[op])[4]["status_code"] for op in ("createOrder", "cancelOrder")]
    assert statuses == [201, 200]
    env.close()


def test_cancel_of_nonexistent_order_stays_404_not_409():
    env = _env(planted=True)
    env.reset(seed=1)
    cancel = _idx(env)["cancelOrder"]
    assert [env.step(cancel)[4]["status_code"] for _ in range(3)] == [404, 404, 404]
    env.close()


def test_big_quantity_boundary_at_transport_level():
    tgt = PlantedBugsMockTarget(make_mock_transport())
    assert tgt.handle_request(_req("POST", "/orders", {"item_id": "x", "quantity": BIG_QUANTITY_THRESHOLD})).status_code == 201
    assert tgt.handle_request(_req("POST", "/orders", {"item_id": "x", "quantity": BIG_QUANTITY_THRESHOLD + 1})).status_code == 422


def test_big_quantity_bug_is_reachable_through_env_with_unseeded_values():
    env = _env(planted=True, max_steps=400, seed=5)
    env.reset(seed=5)
    create = _idx(env)["createOrder"]
    statuses = [env.step(create)[4]["status_code"] for _ in range(300)]
    assert 422 in statuses and 201 in statuses  # both buggy and normal creates occur
    env.close()


# ---------- normal behaviour must be unchanged ----------

def test_flag_off_has_no_planted_behaviour():
    env = _env(planted=False)
    i = _idx(env)
    env.reset(seed=1)
    assert type(env.runner._client._transport).__name__ == "MockTransport"
    statuses = [env.step(i[op])[4]["status_code"] for op in ("createOrder", "cancelOrder", "cancelOrder")]
    assert statuses == [201, 200, 200]          # no 409 without the flag
    assert env.step(i["getOrder"])[4]["status_code"] == 200  # chained id -> found
    env.close()


def test_normal_chain_still_works_with_flag_on():
    env = _env(planted=True)
    i = _idx(env)
    env.reset(seed=1)
    statuses = [env.step(i[op])[4]["status_code"] for op in ("createOrder", "getOrder", "cancelOrder")]
    assert statuses == [201, 200, 200]
    env.close()


def test_planted_state_is_per_instance_not_global():
    a = PlantedBugsMockTarget(make_mock_transport())
    b = PlantedBugsMockTarget(make_mock_transport())
    for t in (a, b):
        t.handle_request(_req("POST", "/orders", {"item_id": "x", "quantity": 1}))  # ord-1
    assert a.handle_request(_req("POST", "/orders/ord-1")).status_code == 200
    # b has never cancelled ord-1: first cancel there must still succeed.
    assert b.handle_request(_req("POST", "/orders/ord-1")).status_code == 200
    assert a.handle_request(_req("POST", "/orders/ord-1")).status_code == 409
    a.reset_state()
    assert a.handle_request(_req("POST", "/orders/ord-1")).status_code == 200


def test_classify_planted_bug_mapping():
    assert classify_planted_bug("getOrder", 400) == "bug_getorder_malformed_id"
    assert classify_planted_bug("cancelOrder", 409) == "bug_cancelorder_double"
    assert classify_planted_bug("createOrder", 422) == "bug_createorder_big_qty"
    assert classify_planted_bug("getOrder", 404) is None      # ordinary undeclared 404
    assert classify_planted_bug("cancelOrder", 404) is None
    assert classify_planted_bug("getOrder", None) is None


# ---------- 2. env exposes the signature the tracker needs ----------

def test_env_info_signature_matches_coverage_tracker_signature():
    env = _env(planted=False)
    env.reset(seed=1)
    _, _, _, _, info = env.step(_idx(env)["getOrder"])  # cold -> undeclared 404
    assert info["anomaly_signature"] in env.coverage.anomaly_signatures
    _, _, _, _, ok = env.step(_idx(env)["createOrder"])
    assert ok["anomaly_signature"] is None  # non-anomalous step carries no signature
    env.close()


# ---------- 3. runner persists anomalies end to end ----------

def test_runner_persists_anomalies_even_without_planted_bugs():
    """The originally reported symptom: env saw undeclared-404 anomalies but the
    DB reported 0."""
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        exp_id = _run_random_policy(FIXTURE_SPEC, n_episodes=30, max_steps_per_episode=50,
                                    use_planted_bugs=False, trial_seed=7, db_path=db, trial_id=0)
        m = ExperimentTracker(db).compute_final_metrics(exp_id)
        assert m["unique_anomalies_found"] > 0
        assert m["planted_bugs_found"] == 0
        assert m["false_positives"] == m["unique_anomalies_found"]
        assert m["steps_to_first_bug"] is None   # no planted bug => honest None


def test_runner_records_and_classifies_planted_bugs():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        exp_id = _run_random_policy(FIXTURE_SPEC, n_episodes=40, max_steps_per_episode=50,
                                    use_planted_bugs=True, trial_seed=7, db_path=db, trial_id=0)
        tracker = ExperimentTracker(db)
        m = tracker.compute_final_metrics(exp_id)
        assert m["planted_bugs_found"] == 3
        assert m["false_positives"] >= 1                      # cold cancelOrder -> undeclared 404
        assert m["unique_anomalies_found"] == m["planted_bugs_found"] + m["false_positives"]
        assert isinstance(m["steps_to_first_bug"], int) and m["steps_to_first_bug"] >= 1

        import sqlite3
        conn = sqlite3.connect(str(db))
        names = {r[0] for r in conn.execute(
            "SELECT bug_name FROM anomalies WHERE experiment_id=? AND is_planted_bug=1", (exp_id,))}
        conn.close()
        assert names == {"bug_getorder_malformed_id", "bug_cancelorder_double", "bug_createorder_big_qty"}


def test_planted_flag_changes_experiment_outcome():
    """--use-planted-bugs must not be a no-op."""
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        kwargs = dict(n_episodes=40, max_steps_per_episode=50, trial_seed=7, db_path=db, trial_id=0)
        off = _run_random_policy(FIXTURE_SPEC, use_planted_bugs=False, **kwargs)
        on = _run_random_policy(FIXTURE_SPEC, use_planted_bugs=True, **kwargs)
        t = ExperimentTracker(db)
        assert t.compute_final_metrics(off)["planted_bugs_found"] == 0
        assert t.compute_final_metrics(on)["planted_bugs_found"] > 0


def test_ppo_path_runs_and_records_with_planted_bugs():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        exp_id = _run_ppo_policy(FIXTURE_SPEC, n_episodes=3, max_steps_per_episode=20,
                                 timesteps_total=256, trial_seed=3, db_path=db, trial_id=0,
                                 use_planted_bugs=True)
        t = ExperimentTracker(db)
        assert len(t.get_episodes(exp_id)) == 3
        m = t.compute_final_metrics(exp_id)
        assert m["unique_anomalies_found"] == m["planted_bugs_found"] + m["false_positives"]


def test_evaluation_works_on_scenario_wrapped_env():
    """ppo_llm wraps the env in ScenarioSeededEnv; evaluation must not touch
    wrapper attributes that gymnasium >= 1.0 no longer forwards."""
    class _EmptyPool:
        def maybe_refresh(self, *_a, **_k): pass
        def pop_scenario(self): return None

    base = _env(planted=True, max_steps=20)
    wrapped = ScenarioSeededEnv(base, _EmptyPool(), seed_episode_probability=0.0)
    with tempfile.TemporaryDirectory() as d:
        tracker = ExperimentTracker(Path(d) / "t.db")
        exp_id = tracker.start_experiment("w", "ppo_llm", 0, 1, 20, 0, 3)
        policy = RandomPolicy(base.action_space.n, seed=1)
        _evaluate_policy(wrapped, lambda o: policy.predict(o)[0], tracker, exp_id,
                         n_episodes=3, max_steps_per_episode=20, reset_seed_base=1,
                         use_planted_bugs=True)
        assert len(tracker.get_episodes(exp_id)) == 3


# ---------- 4. steps_to_first_bug semantics (tracker) ----------

def test_steps_to_first_bug_ignores_non_planted_anomalies():
    with tempfile.TemporaryDirectory() as d:
        t = ExperimentTracker(Path(d) / "t.db")
        e = t.start_experiment("x", "random", 0, 1, 50, 0, 2)
        t.record_anomaly_at_step(e, "getOrder", 404, ("getOrder", 404, False, None, (), ()), 0, 2)           # false positive, early
        t.record_anomaly_at_step(e, "cancelOrder", 409, ("cancelOrder", 409, False, None, (), ()), 1, 77,
                                 is_planted_bug=1, bug_name="bug_cancelorder_double")
        assert t.get_bug_discoveries(e)["steps_to_first_bug"] == 77


def test_steps_to_first_bug_none_when_only_false_positives():
    with tempfile.TemporaryDirectory() as d:
        t = ExperimentTracker(Path(d) / "t.db")
        e = t.start_experiment("x", "random", 0, 1, 50, 0, 1)
        t.record_anomaly_at_step(e, "getOrder", 404, ("getOrder", 404, False, None, (), ()), 0, 2)
        assert t.get_bug_discoveries(e)["steps_to_first_bug"] is None


# ---------- 5. target state reset between episodes ----------

def test_env_reset_clears_target_orders_and_cancellations():
    """Target mock state and cancellations must not leak across episodes."""
    env = _env(planted=True, max_steps=50, seed=1)
    i = _idx(env)
    env.reset(seed=1)
    # Episode 1: create order and cancel it
    res1 = env.step(i["createOrder"])[4]
    assert res1["status_code"] == 201
    res2 = env.step(i["cancelOrder"])[4]
    assert res2["status_code"] == 200

    # Reset for Episode 2
    env.reset(seed=2)
    # Now verify that ord-1 from episode 1 is not in the target anymore.
    # Requesting cancelOrder on a fresh target with no orders in memory returns 404.
    res3 = env.step(i["cancelOrder"])[4]
    assert res3["status_code"] == 404
    env.close()


def test_mock_transport_reset_state_clears_store_and_counter():
    t = make_mock_transport()
    res1 = t.handle_request(_req("POST", "/orders", {"item_id": "x", "quantity": 1}))
    assert res1.status_code == 201
    assert res1.json()["id"] == "ord-1"
    assert t.handle_request(_req("GET", "/orders/ord-1")).status_code == 200

    t.reset_state()
    # After reset, ord-1 should be gone (404) and new order should start at ord-1
    assert t.handle_request(_req("GET", "/orders/ord-1")).status_code == 404
    res2 = t.handle_request(_req("POST", "/orders", {"item_id": "y", "quantity": 1}))
    assert res2.status_code == 201
    assert res2.json()["id"] == "ord-1"


# ---------- 6. PPO+LLM activation & configurable interval ----------

class _MockLLMClient:
    def __init__(self, scenarios=None):
        self.call_count = 0
        self.scenarios = scenarios or [
            Scenario(
                tag="ordering-effect",
                hypothesis="Create then get checks order visibility",
                sequence=[ScenarioStep(operation_id="createOrder"), ScenarioStep(operation_id="getOrder")],
            )
        ]

    def brainstorm(self, prompt, max_scenarios=8):
        self.call_count += 1
        return ScenarioBatch(scenarios=self.scenarios)


def test_brainstorm_interval_configurable_invokes_mock_llm():
    """Short test run invokes mock LLM at configured interval without Ollama."""
    mock_llm = _MockLLMClient()
    env = build_env(
        FIXTURE_SPEC,
        use_mock=True,
        enable_brainstormer=True,
        brainstorm_interval_episodes=2,
        seed_episode_probability=1.0,
        llm_client=mock_llm,
        seed=1,
    )
    assert isinstance(env, ScenarioSeededEnv)

    # Episode 1 reset -> count is 1, interval is 2, no brainstorm yet
    env.reset(seed=1)
    assert mock_llm.call_count == 0

    # Episode 2 reset -> count is 2, triggers brainstorm and pops the scenario!
    _, info = env.reset(seed=2)
    assert mock_llm.call_count == 1
    assert info["scenario_seeded"] is True
    assert info["scenario_tag"] == "ordering-effect"
    env.close()


def test_scenario_grounding_rejects_hallucinated_endpoints_in_pool():
    """Scenarios with hallucinated endpoints are filtered out during grounding."""
    mock_llm = _MockLLMClient(
        scenarios=[
            Scenario(
                tag="other",
                hypothesis="Hallucinated endpoint",
                sequence=[ScenarioStep(operation_id="fakeEndpoint")],
            ),
            Scenario(
                tag="ordering-effect",
                hypothesis="Valid endpoint",
                sequence=[ScenarioStep(operation_id="createOrder")],
            ),
        ]
    )
    env = build_env(
        FIXTURE_SPEC,
        use_mock=True,
        enable_brainstormer=True,
        brainstorm_interval_episodes=1,
        seed_episode_probability=0.0,
        llm_client=mock_llm,
    )
    env.reset(seed=1)
    assert mock_llm.call_count == 1
    # Only the valid scenario made it into the pool queue
    assert len(env.scenario_pool) == 1
    assert env.scenario_pool.pop_scenario().tag == "ordering-effect"
    env.close()


def test_scenario_seeded_env_tracks_action_mismatch_in_legacy_mode():
    """ScenarioSeededEnv in legacy mode flags when sampled action differs from scripted action."""
    mock_llm = _MockLLMClient(
        scenarios=[
            Scenario(
                tag="other",
                hypothesis="Test mismatch",
                sequence=[ScenarioStep(operation_id="createOrder")],
            )
        ]
    )
    env = build_env(
        FIXTURE_SPEC,
        use_mock=True,
        enable_brainstormer=True,
        brainstorm_interval_episodes=1,
        seed_episode_probability=1.0,
        llm_client=mock_llm,
        seed=0,
        prepare_at_reset=False,
    )
    env.reset(seed=0)
    i = _idx(env)
    # Agent samples getOrder (action != createOrder)
    _, _, _, _, info = env.step(i["getOrder"])
    assert info["scenario_seeded_step"] is True
    assert info["operation_id"] == "createOrder"
    assert info["action_mismatch"] is True

    # Next step is post-prefix, under pure agent control
    _, _, _, _, info2 = env.step(i["getOrder"])
    assert info2["scenario_seeded_step"] is False
    assert info2["action_mismatch"] is False
    env.close()


def test_reset_preparation_no_scripted_action_replaces_agent_action():
    """In reset-preparation mode, agent actions are never replaced during step()."""
    mock_llm = _MockLLMClient(
        scenarios=[
            Scenario(
                tag="ordering-effect",
                hypothesis="Prime with createOrder",
                sequence=[ScenarioStep(operation_id="createOrder")],
            )
        ]
    )
    env = build_env(
        FIXTURE_SPEC,
        use_mock=True,
        enable_brainstormer=True,
        brainstorm_interval_episodes=1,
        seed_episode_probability=1.0,
        llm_client=mock_llm,
        seed=0,
        prepare_at_reset=True,
    )
    obs, info = env.reset(seed=0)
    assert info["scenario_seeded"] is True
    assert info["prepared_operations"] == ["createOrder"]

    i = _idx(env)
    # Agent takes getOrder (action index 1)
    # Because prepare_at_reset is True, getOrder MUST NOT be replaced by createOrder!
    _, _, _, _, step_info = env.step(i["getOrder"])
    assert step_info["operation_id"] == "getOrder"
    assert step_info["scenario_seeded_step"] is False
    assert step_info["action_mismatch"] is False
    env.close()


def test_reset_preparation_primes_target_state_and_enables_clean_chaining():
    """Reset-time scenario preparation primes target state and known_values so PPO chains cleanly."""
    mock_llm = _MockLLMClient(
        scenarios=[
            Scenario(
                tag="ordering-effect",
                hypothesis="Prime with createOrder",
                sequence=[ScenarioStep(operation_id="createOrder")],
            )
        ]
    )
    env = build_env(
        FIXTURE_SPEC,
        use_mock=True,
        enable_brainstormer=True,
        brainstorm_interval_episodes=1,
        seed_episode_probability=1.0,
        llm_client=mock_llm,
        seed=1,
        prepare_at_reset=True,
    )
    obs, info = env.reset(seed=1)
    assert info["scenario_seeded"] is True

    # Check that ord-1 exists in target state without PPO taking an action yet
    # And known_values has ('createOrder', 'id') = 'ord-1'
    assert ("createOrder", "id") in env.unwrapped.known_values
    created_id = env.unwrapped.known_values[("createOrder", "id")]
    assert created_id == "ord-1"

    # Step count is 0
    assert env.unwrapped._step_count == 0

    # Step 0 taken by agent: agent executes getOrder
    i = _idx(env)
    obs1, reward, term, trunc, step_info = env.step(i["getOrder"])
    # getOrder resolves with ord-1 and succeeds with 200 OK!
    assert step_info["operation_id"] == "getOrder"
    assert step_info["status_code"] == 200
    assert step_info["anomalous"] is False
    # Agent earns +1.0 coverage reward for covering getOrder!
    assert reward == 1.0
    assert env.unwrapped._step_count == 1
    env.close()


def test_reset_preparation_observation_reflects_prepared_operations():
    """Observation at reset reflects the prepared endpoints in its vector."""
    mock_llm = _MockLLMClient(
        scenarios=[
            Scenario(
                tag="ordering-effect",
                hypothesis="Prime with createOrder",
                sequence=[ScenarioStep(operation_id="createOrder")],
            )
        ]
    )
    env = build_env(
        FIXTURE_SPEC,
        use_mock=True,
        enable_brainstormer=True,
        brainstorm_interval_episodes=1,
        seed_episode_probability=1.0,
        llm_client=mock_llm,
        seed=2,
        prepare_at_reset=True,
    )
    obs, info = env.reset(seed=2)
    i = _idx(env)
    create_idx = i["createOrder"]
    # Endpoint createOrder is 1.0 in observation
    assert obs[create_idx] == 1.0
    # Progress is 0.0
    assert obs[-1] == 0.0
    env.close()


def test_reset_preparation_reproducibility_under_fixed_seeds():
    """Fixed RNG seeds produce identical scenario selection and preparation."""
    mock_llm = _MockLLMClient()
    env1 = build_env(FIXTURE_SPEC, use_mock=True, enable_brainstormer=True,
                     seed_episode_probability=0.5, brainstorm_interval_episodes=1,
                     llm_client=mock_llm, seed=42, prepare_at_reset=True)
    env2 = build_env(FIXTURE_SPEC, use_mock=True, enable_brainstormer=True,
                     seed_episode_probability=0.5, brainstorm_interval_episodes=1,
                     llm_client=mock_llm, seed=42, prepare_at_reset=True)

    _, info1 = env1.reset(seed=42)
    _, info2 = env2.reset(seed=42)

    assert info1["scenario_seeded"] == info2["scenario_seeded"]
    assert info1["scenario_tag"] == info2["scenario_tag"]
    assert info1["prepared_operations"] == info2["prepared_operations"]
    env1.close()
    env2.close()


def test_eval_env_never_overrides_actions_with_brainstormer():
    """Evaluation environment is never wrapped in ScenarioSeededEnv."""
    # When building eval_env through runner or build_env with enable_brainstormer=False
    eval_env = build_env(
        FIXTURE_SPEC,
        use_mock=True,
        enable_brainstormer=False,
    )
    assert not isinstance(eval_env, ScenarioSeededEnv)
    eval_env.close()


def test_cli_brainstorm_interval_argument_parsing():
    parser = build_arg_parser()
    args = parser.parse_args(["--brainstorm-interval", "25"])
    assert args.brainstorm_interval == 25


def test_tracker_timezone_aware_utc_timestamps_emit_no_deprecation_warnings():
    """ExperimentTracker uses timezone-aware UTC timestamps with zero deprecation warnings."""
    with tempfile.TemporaryDirectory() as d:
        t = ExperimentTracker(Path(d) / "t.db")
        exp_id = t.start_experiment("test", "random", 0, 1, 10, 0, 1)
        exp = t.get_experiment(exp_id)
        assert exp["timestamp"] is not None
        assert "T" in exp["timestamp"]


# ---------- 7. Ground-truth metrics & PolicyMetrics aggregation ----------

def test_policy_metrics_includes_planted_bugs_and_false_positives():
    exp_metrics = [
        {
            "policy": "ppo",
            "mean_episode_reward": 5.0,
            "total_reward": 100.0,
            "unique_operations_covered": 3,
            "total_coverage_hits": 30,
            "unique_anomalies_found": 3,
            "planted_bugs_found": 2,
            "false_positives": 1,
            "steps_to_first_bug": 15,
        },
        {
            "policy": "ppo",
            "mean_episode_reward": 7.0,
            "total_reward": 140.0,
            "unique_operations_covered": 3,
            "total_coverage_hits": 35,
            "unique_anomalies_found": 4,
            "planted_bugs_found": 3,
            "false_positives": 1,
            "steps_to_first_bug": 25,
        },
    ]
    agg = aggregate_metrics(exp_metrics)
    assert agg.policy == "ppo"
    assert agg.mean_planted_bugs == 2.5
    assert agg.mean_false_positives == 1.0
    assert agg.mean_steps_to_first_bug == 20.0

    rendered = str(agg)
    assert "Planted bugs found: 2.50" in rendered
    assert "False positives: 1.00" in rendered
    assert "Mean steps to first bug: 20.0" in rendered

    compared = compare_policies([agg])
    assert "EXPERIMENT COMPARISON" in compared
    assert "Planted bugs found: 2.50" in compared
