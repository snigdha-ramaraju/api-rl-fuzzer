"""
Phase 5 tests: local-only LLM brainstormer + episode-seeding wrapper.

No real Ollama server is required to run this suite -- LLMClient is
tested with a mocked `ollama.Client` (proving it never talks to anything
else), and the pool/wrapper tests use lightweight fake LLM clients so the
seeding logic itself is verified independent of any specific model.
"""

import sys
import json
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spec.parser import load_spec, parse_endpoints
from spec.graph import build_dependency_graph
from corpus.seed_dict import build_seed_corpus, SeedCorpus

from brainstormer.scenarios import ScenarioBatch, Scenario, ScenarioStep
from brainstormer.hardware import resolve_model, MODEL_TIERS
from brainstormer.grounding import validate_batch
from brainstormer.context import build_brainstorm_prompt
from brainstormer.pool import ScenarioPool

from env.scenario_seeding import ScenarioSeededEnv
from agent.train_ppo import build_env

FIXTURE_SPEC = Path(__file__).parent / "fixtures" / "sample_api.yaml"


def _endpoints():
    spec = load_spec(FIXTURE_SPEC)
    return parse_endpoints(spec)


# ---------- hardware / model tier ----------

def test_resolve_model_explicit_tiers():
    assert resolve_model("small") == MODEL_TIERS["small"]
    assert resolve_model("medium") == MODEL_TIERS["medium"]
    assert resolve_model("large") == MODEL_TIERS["large"]


def test_resolve_model_rejects_unknown_tier():
    try:
        resolve_model("ultra")
        assert False, "should have raised"
    except ValueError:
        pass


# ---------- grounding ----------

def test_grounding_rejects_hallucinated_operation_id():
    endpoints = _endpoints()
    fake_response = {
        "scenarios": [
            {
                "sequence": [{"operation_id": "createOrder", "hint": ""},
                             {"operation_id": "getOrder", "hint": "immediately after create"}],
                "hypothesis": "order visible before payment confirmed",
                "tag": "state-inconsistency",
            },
            {
                "sequence": [{"operation_id": "deleteEverything", "hint": ""}],  # hallucinated
                "hypothesis": "nonexistent op",
                "tag": "other",
            },
        ]
    }
    batch = ScenarioBatch.model_validate(fake_response)
    report = validate_batch(batch, endpoints)

    assert len(report.valid) == 1
    assert report.valid[0].sequence[0].operation_id == "createOrder"
    assert len(report.rejected) == 1
    assert "deleteEverything" in report.rejected[0]["reason"]


# ---------- local-only LLM client ----------

@patch("brainstormer.llm_client.ollama.Client")
def test_llm_client_uses_local_ollama_only(mock_client_cls):
    from brainstormer.llm_client import LLMClient

    mock_instance = MagicMock()
    mock_instance.list.return_value = {"models": [{"model": "qwen2.5:7b-instruct"}]}
    mock_instance.chat.return_value = {
        "message": {
            "content": json.dumps({
                "scenarios": [{
                    "sequence": [{"operation_id": "createOrder", "hint": ""}],
                    "hypothesis": "test",
                    "tag": "other",
                }]
            })
        }
    }
    mock_client_cls.return_value = mock_instance

    client = LLMClient(host="http://localhost:11434", model_tier_setting="medium")
    result = client.brainstorm("dummy prompt", max_scenarios=8)

    assert isinstance(result, ScenarioBatch)
    assert result.scenarios[0].sequence[0].operation_id == "createOrder"
    mock_client_cls.assert_called_once_with(host="http://localhost:11434")


@patch("brainstormer.llm_client.ollama.Client")
def test_llm_client_fails_loudly_when_ollama_unreachable(mock_client_cls):
    from brainstormer.llm_client import LLMClient

    mock_instance = MagicMock()
    mock_instance.list.side_effect = ConnectionError("refused")
    mock_client_cls.return_value = mock_instance

    try:
        LLMClient(host="http://localhost:11434", model_tier_setting="small")
        assert False, "should have raised"
    except RuntimeError as e:
        assert "ollama" in str(e).lower() or "Ollama" in str(e)


# ---------- context builder ----------

def test_context_prompt_includes_endpoints_and_coverage():
    endpoints = _endpoints()
    corpus = build_seed_corpus(endpoints)
    prompt = build_brainstorm_prompt(
        endpoints, corpus,
        coverage_summary={"covered_endpoints": 1, "total_endpoints": 3, "distinct_status_count": 2},
        api_description="Sample Orders API",
    )
    assert "createOrder" in prompt
    assert "getOrder" in prompt
    assert "cancelOrder" in prompt
    assert "Sample Orders API" in prompt
    assert "1/3" in prompt


# ---------- scenario pool (fake LLM client, no real Ollama needed) ----------

class _FakeLLMClient:
    def __init__(self, batch_dict):
        self.batch_dict = batch_dict
        self.call_count = 0

    def brainstorm(self, prompt, max_scenarios):
        self.call_count += 1
        return ScenarioBatch.model_validate(self.batch_dict)


def _sample_batch_dict():
    return {
        "scenarios": [
            {
                "sequence": [{"operation_id": "createOrder", "hint": ""},
                             {"operation_id": "getOrder", "hint": ""}],
                "hypothesis": "chained state check",
                "tag": "ordering-effect",
            }
        ]
    }


def test_pool_only_brainstorms_at_the_configured_interval():
    endpoints = _endpoints()
    corpus = build_seed_corpus(endpoints)
    fake_llm = _FakeLLMClient(_sample_batch_dict())
    pool = ScenarioPool(llm_client=fake_llm, endpoints=endpoints, corpus=corpus,
                         brainstorm_interval_episodes=3)

    added = [pool.maybe_refresh() for _ in range(3)]
    assert added == [0, 0, 1]  # only the 3rd episode boundary triggers a call
    assert fake_llm.call_count == 1
    assert len(pool) == 1


def test_pool_pop_scenario_drains_queue():
    endpoints = _endpoints()
    corpus = build_seed_corpus(endpoints)
    fake_llm = _FakeLLMClient(_sample_batch_dict())
    pool = ScenarioPool(llm_client=fake_llm, endpoints=endpoints, corpus=corpus,
                         brainstorm_interval_episodes=1)

    pool.maybe_refresh()
    assert len(pool) == 1
    scenario = pool.pop_scenario()
    assert scenario is not None
    assert scenario.sequence[0].operation_id == "createOrder"
    assert pool.pop_scenario() is None  # drained


def test_pool_drops_hallucinated_scenarios_via_grounding():
    endpoints = _endpoints()
    corpus = build_seed_corpus(endpoints)
    bad_batch = {
        "scenarios": [
            {"sequence": [{"operation_id": "nonexistentOp", "hint": ""}],
             "hypothesis": "x", "tag": "other"}
        ]
    }
    fake_llm = _FakeLLMClient(bad_batch)
    pool = ScenarioPool(llm_client=fake_llm, endpoints=endpoints, corpus=corpus,
                         brainstorm_interval_episodes=1)

    added = pool.maybe_refresh()
    assert added == 0
    assert len(pool) == 0
    assert "nonexistentOp" in pool.last_rejected[0]["reason"]


# ---------- episode-seeding wrapper ----------

class _FakeScenarioPool:
    """Test double: always returns the same fixed scenario (or None)."""
    def __init__(self, scenario=None):
        self._scenario = scenario
        self.refresh_calls = 0

    def maybe_refresh(self, coverage_summary=None):
        self.refresh_calls += 1
        return 0

    def pop_scenario(self):
        return self._scenario


def _fixed_scenario():
    return Scenario(
        sequence=[ScenarioStep(operation_id="createOrder"), ScenarioStep(operation_id="getOrder")],
        hypothesis="create then immediately read",
        tag="ordering-effect",
    )


def test_wrapper_forces_scripted_sequence_and_dependency_graph_still_chains():
    """The core Phase 5 integration proof: an LLM-seeded scenario forces
    createOrder -> getOrder regardless of the action passed to step(), and
    getOrder correctly resolves via the SAME dependency-graph mechanism
    Phase 3 proved -- yielding 200, not 404."""
    base_env = build_env(FIXTURE_SPEC, use_mock=True, max_steps_per_episode=10, seed=1)
    pool = _FakeScenarioPool(scenario=_fixed_scenario())
    wrapped = ScenarioSeededEnv(base_env, pool, seed_episode_probability=1.0, rng=random.Random(0))

    obs, info = wrapped.reset(seed=1)
    assert info["scenario_seeded"] is True
    assert info["scenario_tag"] == "ordering-effect"
    assert pool.refresh_calls == 1

    _, _, _, _, info1 = wrapped.step(0)  # action arg irrelevant during scripted prefix
    assert info1["operation_id"] == "createOrder"
    assert info1["scenario_seeded_step"] is True
    assert info1["status_code"] == 201

    _, _, _, _, info2 = wrapped.step(0)
    assert info2["operation_id"] == "getOrder"
    assert info2["scenario_seeded_step"] is True
    assert info2["status_code"] == 200  # proves dependency-graph chaining survived seeding
    assert info2["anomalous"] is False

    wrapped.close()


def test_wrapper_reverts_to_agent_control_after_scripted_prefix():
    base_env = build_env(FIXTURE_SPEC, use_mock=True, max_steps_per_episode=10, seed=1)
    pool = _FakeScenarioPool(scenario=_fixed_scenario())
    wrapped = ScenarioSeededEnv(base_env, pool, seed_episode_probability=1.0, rng=random.Random(0))
    wrapped.reset(seed=1)

    wrapped.step(0)  # scripted: createOrder
    wrapped.step(0)  # scripted: getOrder

    cancel_idx = next(i for i, e in enumerate(base_env.endpoints) if e.operation_id == "cancelOrder")
    _, _, _, _, info3 = wrapped.step(cancel_idx)  # agent-controlled now
    assert info3["operation_id"] == "cancelOrder"
    assert info3["scenario_seeded_step"] is False
    wrapped.close()


def test_wrapper_never_seeds_when_probability_is_zero():
    base_env = build_env(FIXTURE_SPEC, use_mock=True, max_steps_per_episode=10, seed=1)
    pool = _FakeScenarioPool(scenario=_fixed_scenario())
    wrapped = ScenarioSeededEnv(base_env, pool, seed_episode_probability=0.0, rng=random.Random(0))

    obs, info = wrapped.reset(seed=1)
    assert info["scenario_seeded"] is False

    get_idx = next(i for i, e in enumerate(base_env.endpoints) if e.operation_id == "getOrder")
    _, _, _, _, info1 = wrapped.step(get_idx)  # should NOT be overridden to createOrder
    assert info1["operation_id"] == "getOrder"
    assert info1["scenario_seeded_step"] is False
    wrapped.close()


def test_wrapper_handles_empty_pool_gracefully():
    base_env = build_env(FIXTURE_SPEC, use_mock=True, max_steps_per_episode=10, seed=1)
    pool = _FakeScenarioPool(scenario=None)  # nothing to pop
    wrapped = ScenarioSeededEnv(base_env, pool, seed_episode_probability=1.0, rng=random.Random(0))

    obs, info = wrapped.reset(seed=1)
    assert info["scenario_seeded"] is False  # probability rolled True, but pool had nothing
    wrapped.close()


# ---------- train_ppo.py opt-in wiring (Phase 4 default path unaffected) ----------

def test_build_env_without_brainstormer_flag_returns_plain_apifuzzenv():
    from env.api_env import ApiFuzzEnv
    env = build_env(FIXTURE_SPEC, use_mock=True)
    assert type(env) is ApiFuzzEnv  # exact Phase 4 behavior, not wrapped
    env.close()


@patch("brainstormer.llm_client.ollama.Client")
def test_build_env_with_brainstormer_flag_returns_wrapped_env(mock_client_cls):
    mock_instance = MagicMock()
    mock_instance.list.return_value = {"models": [{"model": "qwen2.5:3b-instruct"}]}
    mock_instance.chat.return_value = {
        "message": {"content": json.dumps({"scenarios": []})}
    }
    mock_client_cls.return_value = mock_instance

    env = build_env(
        FIXTURE_SPEC, use_mock=True, enable_brainstormer=True,
        llm_model_tier="small", seed_episode_probability=0.5, seed=1,
    )
    assert isinstance(env, ScenarioSeededEnv)
    assert env.action_space.n == 3  # spaces forwarded correctly from wrapped env
    env.close()
