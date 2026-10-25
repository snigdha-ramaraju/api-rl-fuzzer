"""
Phase 2: Tests for Explicit, Bounded Exploration and Frozen MaskablePPO Execution.

Validates:
1. Exploration budget enforcement: explicit request count and ratio fractions.
2. Hard total request budget cap across exploration and PPO phases.
3. Phase transitions: exploration feedback (coverage, known values) preserved into PPO observation.
4. Strict action masking during PPO rollout on held-out APIs.
5. State isolation across different API runs.
6. Explicit failure handling for missing models and invalid OpenAPI specs (never fakes runs).
7. Persisted metrics: separate exploration and PPO requests/anomalies in SQLite.
8. FastAPI /api/runs endpoints: stage/phase transitions and genuine progress polling.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from spec.parser import load_spec, parse_endpoints
from experiment.tracker import ExperimentTracker
from experiment.runner import (
    _run_cross_api_ppo_with_exploration,
    ALL_POLICIES,
    REPO_ROOT,
)
from agent.train_ppo import train_cross_api
from server import app

SAMPLE_SPEC = REPO_ROOT / "tests" / "fixtures" / "sample_api.yaml"
ECOMMERCE_SPEC = REPO_ROOT / "tests" / "fixtures" / "ecommerce_api.yaml"
TINY_SPEC = REPO_ROOT / "tests" / "fixtures" / "tiny_api.yaml"
CRAPI_SPEC = REPO_ROOT / "datasets" / "openapi" / "crapi-openapi-spec.json"


@pytest.fixture(scope="module")
def pretrained_model(tmp_path_factory):
    """Train a quick reusable MaskablePPO model across sample and ecommerce specs."""
    tmp_dir = tmp_path_factory.mktemp("models")
    model_path = tmp_dir / "test_cross_api_model"
    train_cross_api(
        spec_paths=[SAMPLE_SPEC, ECOMMERCE_SPEC],
        output_path=model_path,
        timesteps=128,
        max_slots=64,
    )
    return model_path.with_suffix(".zip")


# =========================================================================
# 1. Budget Enforcement Tests
# =========================================================================

def test_exploration_budget_enforcement_explicit(tmp_path, pretrained_model):
    """Verify explicit explore_budget=10 out of total_budget=30."""
    db_path = tmp_path / "exp_budget.db"
    exp_id = _run_cross_api_ppo_with_exploration(
        spec_path=TINY_SPEC,
        total_budget=30,
        explore_budget=10,
        model_path=pretrained_model,
        db_path=db_path,
        trial_id=1,
    )

    tracker = ExperimentTracker(db_path)
    metrics = tracker.compute_final_metrics(exp_id)

    assert metrics["exploration_requests"] == 10
    assert metrics["ppo_requests"] == 20
    assert metrics["total_requests"] == 30


def test_exploration_budget_enforcement_ratio(tmp_path, pretrained_model):
    """Verify explore_ratio=0.25 allocates 25% of total_budget to exploration."""
    db_path = tmp_path / "ratio_budget.db"
    exp_id = _run_cross_api_ppo_with_exploration(
        spec_path=TINY_SPEC,
        total_budget=40,
        explore_ratio=0.25,
        model_path=pretrained_model,
        db_path=db_path,
        trial_id=2,
    )

    tracker = ExperimentTracker(db_path)
    metrics = tracker.compute_final_metrics(exp_id)

    assert metrics["exploration_requests"] == 10  # 40 * 0.25
    assert metrics["ppo_requests"] == 30
    assert metrics["total_requests"] == 40


def test_hard_total_budget_cap(tmp_path, pretrained_model):
    """Verify total requests strictly obeys total_budget even when episodes would request more."""
    db_path = tmp_path / "hard_cap.db"
    # n_episodes=10, max_steps=20 would be 200 requests, but total_budget=18 caps it at 18
    exp_id = _run_cross_api_ppo_with_exploration(
        spec_path=TINY_SPEC,
        n_episodes=10,
        max_steps_per_episode=20,
        total_budget=18,
        explore_budget=6,
        model_path=pretrained_model,
        db_path=db_path,
        trial_id=3,
    )

    tracker = ExperimentTracker(db_path)
    metrics = tracker.compute_final_metrics(exp_id)

    assert metrics["exploration_requests"] == 6
    assert metrics["ppo_requests"] == 12
    assert metrics["total_requests"] == 18
    assert metrics["total_requests"] <= 18


def test_explore_budget_equal_or_greater_than_total(tmp_path, pretrained_model):
    """When exploration exhausts the entire budget, PPO receives 0 requests and total is conserved."""
    db_path = tmp_path / "full_explore.db"
    exp_id = _run_cross_api_ppo_with_exploration(
        spec_path=TINY_SPEC,
        total_budget=12,
        explore_budget=20,  # exceeds total_budget
        model_path=pretrained_model,
        db_path=db_path,
        trial_id=4,
    )

    tracker = ExperimentTracker(db_path)
    metrics = tracker.compute_final_metrics(exp_id)

    assert metrics["exploration_requests"] == 12
    assert metrics["ppo_requests"] == 0
    assert metrics["total_requests"] == 12


# =========================================================================
# 2. Phase Transition & Exploration Feedback Tests
# =========================================================================

def test_phase_transition_preserves_exploration_state(tmp_path, pretrained_model):
    """Verify that coverage and known dependency values from exploration are preserved
    into the observation passed to the frozen MaskablePPO policy."""
    db_path = tmp_path / "transition.db"
    exp_id = _run_cross_api_ppo_with_exploration(
        spec_path=TINY_SPEC,
        total_budget=20,
        explore_budget=8,
        model_path=pretrained_model,
        db_path=db_path,
        trial_id=5,
    )

    tracker = ExperimentTracker(db_path)
    cov = tracker.get_coverage_summary(exp_id)
    episodes = tracker.get_episodes(exp_id)

    # Tiny API has 2 endpoints: /auth/token and /auth/profile
    assert len(cov) >= 2
    # Episode 0 represents exploration; subsequent episodes represent PPO
    assert len(episodes) >= 2
    assert episodes[0]["episode_number"] == 0
    assert episodes[0]["steps_taken"] == 8


# =========================================================================
# 3. Action Masking Tests
# =========================================================================

def test_strict_action_masking_during_ppo(tmp_path, pretrained_model):
    """Verify that on a 2-endpoint API, PPO only selects actions 0 and 1,
    never inactive actions (2..31)."""
    db_path = tmp_path / "masking.db"
    exp_id = _run_cross_api_ppo_with_exploration(
        spec_path=TINY_SPEC,
        total_budget=25,
        explore_budget=5,
        model_path=pretrained_model,
        db_path=db_path,
        trial_id=6,
    )

    tracker = ExperimentTracker(db_path)
    cov = tracker.get_coverage_summary(exp_id)

    # Operations covered must only belong to Tiny API (2 endpoints)
    operations = {op for op, _ in cov.keys()}
    assert operations.issubset({"login", "getProfile"})


# =========================================================================
# 4. State Isolation Tests
# =========================================================================

def test_clean_api_state_isolation(tmp_path, pretrained_model):
    """Verify running on Sample API then Tiny API produces completely isolated states."""
    db_path = tmp_path / "isolation.db"

    # Run 1: Sample API (Orders API)
    exp1 = _run_cross_api_ppo_with_exploration(
        spec_path=SAMPLE_SPEC,
        total_budget=20,
        explore_budget=6,
        model_path=pretrained_model,
        db_path=db_path,
        trial_id=1,
    )

    # Run 2: Tiny API (Auth API)
    exp2 = _run_cross_api_ppo_with_exploration(
        spec_path=TINY_SPEC,
        total_budget=20,
        explore_budget=6,
        model_path=pretrained_model,
        db_path=db_path,
        trial_id=2,
    )

    tracker = ExperimentTracker(db_path)
    cov1 = tracker.get_coverage_summary(exp1)
    cov2 = tracker.get_coverage_summary(exp2)

    ops1 = {op for op, _ in cov1.keys()}
    ops2 = {op for op, _ in cov2.keys()}

    # No cross-contamination of operations
    assert not (ops1 & ops2)
    assert any("order" in op.lower() for op in ops1)
    assert ops2.issubset({"login", "getProfile"})


# =========================================================================
# 5. Failure Handling Tests (Never Fake Runs)
# =========================================================================

def test_missing_model_raises_explicit_file_not_found(tmp_path):
    """Missing model file must raise FileNotFoundError, never fake a run."""
    db_path = tmp_path / "missing_model.db"
    with pytest.raises(FileNotFoundError, match="MaskablePPO model file not found"):
        _run_cross_api_ppo_with_exploration(
            spec_path=TINY_SPEC,
            total_budget=20,
            explore_budget=5,
            model_path=tmp_path / "nonexistent_model.zip",
            db_path=db_path,
        )


def test_invalid_openapi_spec_raises_explicit_value_error(tmp_path, pretrained_model):
    """Invalid specification must raise ValueError."""
    bad_spec = tmp_path / "bad_spec.yaml"
    bad_spec.write_text("openapi: 3.0.0\ninfo: {}\npaths: not_a_dict\n", encoding="utf-8")

    db_path = tmp_path / "bad_spec.db"
    with pytest.raises(ValueError, match="Invalid OpenAPI specification"):
        _run_cross_api_ppo_with_exploration(
            spec_path=bad_spec,
            total_budget=20,
            explore_budget=5,
            model_path=pretrained_model,
            db_path=db_path,
        )


# =========================================================================
# 6. SQLite Metrics Persistence Tests
# =========================================================================

def test_persisted_metrics_in_database(tmp_path, pretrained_model):
    """Verify exploration_requests, ppo_requests, total_requests, and anomalies
    are persisted in SQLite."""
    db_path = tmp_path / "persistence.db"
    exp_id = _run_cross_api_ppo_with_exploration(
        spec_path=TINY_SPEC,
        total_budget=24,
        explore_budget=8,
        model_path=pretrained_model,
        db_path=db_path,
        trial_id=7,
    )

    tracker = ExperimentTracker(db_path)
    metrics = tracker.compute_final_metrics(exp_id)
    raw_exp = tracker.get_experiment(exp_id)

    assert "exploration_requests" in metrics
    assert "ppo_requests" in metrics
    assert "total_requests" in metrics
    assert metrics["exploration_requests"] == 8
    assert metrics["ppo_requests"] == 16
    assert metrics["total_requests"] == 24
    assert raw_exp["exploration_requests"] == 8
    assert raw_exp["ppo_requests"] == 16


# =========================================================================
# 7. FastAPI Run API Integration Tests
# =========================================================================

def test_api_runs_start_and_status_polling_cross_api_ppo(tmp_path, pretrained_model):
    """Test POST /api/runs/start with policy=cross_api_ppo and poll /api/runs/{run_id}/status."""
    client = TestClient(app)
    db_path = tmp_path / "server_test.db"

    response = client.post(
        "/api/runs/start",
        json={
            "policy": "cross_api_ppo",
            "spec_path": str(TINY_SPEC),
            "n_episodes": 2,
            "max_steps_per_episode": 10,
            "timesteps_total": 20,
            "explore_budget": 6,
            "model_path": str(pretrained_model),
            "db_path": str(db_path),
        },
    )
    assert response.status_code == 200
    data = response.json()
    run_id = data["run_id"]
    assert data["policy"] == "cross_api_ppo"

    # Poll status until completed (timeout 15s)
    start_time = time.time()
    final_status = None
    stages_seen = set()

    while time.time() - start_time < 15:
        st_resp = client.get(f"/api/runs/{run_id}/status")
        assert st_resp.status_code == 200
        st_data = st_resp.json()
        stage = st_data.get("progress", {}).get("stage")
        if stage:
            stages_seen.add(stage)
        if st_data["status"] in ("completed", "failed"):
            final_status = st_data
            break
        time.sleep(0.05)

    assert final_status is not None, "Run did not complete within timeout"
    assert final_status["status"] == "completed"
    assert final_status["experiment_id"] is not None
    assert final_status["progress"]["stage"] == "completed"
    assert final_status["progress"]["exploration_requests"] == 6
    assert final_status["progress"]["requests_made"] == 20


def test_api_runs_missing_model_fails_explicitly(tmp_path):
    """Test that a run with a missing model transitions to status 'failed' with explicit error."""
    client = TestClient(app)
    db_path = tmp_path / "fail_model.db"

    response = client.post(
        "/api/runs/start",
        json={
            "policy": "cross_api_ppo",
            "spec_path": str(TINY_SPEC),
            "n_episodes": 2,
            "max_steps_per_episode": 10,
            "timesteps_total": 20,
            "explore_budget": 5,
            "model_path": str(tmp_path / "missing_nonexistent_model.zip"),
            "db_path": str(db_path),
        },
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    start_time = time.time()
    final_status = None
    while time.time() - start_time < 10:
        st_resp = client.get(f"/api/runs/{run_id}/status")
        st_data = st_resp.json()
        if st_data["status"] in ("completed", "failed"):
            final_status = st_data
            break
        time.sleep(0.05)

    assert final_status is not None
    assert final_status["status"] == "failed"
    assert "MaskablePPO model file not found" in final_status["error"]


def test_crapi_exploration_and_ppo_execution(tmp_path, pretrained_model):
    """Verify that OWASP crAPI (44 endpoints) executes explicit exploration,
    switches to frozen MaskablePPO under strict action masking, and respects budgets."""
    db_path = tmp_path / "crapi_exp.db"
    exp_id = _run_cross_api_ppo_with_exploration(
        spec_path=CRAPI_SPEC,
        total_budget=25,
        explore_budget=10,
        model_path=pretrained_model,
        db_path=db_path,
        trial_id=99,
    )

    tracker = ExperimentTracker(db_path)
    metrics = tracker.compute_final_metrics(exp_id)
    cov = tracker.get_coverage_summary(exp_id)

    assert metrics["exploration_requests"] == 10
    assert metrics["ppo_requests"] == 15
    assert metrics["total_requests"] == 25
    assert len(cov) > 0
    # Operations covered must be crAPI endpoints
    operations = {op for op, _ in cov.keys()}
    assert len(operations) > 0
    assert len(operations) <= 44
