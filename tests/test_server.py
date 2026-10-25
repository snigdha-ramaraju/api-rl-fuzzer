"""
Tests for FastAPI server endpoints:
- GET /api/health
- POST /api/spec/parse
- GET /api/experiments
- GET /api/experiments/{id}
- POST /api/runs/start
- GET /api/runs/{run_id}/status
- GET /api/runs
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from server import app, _runs, _runs_lock, _execute_run_worker
from experiment.tracker import ExperimentTracker
from experiment.runner import ALL_POLICIES

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SPEC_PATH = REPO_ROOT / "tests" / "fixtures" / "sample_api.yaml"
ECOMMERCE_SPEC_PATH = REPO_ROOT / "tests" / "fixtures" / "ecommerce_api.yaml"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def temp_tracker_db(tmp_path):
    """Creates a temporary sqlite DB with populated experiment data for test isolation."""
    db_file = tmp_path / "test_experiments.db"
    tracker = ExperimentTracker(db_file)
    exp_id = tracker.start_experiment(
        name="test_trial_0",
        policy="random",
        trial_id=0,
        seed=42,
        max_steps_per_episode=10,
        timesteps_total=20,
        n_episodes=2,
    )
    # Record an episode
    cov_hits = {("createOrder", 201): 5, ("getOrder", 200): 3}
    signatures = {("cancelOrder", 404, True)}
    tracker.record_episode(
        experiment_id=exp_id,
        episode_number=0,
        total_reward=15.5,
        steps_taken=10,
        coverage_hits=cov_hits,
        anomaly_signatures=signatures,
        new_anomaly_count=1,
    )
    # Record an anomaly
    tracker.record_anomaly_at_step(
        experiment_id=exp_id,
        operation_id="cancelOrder",
        status_code=404,
        signature=("cancelOrder", 404, True),
        episode=0,
        step=5,
        is_planted_bug=0,
    )
    return db_file, exp_id


# =========================================================================
# Existing Endpoint Tests
# =========================================================================

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.2.0"
    assert data["supported_policies"] == ALL_POLICIES


def test_spec_parse_json_dict(client):
    with open(SAMPLE_SPEC_PATH, "r", encoding="utf-8") as f:
        spec_dict = yaml.safe_load(f)

    response = client.post("/api/spec/parse", json=spec_dict)
    assert response.status_code == 200
    data = response.json()

    assert "nodes" in data
    assert "edges" in data
    assert "unresolved" in data
    assert data["endpoints_count"] == 3
    assert data["edges_count"] == 6

    # Verify ReactFlow node schema
    first_node = data["nodes"][0]
    assert "id" in first_node
    assert first_node["type"] == "endpointNode"
    assert "position" in first_node
    assert "x" in first_node["position"] and "y" in first_node["position"]
    assert "data" in first_node
    assert first_node["data"]["operation_id"] in ["createOrder", "getOrder", "cancelOrder"]
    assert "method" in first_node["data"]
    assert "path" in first_node["data"]

    # Verify ReactFlow edge schema
    first_edge = data["edges"][0]
    assert "id" in first_edge
    assert "source" in first_edge
    assert "target" in first_edge
    assert "label" in first_edge
    assert "animated" in first_edge
    assert "data" in first_edge
    assert "producer_op" in first_edge["data"]
    assert "consumer_op" in first_edge["data"]


def test_spec_parse_yaml_raw_body(client):
    with open(ECOMMERCE_SPEC_PATH, "r", encoding="utf-8") as f:
        yaml_content = f.read()

    response = client.post(
        "/api/spec/parse",
        content=yaml_content,
        headers={"content-type": "application/x-yaml"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["endpoints_count"] == 6
    assert data["edges_count"] > 0
    assert len(data["nodes"]) == 6


def test_spec_parse_file_upload(client):
    with open(SAMPLE_SPEC_PATH, "rb") as f:
        response = client.post(
            "/api/spec/parse",
            files={"file": ("sample_api.yaml", f, "application/x-yaml")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["endpoints_count"] == 3
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 6


def test_spec_parse_invalid_syntax(client):
    response = client.post(
        "/api/spec/parse",
        content="this is not: [valid yaml: {",
        headers={"content-type": "text/plain"},
    )
    assert response.status_code == 400


def test_spec_parse_invalid_openapi_schema(client):
    invalid_spec = {
        "openapi": "3.0.0",
        "info": {"title": "Bad Spec"},
        # Missing 'paths' object which is mandatory in OpenAPI
    }
    response = client.post("/api/spec/parse", json=invalid_spec)
    assert response.status_code == 400
    assert "OpenAPI validation error" in response.json()["detail"]


def test_spec_parse_empty_payload(client):
    response = client.post("/api/spec/parse")
    assert response.status_code == 400


def test_get_experiments_list(client, temp_tracker_db):
    db_file, exp_id = temp_tracker_db
    response = client.get(f"/api/experiments?db_path={db_file}")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["experiments"]) == 1
    assert data["experiments"][0]["id"] == exp_id
    assert data["experiments"][0]["policy"] == "random"


def test_get_experiments_filter(client, temp_tracker_db):
    db_file, exp_id = temp_tracker_db
    # Matching policy
    res_match = client.get(f"/api/experiments?db_path={db_file}&policy=random")
    assert res_match.status_code == 200
    assert res_match.json()["total"] == 1

    # Non-matching policy
    res_miss = client.get(f"/api/experiments?db_path={db_file}&policy=ppo")
    assert res_miss.status_code == 200
    assert res_miss.json()["total"] == 0


def test_get_experiments_missing_db(client):
    response = client.get("/api/experiments?db_path=nonexistent_db_12345.db")
    assert response.status_code == 404
    assert "Database file not found" in response.json()["detail"]


def test_get_experiment_details_success(client, temp_tracker_db):
    db_file, exp_id = temp_tracker_db
    response = client.get(f"/api/experiments/{exp_id}?db_path={db_file}")
    assert response.status_code == 200
    data = response.json()

    assert data["id"] == exp_id
    assert data["experiment"]["policy"] == "random"

    # Metrics
    assert "metrics" in data
    assert data["metrics"]["experiment_id"] == exp_id
    assert data["metrics"]["n_episodes"] == 2

    # Coverage
    assert "coverage" in data
    assert len(data["coverage"]) == 2
    ops = {item["operation_id"] for item in data["coverage"]}
    assert "createOrder" in ops and "getOrder" in ops

    # Anomalies
    assert "anomalies" in data
    assert len(data["anomalies"]) == 1
    assert data["anomalies"][0]["operation_id"] == "cancelOrder"
    assert data["anomalies"][0]["status_code"] == 404

    # Episodes & bug discoveries
    assert len(data["episodes"]) == 1
    assert "bug_discoveries" in data


def test_get_experiment_details_not_found(client, temp_tracker_db):
    db_file, _ = temp_tracker_db
    response = client.get(f"/api/experiments/99999?db_path={db_file}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# =========================================================================
# Step 2: Run Management Tests (POST /api/runs/start, GET /api/runs/{id}/status)
# =========================================================================

def test_runs_start_and_poll_completion(client, tmp_path):
    """Test starting a fuzzing run asynchronously and polling its status to completion."""
    db_file = tmp_path / "run_test.db"

    payload = {
        "policy": "random",
        "spec_path": str(SAMPLE_SPEC_PATH),
        "n_episodes": 2,
        "max_steps_per_episode": 5,
        "seed": 42,
        "db_path": str(db_file),
    }

    response = client.post("/api/runs/start", json=payload)
    assert response.status_code == 200
    start_data = response.json()
    assert "run_id" in start_data
    assert start_data["status"] == "queued"
    assert start_data["policy"] == "random"

    run_id = start_data["run_id"]

    # Poll status until completed (timeout after 5s)
    completed = False
    status_data = None
    for _ in range(25):
        time.sleep(0.1)
        res_poll = client.get(f"/api/runs/{run_id}/status")
        assert res_poll.status_code == 200
        status_data = res_poll.json()
        if status_data["status"] == "completed":
            completed = True
            break
        elif status_data["status"] == "failed":
            pytest.fail(f"Run failed unexpectedly: {status_data.get('error')}")

    assert completed, f"Run did not complete in time: {status_data}"
    assert status_data["experiment_id"] is not None
    assert status_data["progress"]["current_episode"] == 2
    assert status_data["progress"]["requests_made"] == 10
    assert status_data["completed_at"] is not None

    # Verify experiment was actually persisted in SQLite
    exp_res = client.get(f"/api/experiments/{status_data['experiment_id']}?db_path={db_file}")
    assert exp_res.status_code == 200
    assert exp_res.json()["experiment"]["policy"] == "random"


def test_runs_start_with_inline_yaml_spec(client, tmp_path):
    """Test starting a run with inline YAML string instead of a file path."""
    db_file = tmp_path / "inline_test.db"
    with open(SAMPLE_SPEC_PATH, "r", encoding="utf-8") as f:
        spec_text = f.read()

    payload = {
        "policy": "greedy",
        "spec": spec_text,
        "n_episodes": 2,
        "max_steps_per_episode": 5,
        "seed": 100,
        "db_path": str(db_file),
    }

    response = client.post("/api/runs/start", json=payload)
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    # Poll for completion
    completed = False
    for _ in range(25):
        time.sleep(0.1)
        status_data = client.get(f"/api/runs/{run_id}/status").json()
        if status_data["status"] == "completed":
            completed = True
            assert status_data["policy"] == "greedy"
            assert status_data["experiment_id"] is not None
            break

    assert completed


def test_runs_start_invalid_policy(client):
    """Validation test: reject unknown/unsupported policy."""
    payload = {
        "policy": "magic_unsupported_policy",
        "spec_path": str(SAMPLE_SPEC_PATH),
    }
    response = client.post("/api/runs/start", json=payload)
    assert response.status_code == 400
    assert "Unsupported policy" in response.json()["detail"]


def test_runs_start_invalid_episodes_budget(client):
    """Validation test: reject invalid episode counts."""
    payload = {
        "policy": "random",
        "spec_path": str(SAMPLE_SPEC_PATH),
        "n_episodes": 0,  # invalid: must be >= 1
    }
    response = client.post("/api/runs/start", json=payload)
    assert response.status_code == 400
    assert "n_episodes must be between 1 and 500" in response.json()["detail"]


def test_runs_start_invalid_steps_budget(client):
    """Validation test: reject invalid max_steps_per_episode."""
    payload = {
        "policy": "random",
        "spec_path": str(SAMPLE_SPEC_PATH),
        "max_steps_per_episode": -5,  # invalid
    }
    response = client.post("/api/runs/start", json=payload)
    assert response.status_code == 400
    assert "max_steps_per_episode must be between 1 and 200" in response.json()["detail"]


def test_runs_start_missing_spec(client):
    """Validation test: reject request missing both spec and spec_path."""
    payload = {
        "policy": "random",
    }
    response = client.post("/api/runs/start", json=payload)
    assert response.status_code == 400
    assert "An OpenAPI specification must be provided" in response.json()["detail"]


def test_runs_start_nonexistent_spec_path(client):
    """Validation test: reject non-existent spec file path."""
    payload = {
        "policy": "random",
        "spec_path": "nonexistent_spec_file_9999.yaml",
    }
    response = client.post("/api/runs/start", json=payload)
    assert response.status_code == 400
    assert "Specification file not found" in response.json()["detail"]


def test_runs_start_invalid_spec_yaml(client):
    """Validation test: reject invalid YAML / OpenAPI schema."""
    payload = {
        "policy": "random",
        "spec": "invalid: [yaml: syntax",
    }
    response = client.post("/api/runs/start", json=payload)
    assert response.status_code == 400


def test_runs_status_not_found(client):
    """Querying an unknown run ID returns 404."""
    response = client.get("/api/runs/nonexistent_run_id_404/status")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_runs_list_endpoint(client):
    """Verify listing runs works."""
    response = client.get("/api/runs")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "runs" in data
    assert isinstance(data["runs"], list)


def test_runs_failure_handling(client, tmp_path):
    """Verify that worker runtime exceptions set status to 'failed' and record error message."""
    run_id = "test_failed_run_123"
    with _runs_lock:
        _runs[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "policy": "random",
            "created_at": "2026-01-01T00:00:00Z",
            "started_at": None,
            "completed_at": None,
            "experiment_id": None,
            "error": None,
            "progress": {"stage": "queued"},
            "config": {},
        }

    # Execute worker with a bogus spec path that will fail inside the runner
    _execute_run_worker(
        run_id=run_id,
        spec_path=Path("non_existent_file.yaml"),
        policy="random",
        n_episodes=2,
        max_steps_per_episode=5,
        timesteps_total=256,
        seed=42,
        use_planted_bugs=False,
        db_path=tmp_path / "test.db",
        trial_id=0,
    )

    status_res = client.get(f"/api/runs/{run_id}/status")
    assert status_res.status_code == 200
    data = status_res.json()
    assert data["status"] == "failed"
    assert data["error"] is not None
    assert data["completed_at"] is not None
