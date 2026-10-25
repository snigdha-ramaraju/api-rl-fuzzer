"""
Lightweight FastAPI server exposing endpoints for:
- Server health and supported policies
- OpenAPI specification parsing and ReactFlow graph generation
- Experiment tracking and analytics retrieval from SQLite
- Asynchronous fuzzing run execution and real-time status polling
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from openapi_spec_validator import validate
from pydantic import BaseModel, Field as PyField

# Ensure src/ and repo root are in sys.path for direct script execution
_src_dir = Path(__file__).resolve().parent
_repo_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from spec.parser import load_spec, parse_endpoints, Endpoint, Field
from spec.graph import build_dependency_graph, DependencyEdge
from experiment.tracker import ExperimentTracker
from experiment.runner import (
    ALL_POLICIES,
    DEFAULT_DB,
    REPO_ROOT,
    _run_random_policy,
    _run_budget_matched_random_policy,
    _run_greedy_policy,
    _run_bandit_policy,
    _run_ppo_policy,
    _run_cross_api_ppo_with_exploration,
)

app = FastAPI(
    title="API RL Fuzzer API",
    version="0.2.0",
    description="Backend API for RL-based API fuzzing agent, dependency graph visualization, experiment tracking, and run management.",
)

# Enable CORS for local web frontends (e.g. React/Vite on localhost:5173 or 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread-safe in-memory run tracking registry
_runs_lock = threading.Lock()
_runs: dict[str, dict[str, Any]] = {}


def get_tracker(db_path: Optional[str] = None, must_exist: bool = True) -> ExperimentTracker:
    """Resolve and instantiate an ExperimentTracker.

    Priority:
    1. Explicit db_path argument (if must_exist=True, file must already exist)
    2. FUZZER_DB_PATH environment variable
    3. DEFAULT_DB (runs/phase6_experiments.db)
    4. First existing .db file in runs/ directory
    """
    target_path: Path
    if db_path:
        target_path = Path(db_path)
        if not target_path.is_absolute():
            target_path = REPO_ROOT / target_path
        if must_exist and not target_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database file not found: {db_path}",
            )
    else:
        env_path = os.getenv("FUZZER_DB_PATH")
        if env_path:
            target_path = Path(env_path)
            if not target_path.is_absolute():
                target_path = REPO_ROOT / target_path
            if not target_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Database file specified by FUZZER_DB_PATH not found: {env_path}",
                )
        elif DEFAULT_DB.exists():
            target_path = DEFAULT_DB
        else:
            runs_dir = REPO_ROOT / "runs"
            existing_dbs = sorted(runs_dir.glob("*.db")) if runs_dir.exists() else []
            if existing_dbs:
                target_path = existing_dbs[0]
            else:
                target_path = DEFAULT_DB

    try:
        return ExperimentTracker(target_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize tracker with database {target_path}: {e}",
        )


@app.get("/api/health")
def health_check():
    """Report server status, version, and supported fuzzing policies."""
    resolved_db = None
    try:
        tracker = get_tracker()
        resolved_db = str(tracker.db_path)
    except Exception:
        pass

    return {
        "status": "healthy",
        "version": "0.2.0",
        "supported_policies": ALL_POLICIES,
        "default_db": resolved_db,
    }


@app.post("/api/spec/parse")
async def parse_spec(
    request: Request,
    file: Optional[UploadFile] = File(None),
    spec: Optional[str] = Form(None),
):
    """Accept an OpenAPI YAML/JSON upload or document, parse it using the existing
    parser and dependency graph, and return endpoint nodes and dependency edges for ReactFlow.
    """
    spec_content: Any = None

    if file is not None:
        try:
            raw_bytes = await file.read()
            text = raw_bytes.decode("utf-8")
            spec_content = yaml.safe_load(text)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read and parse uploaded spec file: {e}",
            )
    elif spec is not None:
        try:
            spec_content = yaml.safe_load(spec)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse form parameter 'spec': {e}",
            )
    else:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body_json = await request.json()
                if isinstance(body_json, dict) and "spec" in body_json:
                    val = body_json["spec"]
                    if isinstance(val, str):
                        spec_content = yaml.safe_load(val)
                    elif isinstance(val, dict):
                        spec_content = val
                    else:
                        spec_content = val
                else:
                    spec_content = body_json
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON body: {e}",
                )
        else:
            raw_body = await request.body()
            if raw_body:
                try:
                    text = raw_body.decode("utf-8")
                    spec_content = yaml.safe_load(text)
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to parse raw request body as YAML/JSON: {e}",
                    )

    if not spec_content or not isinstance(spec_content, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid OpenAPI specification document provided. Expected JSON or YAML dictionary.",
        )

    # Validate against OpenAPI 3.x schema
    try:
        validate(spec_content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OpenAPI validation error: {e}",
        )

    # Parse endpoints and build dependency graph using existing modules
    try:
        endpoints: list[Endpoint] = parse_endpoints(spec_content)
        edges: list[DependencyEdge]
        unresolved: list[dict]
        edges, unresolved = build_dependency_graph(endpoints)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse endpoints and build dependency graph: {e}",
        )

    # Build ReactFlow nodes with non-overlapping positions
    cols = 3
    x_spacing = 320
    y_spacing = 160
    nodes = []

    for idx, ep in enumerate(endpoints):
        col = idx % cols
        row = idx // cols
        nodes.append({
            "id": ep.operation_id,
            "type": "endpointNode",
            "position": {"x": col * x_spacing, "y": row * y_spacing},
            "data": {
                "label": f"{ep.method} {ep.path}",
                "operation_id": ep.operation_id,
                "method": ep.method,
                "path": ep.path,
                "tags": ep.tags,
                "description": ep.description,
                "path_params": [asdict(f) for f in ep.path_params],
                "query_params": [asdict(f) for f in ep.query_params],
                "header_params": [asdict(f) for f in ep.header_params],
                "request_body": [asdict(f) for f in ep.request_body],
                "responses": {k: [asdict(f) for f in v] for k, v in ep.responses.items()},
                "all_input_fields": [asdict(f) for f in ep.all_input_fields],
            },
        })

    # Build ReactFlow edges
    react_edges = []
    for idx, edge in enumerate(edges):
        react_edges.append({
            "id": f"e-{edge.producer_op}-{edge.consumer_op}-{edge.producer_field}-{edge.consumer_field}-{idx}",
            "source": edge.producer_op,
            "target": edge.consumer_op,
            "label": f"{edge.producer_field} -> {edge.consumer_field} ({edge.confidence})",
            "animated": edge.confidence == "exact",
            "data": {
                "producer_op": edge.producer_op,
                "producer_field": edge.producer_field,
                "consumer_op": edge.consumer_op,
                "consumer_field": edge.consumer_field,
                "confidence": edge.confidence,
            },
        })

    return {
        "title": spec_content.get("info", {}).get("title", "OpenAPI Specification"),
        "version": spec_content.get("info", {}).get("version", ""),
        "nodes": nodes,
        "edges": react_edges,
        "unresolved": unresolved,
        "endpoints_count": len(nodes),
        "edges_count": len(react_edges),
    }


@app.get("/api/experiments")
def get_experiments(
    policy: Optional[str] = Query(None, description="Filter experiments by policy name"),
    trial_id: Optional[int] = Query(None, description="Filter experiments by trial ID"),
    db_path: Optional[str] = Query(None, description="Optional custom path to SQLite database"),
):
    """Return existing experiment records from SQLite using ExperimentTracker schema."""
    tracker = get_tracker(db_path)
    try:
        experiments = tracker.get_experiments(policy=policy, trial_id=trial_id)
        return {
            "db_path": str(tracker.db_path),
            "total": len(experiments),
            "experiments": experiments,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query experiments: {e}",
        )


@app.get("/api/experiments/{experiment_id}")
def get_experiment_by_id(
    experiment_id: int,
    db_path: Optional[str] = Query(None, description="Optional custom path to SQLite database"),
):
    """Return available metrics, coverage, and anomaly summaries for an existing experiment."""
    tracker = get_tracker(db_path)
    exp = tracker.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found in database {tracker.db_path.name}",
        )

    try:
        metrics = tracker.compute_final_metrics(experiment_id)
        cov_summary = tracker.get_coverage_summary(experiment_id)
        coverage_list = [
            {"operation_id": op, "status_code": sc, "count": count}
            for (op, sc), count in cov_summary.items()
        ]
        anomalies = tracker.get_anomalies_summary(experiment_id)
        episodes = tracker.get_episodes(experiment_id)
        bug_discoveries = tracker.get_bug_discoveries(experiment_id)

        return {
            "db_path": str(tracker.db_path),
            "id": experiment_id,
            "experiment": exp,
            "metrics": metrics,
            "coverage": coverage_list,
            "anomalies": anomalies,
            "episodes": episodes,
            "bug_discoveries": bug_discoveries,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve experiment details for ID {experiment_id}: {e}",
        )


# =========================================================================
# Run Management: Background Execution & Status Polling
# =========================================================================

class RunStartRequest(BaseModel):
    policy: str = PyField(..., description="Fuzzing policy to run. Must be one of supported policies.")
    spec: Optional[Union[str, dict[str, Any]]] = PyField(None, description="OpenAPI specification as a YAML/JSON string or JSON dict.")
    spec_path: Optional[str] = PyField(None, description="Path to an existing OpenAPI spec file on disk.")
    n_episodes: int = PyField(5, description="Number of evaluation episodes.")
    max_steps_per_episode: int = PyField(20, description="Maximum steps per episode.")
    timesteps_total: int = PyField(256, description="Total timesteps budget (for PPO or budget-matched random).")
    seed: int = PyField(42, description="Random seed.")
    use_planted_bugs: bool = PyField(False, description="Enable planted bugs in mock environment.")
    db_path: Optional[str] = PyField(None, description="Target SQLite database path.")
    trial_id: int = PyField(0, description="Trial index.")
    explore_budget: Optional[int] = PyField(None, description="Explicit request budget for initial exploration phase.")
    explore_ratio: float = PyField(0.2, description="Fraction of total request budget allocated to exploration (default 0.2).")
    model_path: Optional[str] = PyField(None, description="Path to saved MaskablePPO model file for cross_api_ppo policy.")


def _execute_run_worker(
    run_id: str,
    spec_path: Path,
    policy: str,
    n_episodes: int,
    max_steps_per_episode: int,
    timesteps_total: int,
    seed: int,
    use_planted_bugs: bool,
    db_path: Path,
    trial_id: int,
    explore_budget: Optional[int] = None,
    explore_ratio: float = 0.2,
    model_path: Optional[str] = None,
):
    """Background worker executing real fuzzing policy rollouts and updating progress."""
    with _runs_lock:
        if run_id in _runs:
            _runs[run_id]["status"] = "running"
            _runs[run_id]["started_at"] = datetime.now(timezone.utc).isoformat()
            initial_stage = "exploring" if policy == "cross_api_ppo" else ("training" if policy in ("ppo", "ppo_llm") else "evaluating")
            initial_phase = "exploration" if policy == "cross_api_ppo" else ("training" if policy in ("ppo", "ppo_llm") else "evaluating")
            _runs[run_id]["progress"]["stage"] = initial_stage
            _runs[run_id]["progress"]["phase"] = initial_phase

    try:
        if policy == "random":
            exp_id = _run_random_policy(
                spec_path=spec_path,
                n_episodes=n_episodes,
                max_steps_per_episode=max_steps_per_episode,
                use_planted_bugs=use_planted_bugs,
                trial_seed=seed,
                db_path=db_path,
                trial_id=trial_id,
            )
        elif policy == "random_budget_matched":
            exp_id = _run_budget_matched_random_policy(
                spec_path=spec_path,
                n_episodes=n_episodes,
                max_steps_per_episode=max_steps_per_episode,
                timesteps_total=timesteps_total,
                use_planted_bugs=use_planted_bugs,
                trial_seed=seed,
                db_path=db_path,
                trial_id=trial_id,
            )
        elif policy == "greedy":
            exp_id = _run_greedy_policy(
                spec_path=spec_path,
                n_episodes=n_episodes,
                max_steps_per_episode=max_steps_per_episode,
                use_planted_bugs=use_planted_bugs,
                trial_seed=seed,
                db_path=db_path,
                trial_id=trial_id,
            )
        elif policy == "bandit":
            exp_id = _run_bandit_policy(
                spec_path=spec_path,
                n_episodes=n_episodes,
                max_steps_per_episode=max_steps_per_episode,
                use_planted_bugs=use_planted_bugs,
                trial_seed=seed,
                db_path=db_path,
                trial_id=trial_id,
            )
        elif policy in ("ppo", "ppo_llm"):
            exp_id = _run_ppo_policy(
                spec_path=spec_path,
                n_episodes=n_episodes,
                max_steps_per_episode=max_steps_per_episode,
                timesteps_total=timesteps_total,
                trial_seed=seed,
                db_path=db_path,
                trial_id=trial_id,
                enable_brainstormer=(policy == "ppo_llm"),
                use_planted_bugs=use_planted_bugs,
            )
        elif policy == "cross_api_ppo":
            def _on_progress(p_info: dict):
                with _runs_lock:
                    if run_id in _runs:
                        _runs[run_id]["progress"].update({
                            "stage": p_info.get("stage", _runs[run_id]["progress"].get("stage")),
                            "phase": p_info.get("phase", _runs[run_id]["progress"].get("phase")),
                            "requests_made": p_info.get("requests_made", 0),
                            "exploration_requests": p_info.get("exploration_requests", 0),
                            "ppo_requests": p_info.get("ppo_requests", 0),
                            "coverage_count": p_info.get("coverage_count", 0),
                            "anomalies_found": p_info.get("anomalies_found", 0),
                        })

            exp_id = _run_cross_api_ppo_with_exploration(
                spec_path=spec_path,
                n_episodes=n_episodes,
                max_steps_per_episode=max_steps_per_episode,
                timesteps_total=timesteps_total,
                explore_budget=explore_budget,
                explore_ratio=explore_ratio,
                model_path=model_path,
                use_planted_bugs=use_planted_bugs,
                trial_seed=seed,
                db_path=db_path,
                trial_id=trial_id,
                on_progress=_on_progress,
            )
        else:
            raise ValueError(f"Unsupported policy: '{policy}'")

        # Fetch actual results directly from ExperimentTracker
        tracker = ExperimentTracker(db_path)
        metrics = tracker.compute_final_metrics(exp_id) or {}
        cov = tracker.get_coverage_summary(exp_id)
        anom = tracker.get_anomalies_summary(exp_id)
        episodes = tracker.get_episodes(exp_id)

        with _runs_lock:
            if run_id in _runs:
                _runs[run_id]["status"] = "completed"
                _runs[run_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                _runs[run_id]["experiment_id"] = exp_id
                _runs[run_id]["progress"].update({
                    "stage": "completed",
                    "phase": "completed",
                    "current_episode": len(episodes),
                    "total_episodes": n_episodes,
                    "requests_made": metrics.get("total_requests", 0),
                    "exploration_requests": metrics.get("exploration_requests", 0),
                    "ppo_requests": metrics.get("ppo_requests", 0),
                    "anomalies_found": metrics.get("unique_anomalies_found", len(anom)),
                    "coverage_count": len(cov),
                })
    except Exception as e:
        with _runs_lock:
            if run_id in _runs:
                _runs[run_id]["status"] = "failed"
                _runs[run_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
                _runs[run_id]["error"] = str(e)
                _runs[run_id]["progress"]["stage"] = "failed"


@app.post("/api/runs/start")
def start_run(req: RunStartRequest):
    """Start a real fuzzing experiment asynchronously in the background.

    Validates policy, step budget, and OpenAPI spec. Returns run ID for polling.
    """
    # 1. Policy validation
    if req.policy not in ALL_POLICIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported policy '{req.policy}'. Supported policies: {ALL_POLICIES}",
        )

    # 2. Step and episode budget validation
    if req.n_episodes < 1 or req.n_episodes > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"n_episodes must be between 1 and 500, got {req.n_episodes}",
        )
    if req.max_steps_per_episode < 1 or req.max_steps_per_episode > 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"max_steps_per_episode must be between 1 and 200, got {req.max_steps_per_episode}",
        )
    if req.timesteps_total < 1 or req.timesteps_total > 100_000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"timesteps_total must be between 1 and 100000, got {req.timesteps_total}",
        )

    # 3. Spec resolution & OpenAPI schema validation
    resolved_spec_path: Path
    if req.spec_path:
        spec_file = Path(req.spec_path)
        if not spec_file.is_absolute():
            spec_file = REPO_ROOT / spec_file
        if not spec_file.exists() or not spec_file.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Specification file not found at: {req.spec_path}",
            )
        try:
            raw_spec = load_spec(spec_file)
            validate(raw_spec)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"OpenAPI validation error in spec_path: {e}",
            )
        resolved_spec_path = spec_file
    elif req.spec:
        raw_spec = None
        if isinstance(req.spec, dict):
            raw_spec = req.spec
        elif isinstance(req.spec, str):
            try:
                raw_spec = yaml.safe_load(req.spec)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to parse OpenAPI YAML/JSON string: {e}",
                )
        if not raw_spec or not isinstance(raw_spec, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OpenAPI specification must be a valid dictionary.",
            )
        try:
            validate(raw_spec)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"OpenAPI validation error: {e}",
            )

        # Write to safe runs/specs directory for runner to access
        specs_dir = REPO_ROOT / "runs" / "specs"
        specs_dir.mkdir(parents=True, exist_ok=True)
        run_spec_file = specs_dir / f"spec_{uuid.uuid4().hex[:12]}.yaml"
        with open(run_spec_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw_spec, f)
        resolved_spec_path = run_spec_file
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An OpenAPI specification must be provided via 'spec' or 'spec_path'.",
        )

    # 4. Resolve target SQLite database
    tracker = get_tracker(req.db_path, must_exist=False)
    resolved_db_path = tracker.db_path

    # 5. Create run metadata record
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()
    run_record = {
        "run_id": run_id,
        "status": "queued",
        "policy": req.policy,
        "created_at": now_iso,
        "started_at": None,
        "completed_at": None,
        "experiment_id": None,
        "error": None,
        "db_path": str(resolved_db_path),
        "spec_path": str(resolved_spec_path),
        "config": {
            "policy": req.policy,
            "n_episodes": req.n_episodes,
            "max_steps_per_episode": req.max_steps_per_episode,
            "timesteps_total": req.timesteps_total,
            "seed": req.seed,
            "use_planted_bugs": req.use_planted_bugs,
            "trial_id": req.trial_id,
            "explore_budget": req.explore_budget,
            "explore_ratio": req.explore_ratio,
            "model_path": req.model_path,
        },
        "progress": {
            "stage": "queued",
            "phase": "queued",
            "current_episode": 0,
            "total_episodes": req.n_episodes,
            "max_steps_per_episode": req.max_steps_per_episode,
            "requests_made": 0,
            "exploration_requests": 0,
            "ppo_requests": 0,
            "anomalies_found": 0,
            "coverage_count": 0,
        },
    }

    with _runs_lock:
        _runs[run_id] = run_record

    # 6. Execute in background daemon thread (non-blocking)
    thread = threading.Thread(
        target=_execute_run_worker,
        kwargs={
            "run_id": run_id,
            "spec_path": resolved_spec_path,
            "policy": req.policy,
            "n_episodes": req.n_episodes,
            "max_steps_per_episode": req.max_steps_per_episode,
            "timesteps_total": req.timesteps_total,
            "seed": req.seed,
            "use_planted_bugs": req.use_planted_bugs,
            "db_path": resolved_db_path,
            "trial_id": req.trial_id,
            "explore_budget": req.explore_budget,
            "explore_ratio": req.explore_ratio,
            "model_path": req.model_path,
        },
        daemon=True,
    )
    thread.start()

    return {
        "run_id": run_id,
        "status": "queued",
        "policy": req.policy,
        "message": f"Fuzzing experiment started with policy '{req.policy}'.",
        "created_at": now_iso,
    }


@app.get("/api/runs/{run_id}/status")
@app.get("/api/runs/{run_id}")
def get_run_status(run_id: str):
    """Return real run state, progress, errors, and resulting experiment ID."""
    with _runs_lock:
        run = _runs.get(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found",
            )
        run_data = dict(run)
        run_data["progress"] = dict(run.get("progress", {}))
        run_data["config"] = dict(run.get("config", {}))

    return run_data


@app.get("/api/runs")
def list_runs(limit: int = Query(50, ge=1, le=200)):
    """List recent fuzzing runs in reverse chronological order."""
    with _runs_lock:
        items = [dict(r) for r in _runs.values()]
    items.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return {
        "total": len(items),
        "runs": items[:limit],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
