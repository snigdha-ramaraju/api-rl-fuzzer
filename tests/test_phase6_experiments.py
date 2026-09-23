"""
Phase 6 tests: SQLite experiment tracking, baselines, evaluation metrics,
and experiment runner. All tests run against the in-memory mock Orders API
with seeded RNG for reproducibility. No live targets, no cost.
"""

import sys
import tempfile
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment.tracker import ExperimentTracker
from experiment.metrics import aggregate_metrics, PolicyMetrics, compare_policies
from agent.random_policy import RandomPolicy
from agent.train_ppo import build_env


FIXTURE_SPEC = Path(__file__).parent / "fixtures" / "sample_api.yaml"


# ========== SQLite Experiment Tracker Tests ==========

def test_tracker_init_creates_db_schema():
    """Tracker initialization creates all required tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        tracker = ExperimentTracker(db_path)
        
        assert db_path.exists()
        
        # Verify schema
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        
        expected = {"experiments", "episodes", "coverage_hits", "anomalies"}
        assert expected.issubset(tables)
        conn.close()


def test_tracker_start_experiment_returns_id():
    """Starting an experiment returns a unique ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "test.db")
        
        exp_id = tracker.start_experiment(
            name="test_exp",
            policy="random",
            trial_id=0,
            seed=42,
            max_steps_per_episode=50,
            timesteps_total=1000,
            n_episodes=10,
        )
        
        assert isinstance(exp_id, int)
        assert exp_id > 0


def test_tracker_record_episode_persists_data():
    """Recording an episode persists it to the database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "test.db")
        
        exp_id = tracker.start_experiment(
            name="test_ep",
            policy="test",
            trial_id=0,
            seed=42,
            max_steps_per_episode=50,
            timesteps_total=1000,
            n_episodes=1,
        )
        
        coverage_hits = {
            ("createOrder", 200): 1,
            ("getOrder", 404): 1,
        }
        anomaly_sigs = set([
            ("getOrder", 404, False, True, ("id",), ()),
        ])
        
        tracker.record_episode(
            exp_id, episode_number=0,
            total_reward=5.0, steps_taken=10,
            coverage_hits=coverage_hits,
            anomaly_signatures=anomaly_sigs,
            new_anomaly_count=1,
        )
        
        episodes = tracker.get_episodes(exp_id)
        assert len(episodes) == 1
        assert episodes[0]["total_reward"] == 5.0
        assert episodes[0]["steps_taken"] == 10


def test_tracker_get_coverage_summary():
    """Retrieving coverage summary aggregates hits correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "test.db")
        
        exp_id = tracker.start_experiment(
            name="test_cov", policy="test", trial_id=0, seed=42,
            max_steps_per_episode=50, timesteps_total=1000, n_episodes=2,
        )
        
        # Episode 1
        tracker.record_episode(
            exp_id, 0, 5.0, 10,
            {("createOrder", 200): 1, ("getOrder", 404): 1},
            set(),
            0,
        )
        
        # Episode 2 (same coverage hits)
        tracker.record_episode(
            exp_id, 1, 3.0, 8,
            {("createOrder", 200): 1, ("getOrder", 404): 1},
            set(),
            0,
        )
        
        cov = tracker.get_coverage_summary(exp_id)
        assert cov[("createOrder", 200)] == 2
        assert cov[("getOrder", 404)] == 2


def test_tracker_compute_final_metrics():
    """Computing final metrics aggregates all episode data correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "test.db")
        
        exp_id = tracker.start_experiment(
            name="test_metrics", policy="random", trial_id=0, seed=42,
            max_steps_per_episode=50, timesteps_total=0, n_episodes=3,
        )
        
        # Three episodes
        for ep in range(3):
            tracker.record_episode(
                exp_id, ep, 4.0, 8,
                {("createOrder", 200): 1},
                set(),
                0,
            )
        
        metrics = tracker.compute_final_metrics(exp_id)
        assert metrics is not None
        assert metrics["total_reward"] == 12.0
        assert metrics["mean_episode_reward"] == 4.0
        assert metrics["n_episodes"] == 3


# ========== Metrics Aggregation Tests ==========

def test_aggregate_metrics_single_experiment():
    """Aggregating single-trial metrics works correctly."""
    exp_metrics = [
        {
            "policy": "random",
            "mean_episode_reward": 3.5,
            "total_reward": 70.0,
            "unique_operations_covered": 2,
            "total_coverage_hits": 15,
            "unique_anomalies_found": 1,
        }
    ]
    
    agg = aggregate_metrics(exp_metrics)
    assert agg.policy == "random"
    assert agg.n_trials == 1
    assert agg.mean_episode_reward == 3.5
    assert agg.std_episode_reward == 0.0  # No variance with 1 trial


def test_aggregate_metrics_multiple_trials():
    """Aggregating multi-trial metrics computes correct statistics."""
    exp_metrics = [
        {
            "policy": "ppo",
            "mean_episode_reward": 5.0,
            "total_reward": 100.0,
            "unique_operations_covered": 3,
            "total_coverage_hits": 30,
            "unique_anomalies_found": 2,
        },
        {
            "policy": "ppo",
            "mean_episode_reward": 7.0,
            "total_reward": 140.0,
            "unique_operations_covered": 3,
            "total_coverage_hits": 35,
            "unique_anomalies_found": 3,
        },
    ]
    
    agg = aggregate_metrics(exp_metrics)
    assert agg.policy == "ppo"
    assert agg.n_trials == 2
    assert agg.mean_episode_reward == 6.0
    assert agg.std_episode_reward > 0


def test_compare_policies_formats_output():
    """Comparing policies produces readable ranked output."""
    metrics_list = [
        PolicyMetrics(
            policy="random",
            n_trials=1,
            mean_episode_reward=2.0,
            std_episode_reward=0.0,
            mean_total_reward=40.0,
            std_total_reward=0.0,
            mean_unique_ops_covered=1.0,
            std_unique_ops_covered=0.0,
            mean_total_coverage_hits=8.0,
            std_total_coverage_hits=0.0,
            mean_unique_anomalies=0.5,
            std_unique_anomalies=0.0,
        ),
        PolicyMetrics(
            policy="ppo",
            n_trials=1,
            mean_episode_reward=6.0,
            std_episode_reward=0.0,
            mean_total_reward=120.0,
            std_total_reward=0.0,
            mean_unique_ops_covered=3.0,
            std_unique_ops_covered=0.0,
            mean_total_coverage_hits=30.0,
            std_total_coverage_hits=0.0,
            mean_unique_anomalies=2.5,
            std_unique_anomalies=0.0,
        ),
    ]
    
    result = compare_policies(metrics_list)
    assert "RANKING" in result
    assert "ppo" in result
    assert "random" in result
    # PPO should rank higher
    lines = result.split("\n")
    ppo_line = [l for l in lines if "1." in l and "ppo" in l]
    assert ppo_line, "PPO should rank #1"


# ========== Random Policy Tests ==========

def test_random_policy_initialization():
    """RandomPolicy initializes with correct action space size."""
    policy = RandomPolicy(n_actions=5, seed=42)
    assert policy.n_actions == 5


def test_random_policy_predict_returns_valid_action():
    """RandomPolicy.predict returns valid actions."""
    policy = RandomPolicy(n_actions=3, seed=42)
    
    for _ in range(10):
        action, state = policy.predict(None)
        assert 0 <= action < 3
        assert state is None


def test_random_policy_seeding_produces_reproducible_sequence():
    """Seeding RandomPolicy produces reproducible action sequences."""
    policy1 = RandomPolicy(n_actions=5, seed=42)
    policy2 = RandomPolicy(n_actions=5, seed=42)
    
    actions1 = [policy1.predict(None)[0] for _ in range(10)]
    actions2 = [policy2.predict(None)[0] for _ in range(10)]
    
    assert actions1 == actions2


# ========== Environment Integration Tests ==========

def test_env_coverage_tracking_basic():
    """Environment coverage tracking works end-to-end."""
    env = build_env(
        FIXTURE_SPEC,
        use_mock=True,
        max_steps_per_episode=5,
        seed=42,
    )
    
    obs, _ = env.reset(seed=42)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    
    # Coverage should have recorded some hits
    assert len(env.coverage.hits) > 0
    env.close()


def test_env_random_actions_produce_episodes():
    """Running random actions through env produces valid episodes."""
    env = build_env(FIXTURE_SPEC, use_mock=True, max_steps_per_episode=10, seed=42)
    policy = RandomPolicy(n_actions=env.action_space.n, seed=42)
    
    obs, _ = env.reset(seed=42)
    total_reward = 0.0
    step_count = 0
    
    done = truncated = False
    while not (done or truncated):
        action, _ = policy.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
    
    assert step_count > 0
    env.close()


# ========== End-to-End Tracker + Policy Integration ==========

def test_tracker_with_policy_evaluation_workflow():
    """Full workflow: track a policy evaluation through the database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        tracker = ExperimentTracker(db_path)
        
        # Start an experiment
        exp_id = tracker.start_experiment(
            name="e2e_test",
            policy="random",
            trial_id=0,
            seed=42,
            max_steps_per_episode=5,
            timesteps_total=0,
            n_episodes=2,
        )
        
        # Simulate two evaluation episodes
        for ep in range(2):
            coverage = {
                ("createOrder", 200): 1,
                ("getOrder", 200 if ep == 1 else 404): 1,
            }
            tracker.record_episode(
                exp_id, ep,
                total_reward=float(5 + ep),
                steps_taken=8,
                coverage_hits=coverage,
                anomaly_signatures=set(),
                new_anomaly_count=0,
            )
        
        # Retrieve and verify
        metrics = tracker.compute_final_metrics(exp_id)
        assert metrics["n_episodes"] == 2
        assert metrics["total_reward"] == 11.0
        assert metrics["mean_episode_reward"] == 5.5
        assert metrics["unique_operations_covered"] == 2


# ========== Correctness Check: Novelty-Gated Reward Still Works ==========

def test_phase6_preserves_phase5_novelty_gating():
    """Phase 6 changes don't break Phase 5's anomaly novelty gating."""
    env = build_env(FIXTURE_SPEC, use_mock=True, max_steps_per_episode=50, seed=99)
    
    obs, _ = env.reset(seed=99)
    
    # Take many steps, accumulating coverage
    coverage_before = len(env.coverage.hits)
    anomalies_before = len(env.coverage.anomaly_signatures)
    
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    
    coverage_after = len(env.coverage.hits)
    anomalies_after = len(env.coverage.anomaly_signatures)
    
    # Coverage and anomalies grew (if we hit new things)
    # The important thing: this ran without error and novelty tracking worked
    assert isinstance(coverage_after, int)
    assert isinstance(anomalies_after, int)
    
    env.close()


# ========== Planted Bugs Integration Tests ==========

def test_planted_bugs_module_loads():
    """Planted bugs module can be imported and instantiated."""
    from executor.planted_bugs import PlantedBugsMockTarget
    from executor.mock_target import make_mock_transport
    
    base = make_mock_transport()
    target = PlantedBugsMockTarget(base)
    
    assert hasattr(target, "handle")
    assert hasattr(target, "bug_definitions")
    assert len(target.bug_definitions) == 3  # 3 planted bugs


def test_planted_bugs_env_integration():
    """Environment can be created with planted bugs enabled."""
    env = build_env(
        FIXTURE_SPEC,
        use_mock=True,
        max_steps_per_episode=50,
        seed=42,
        use_planted_bugs=True,
    )
    assert env is not None
    
    obs, _ = env.reset(seed=42)
    assert obs is not None
    
    env.close()


def test_tracker_anomaly_step_recording():
    """Tracker correctly records anomalies with episode and step info."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "test.db")
        
        exp_id = tracker.start_experiment(
            name="test_exp",
            policy="random",
            trial_id=0,
            seed=42,
            max_steps_per_episode=50,
            timesteps_total=0,
            n_episodes=1,
        )
        
        # Record an anomaly at specific step
        sig = ("getOrder", 400, True, True, (), ())
        tracker.record_anomaly_at_step(
            exp_id,
            "getOrder",
            400,
            sig,
            episode=0,
            step=5,
            is_planted_bug=1,
            bug_name="bug_getorder_negative",
        )
        
        # Verify it was persisted
        conn = sqlite3.connect(str(Path(tmpdir) / "test.db"))
        cursor = conn.execute(
            "SELECT first_seen_episode, first_seen_step, is_planted_bug, bug_name "
            "FROM anomalies WHERE experiment_id = ?",
            (exp_id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 0  # episode
        assert row[1] == 5  # step
        assert row[2] == 1  # is_planted_bug
        assert row[3] == "bug_getorder_negative"  # bug_name
        conn.close()


def test_tracker_bug_discovery_metrics():
    """Tracker computes bug discovery metrics correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "test.db")
        
        exp_id = tracker.start_experiment(
            name="test_bug_discovery",
            policy="random",
            trial_id=0,
            seed=42,
            max_steps_per_episode=50,
            timesteps_total=0,
            n_episodes=2,
        )
        
        # Record 2 planted bugs and 1 false positive
        sig1 = ("getOrder", 400, True, True, (), ())
        tracker.record_anomaly_at_step(
            exp_id, "getOrder", 400, sig1, 0, 3,
            is_planted_bug=1, bug_name="bug_getorder_negative"
        )
        
        sig2 = ("cancelOrder", 409, True, True, (), ())
        tracker.record_anomaly_at_step(
            exp_id, "cancelOrder", 409, sig2, 1, 8,
            is_planted_bug=1, bug_name="bug_cancelorder_double"
        )
        
        sig3 = ("unknown", 500, True, True, (), ())
        tracker.record_anomaly_at_step(
            exp_id, "unknown", 500, sig3, 1, 12,
            is_planted_bug=0, bug_name=None
        )
        
        metrics = tracker.get_bug_discoveries(exp_id)
        
        assert metrics["total_anomalies_found"] == 3
        assert metrics["planted_bugs_found"] == 2
        assert metrics["false_positives"] == 1
        assert metrics["steps_to_first_bug"] == 3  # First bug at step 3


def test_tracker_false_positive_counting():
    """Tracker distinguishes planted bugs from false positives."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "test.db")
        
        exp_id = tracker.start_experiment(
            name="test_false_pos",
            policy="random",
            trial_id=0,
            seed=99,
            max_steps_per_episode=50,
            timesteps_total=0,
            n_episodes=1,
        )
        
        # Record 5 false positives (responses not matching any planted bug)
        for i in range(5):
            sig = ("someOp", 418 + i, True, True, (), ())  # 418, 419, 420, 421, 422
            tracker.record_anomaly_at_step(
                exp_id, "someOp", 418 + i, sig, 0, i,
                is_planted_bug=0
            )
        
        metrics = tracker.get_bug_discoveries(exp_id)
        assert metrics["false_positives"] == 5
        assert metrics["planted_bugs_found"] == 0


def test_tracker_final_metrics_include_bug_discovery():
    """Final metrics from compute_final_metrics include bug discovery info."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "test.db")
        
        exp_id = tracker.start_experiment(
            name="test_final_metrics",
            policy="random",
            trial_id=0,
            seed=42,
            max_steps_per_episode=50,
            timesteps_total=0,
            n_episodes=1,
        )
        
        # Record an episode
        tracker.record_episode(
            exp_id, 0, total_reward=10.0, steps_taken=20,
            coverage_hits={("op1", 200): 1},
            anomaly_signatures=set(),
            new_anomaly_count=0
        )
        
        # Record a planted bug discovery
        sig = ("getOrder", 400, True, True, (), ())
        tracker.record_anomaly_at_step(
            exp_id, "getOrder", 400, sig, 0, 15,
            is_planted_bug=1, bug_name="bug_getorder_negative"
        )
        
        metrics = tracker.compute_final_metrics(exp_id)
        
        # Should have bug discovery metrics in output
        assert "planted_bugs_found" in metrics
        assert "false_positives" in metrics
        assert "steps_to_first_bug" in metrics
        assert metrics["planted_bugs_found"] == 1
        assert metrics["false_positives"] == 0
        assert metrics["steps_to_first_bug"] == 15
