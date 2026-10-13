"""
Phase 6: SQLite-backed experiment tracker.
Persists coverage, anomalies, and experiment metadata across runs.
No external dependencies beyond sqlite3 (built-in).
"""

from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict


class ExperimentTracker:
    """SQLite-backed experiment tracker. One DB file per experiment run."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                policy TEXT NOT NULL,
                trial_id INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                max_steps_per_episode INTEGER,
                timesteps_total INTEGER,
                n_episodes INTEGER
            );
            
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                episode_number INTEGER NOT NULL,
                total_reward REAL NOT NULL,
                steps_taken INTEGER NOT NULL,
                coverage_count INTEGER NOT NULL,
                anomaly_count INTEGER NOT NULL,
                new_anomalies INTEGER NOT NULL,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            );
            
            CREATE TABLE IF NOT EXISTS coverage_hits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                operation_id TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                count INTEGER DEFAULT 1,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id),
                UNIQUE(experiment_id, operation_id, status_code)
            );
            
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                operation_id TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                signature TEXT NOT NULL,
                first_seen_episode INTEGER NOT NULL,
                first_seen_step INTEGER NOT NULL,
                count INTEGER DEFAULT 1,
                is_planted_bug INTEGER DEFAULT 0,
                bug_name TEXT DEFAULT NULL,
                UNIQUE(experiment_id, operation_id, status_code, signature),
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_experiments_policy ON experiments(policy);
            CREATE INDEX IF NOT EXISTS idx_experiments_trial ON experiments(trial_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_experiment ON episodes(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_coverage_experiment ON coverage_hits(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_anomalies_experiment ON anomalies(experiment_id);
            """)
            conn.commit()
        finally:
            conn.close()

    def start_experiment(self, name: str, policy: str, trial_id: int, seed: int,
                        max_steps_per_episode: int, timesteps_total: int,
                        n_episodes: int) -> int:
        """Start a new experiment; return its ID."""
        conn = self._get_conn()
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """
                INSERT INTO experiments
                (name, policy, trial_id, seed, timestamp, max_steps_per_episode,
                 timesteps_total, n_episodes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (name, policy, trial_id, seed, timestamp, max_steps_per_episode,
                 timesteps_total, n_episodes)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def record_episode(self, experiment_id: int, episode_number: int,
                      total_reward: float, steps_taken: int,
                      coverage_hits: dict[tuple[str, int], int],
                      anomaly_signatures: set[tuple],
                      new_anomaly_count: int):
        """Record a completed episode."""
        conn = self._get_conn()
        try:
            coverage_count = len(coverage_hits)
            anomaly_count = len(anomaly_signatures)

            conn.execute(
                """
                INSERT INTO episodes
                (experiment_id, episode_number, total_reward, steps_taken,
                 coverage_count, anomaly_count, new_anomalies)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (experiment_id, episode_number, total_reward, steps_taken,
                 coverage_count, anomaly_count, new_anomaly_count)
            )

            # Record coverage hits
            for (op_id, status_code), count in coverage_hits.items():
                conn.execute(
                    """
                    INSERT INTO coverage_hits (experiment_id, operation_id, status_code, count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(experiment_id, operation_id, status_code)
                    DO UPDATE SET count = count + ?
                    """,
                    (experiment_id, op_id, status_code, count, count)
                )

            conn.commit()
        finally:
            conn.close()

    def record_anomalies(self, experiment_id: int, operation_id: str,
                        status_code: int, anomaly_signatures: set[tuple],
                        first_episode: int, first_step: int = 0,
                        is_planted_bug: int = 0, bug_name: str = None):
        """Record discovered anomalies."""
        conn = self._get_conn()
        try:
            for sig in anomaly_signatures:
                sig_str = json.dumps(sig)
                conn.execute(
                    """
                    INSERT INTO anomalies
                    (experiment_id, operation_id, status_code, signature, first_seen_episode, 
                     first_seen_step, count, is_planted_bug, bug_name)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT(experiment_id, operation_id, status_code, signature)
                    DO UPDATE SET count = count + 1
                    """,
                    (experiment_id, operation_id, status_code, sig_str, first_episode,
                     first_step, is_planted_bug, bug_name)
                )
            conn.commit()
        finally:
            conn.close()

    def record_anomaly_at_step(self, experiment_id: int, operation_id: str,
                              status_code: int, signature: tuple,
                              episode: int, step: int,
                              is_planted_bug: int = 0, bug_name: str = None):
        """Record a single anomaly with episode and step information."""
        conn = self._get_conn()
        try:
            sig_str = json.dumps(signature)
            conn.execute(
                """
                INSERT INTO anomalies
                (experiment_id, operation_id, status_code, signature, first_seen_episode,
                 first_seen_step, count, is_planted_bug, bug_name)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(experiment_id, operation_id, status_code, signature)
                DO UPDATE SET count = count + 1
                """,
                (experiment_id, operation_id, status_code, sig_str, episode,
                 step, is_planted_bug, bug_name)
            )
            conn.commit()
        finally:
            conn.close()

    def get_experiment(self, experiment_id: int) -> dict:
        """Fetch a single experiment's metadata."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_experiments(self, policy: str = None, trial_id: int = None) -> list[dict]:
        """Fetch experiments, optionally filtered by policy or trial_id."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM experiments WHERE 1=1"
            params = []
            if policy:
                query += " AND policy = ?"
                params.append(policy)
            if trial_id is not None:
                query += " AND trial_id = ?"
                params.append(trial_id)
            query += " ORDER BY id"

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_episodes(self, experiment_id: int) -> list[dict]:
        """Fetch all episodes for an experiment."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE experiment_id = ? ORDER BY episode_number",
                (experiment_id,)
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_coverage_summary(self, experiment_id: int) -> dict[tuple[str, int], int]:
        """Get final coverage hit counts for an experiment."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT operation_id, status_code, count FROM coverage_hits WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchall()
            return {(row[0], row[1]): row[2] for row in rows}
        finally:
            conn.close()

    def get_anomalies_summary(self, experiment_id: int) -> list[dict]:
        """Get all anomalies discovered in an experiment."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT operation_id, status_code, signature, first_seen_episode, count
                FROM anomalies WHERE experiment_id = ? ORDER BY first_seen_episode
                """,
                (experiment_id,)
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_bug_discoveries(self, experiment_id: int) -> dict:
        """Get bug discovery metrics for an experiment."""
        conn = self._get_conn()
        try:
            # Total distinct anomalies
            total_anom = conn.execute(
                "SELECT COUNT(*) as count FROM anomalies WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchone()
            total_anomalies = total_anom["count"] if total_anom else 0

            # Planted bugs (genuinely triggerable bugs)
            planted = conn.execute(
                "SELECT COUNT(*) as count FROM anomalies WHERE experiment_id = ? AND is_planted_bug = 1",
                (experiment_id,)
            ).fetchone()
            planted_bugs_found = planted["count"] if planted else 0

            # False positives (anomalies not matching planted bugs)
            false_pos = conn.execute(
                "SELECT COUNT(*) as count FROM anomalies WHERE experiment_id = ? AND is_planted_bug = 0",
                (experiment_id,)
            ).fetchone()
            false_positives = false_pos["count"] if false_pos else 0

            # First PLANTED bug found. The runner stores first_seen_step as a
            # request index that is monotonic across the whole evaluation run,
            # so MIN(first_seen_step) is the earliest planted bug. Incidental
            # (non-planted) anomalies are deliberately excluded: an undeclared
            # 404 at request 2 is not "the first bug". None if no planted bug
            # was found.
            first_bug = conn.execute(
                """
                SELECT MIN(first_seen_step) as step
                FROM anomalies WHERE experiment_id = ? AND is_planted_bug = 1
                """,
                (experiment_id,)
            ).fetchone()
            steps_to_first_bug = first_bug["step"] if first_bug and first_bug["step"] is not None else None

            return {
                "total_anomalies_found": total_anomalies,
                "planted_bugs_found": planted_bugs_found,
                "false_positives": false_positives,
                "steps_to_first_bug": steps_to_first_bug,
            }
        finally:
            conn.close()

    def compute_final_metrics(self, experiment_id: int) -> dict:
        """Compute aggregate metrics for an experiment."""
        conn = self._get_conn()
        try:
            # Experiment metadata
            exp = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if not exp:
                return None

            exp_dict = dict(exp)

            # Episodes
            episodes = conn.execute(
                "SELECT * FROM episodes WHERE experiment_id = ? ORDER BY episode_number",
                (experiment_id,)
            ).fetchall()
            episode_list = [dict(e) for e in episodes]

            # Coverage
            coverage_rows = conn.execute(
                "SELECT COUNT(DISTINCT operation_id) as unique_ops, COUNT(*) as total_hits FROM coverage_hits WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchone()
            coverage_dict = dict(coverage_rows) if coverage_rows else {}

            # Anomalies
            anomaly_rows = conn.execute(
                "SELECT COUNT(*) as unique_anomalies FROM anomalies WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchone()
            anomaly_count = dict(anomaly_rows)["unique_anomalies"] if anomaly_rows else 0

            # Aggregate metrics
            total_reward = sum(e["total_reward"] for e in episode_list) if episode_list else 0.0
            mean_reward = total_reward / len(episode_list) if episode_list else 0.0
            mean_steps = sum(e["steps_taken"] for e in episode_list) / len(episode_list) if episode_list else 0

            # Get bug discovery metrics
            bug_metrics = self.get_bug_discoveries(experiment_id)

            return {
                "experiment_id": experiment_id,
                "name": exp_dict["name"],
                "policy": exp_dict["policy"],
                "trial_id": exp_dict["trial_id"],
                "seed": exp_dict["seed"],
                "n_episodes": exp_dict["n_episodes"],
                "total_reward": total_reward,
                "mean_episode_reward": mean_reward,
                "mean_steps_per_episode": mean_steps,
                "unique_operations_covered": coverage_dict.get("unique_ops", 0),
                "total_coverage_hits": coverage_dict.get("total_hits", 0),
                "unique_anomalies_found": anomaly_count,
                "planted_bugs_found": bug_metrics.get("planted_bugs_found", 0),
                "false_positives": bug_metrics.get("false_positives", 0),
                "steps_to_first_bug": bug_metrics.get("steps_to_first_bug", None),
                "episodes": episode_list,
            }
        finally:
            conn.close()
