"""
In-memory coverage tracker: which (operation_id, status_code) combinations
have been observed so far in the current episode. Feeds both the reward
function (novelty bonus) and the environment's observation vector.

Also tracks per-episode ANOMALY novelty (`record_anomaly`), independently
of plain (op, status) coverage: the reward's anomaly bonus is gated on
this so that repeatedly triggering the identical anomaly (same operation,
same status, same missing/mismatched-type fields) cannot be farmed for
unlimited reward -- see env/reward.py. A genuinely different anomaly
(e.g. a different missing field on the same status code) is still
considered novel and still counted.

No persistence yet -- cross-run/SQLite-backed coverage is a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spec.oracle import OracleResult


@dataclass
class CoverageTracker:
    endpoint_ids: list[str]
    hits: dict[tuple[str, int], int] = field(default_factory=dict)  # (op_id, status) -> count
    anomaly_signatures: set[tuple] = field(default_factory=set)

    def record(self, operation_id: str, status_code: int) -> bool:
        """Records a hit; returns True if this (op, status) pair is NEW."""
        key = (operation_id, status_code)
        is_new = key not in self.hits
        self.hits[key] = self.hits.get(key, 0) + 1
        return is_new

    def record_anomaly(self, operation_id: str, status_code: int,
                        oracle_result: OracleResult) -> bool:
        """Records a specific anomaly SIGNATURE for this episode and
        returns True if this exact anomaly hasn't been seen before this
        episode. The signature is deliberately more specific than the
        plain (op, status) coverage key -- it also captures *how* the
        response was anomalous (undeclared status vs. which fields were
        missing/mismatched) -- so a genuinely different anomaly on the
        same status code still counts as novel, while calling the same
        endpoint over and over and getting the identical anomaly back
        does not."""
        signature = (
            operation_id,
            status_code,
            oracle_result.status_known,
            oracle_result.shape_valid,
            tuple(sorted(oracle_result.missing_fields)),
            tuple(sorted(oracle_result.unexpected_type_fields)),
        )
        is_new = signature not in self.anomaly_signatures
        self.anomaly_signatures.add(signature)
        return is_new

    def endpoint_covered(self, operation_id: str) -> bool:
        return any(op == operation_id for (op, _status) in self.hits)

    def coverage_vector(self) -> list[float]:
        """One float per known endpoint: 1.0 if ever hit (any status), else 0.0."""
        return [1.0 if self.endpoint_covered(op) else 0.0 for op in self.endpoint_ids]

    def distinct_status_count(self) -> int:
        return len({status for (_op, status) in self.hits})

    def reset(self) -> None:
        self.hits.clear()
        self.anomaly_signatures.clear()
