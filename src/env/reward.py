"""
Reward = coverage novelty + oracle-anomaly bonus - invalid-request/transport
penalty. Deliberately does NOT consult any hand-written business rules or
expected-response tables -- "anomalous" is defined purely by spec/oracle.py
(structural deviation from the spec's own declared response schemas), per
the project's original design decision to let RL discover unknown issues
rather than cap it at a pre-written checklist.

The anomaly bonus is NOVELTY-GATED per episode (`is_new_anomaly`, sourced
from CoverageTracker.record_anomaly): it is only paid out the first time a
given anomaly signature (operation, status, and *how* it deviated from the
spec) is observed in the current episode. Without this gate, an agent could
farm unlimited reward by repeating the exact same anomalous call forever
(e.g. calling an endpoint cold, forever, to keep collecting the same
undeclared-404 bonus) -- a pure reward-hacking exploit that has nothing to
do with discovering new bugs. Plain coverage novelty (`is_new_coverage`)
already had this same protection from Phase 3 onward; this closes the gap
that the anomaly bonus, which is bigger, had been missing it.
"""

from __future__ import annotations

from spec.oracle import OracleResult

NEW_COVERAGE_BONUS = 1.0
ANOMALY_BONUS = 2.0
SCHEMA_INVALID_REQUEST_PENALTY = -0.5
TRANSPORT_ERROR_PENALTY = -0.2


def compute_reward(is_new_coverage: bool, oracle_result: OracleResult | None,
                    request_was_schema_valid: bool, transport_error: bool,
                    is_new_anomaly: bool = True) -> float:
    """`is_new_anomaly` gates the anomaly bonus and must reflect PER-EPISODE
    novelty of this exact anomaly signature (see
    CoverageTracker.record_anomaly), not just "is this response anomalous".
    It defaults to True only for convenience/backward-compatibility when
    `oracle_result` is None or not anomalous, in which case the flag has no
    effect anyway -- callers that DO have an anomalous oracle_result must
    pass the tracker's real novelty verdict, or this gate does nothing."""
    if transport_error:
        return TRANSPORT_ERROR_PENALTY

    reward = 0.0
    if not request_was_schema_valid:
        reward += SCHEMA_INVALID_REQUEST_PENALTY
    if is_new_coverage:
        reward += NEW_COVERAGE_BONUS
    if oracle_result is not None and oracle_result.is_anomalous and is_new_anomaly:
        reward += ANOMALY_BONUS

    return reward
