"""
Every scenario the LLM proposes must be checked against the actual parsed
OpenAPI spec before it's usable. Local open-source models at 3b-14b will
hallucinate endpoint names occasionally -- this is a hard filter, not a
warning: invalid scenarios are dropped, never repaired by guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from spec.parser import Endpoint
from .scenarios import Scenario, ScenarioBatch


@dataclass
class GroundingReport:
    valid: list[Scenario]
    rejected: list[dict]  # {"scenario": Scenario, "reason": str}


def validate_batch(batch: ScenarioBatch, endpoints: list[Endpoint]) -> GroundingReport:
    known_ops = {e.operation_id for e in endpoints}
    valid: list[Scenario] = []
    rejected: list[dict] = []

    for scenario in batch.scenarios:
        unknown_ops = [s.operation_id for s in scenario.sequence if s.operation_id not in known_ops]
        if unknown_ops:
            rejected.append({
                "scenario": scenario,
                "reason": f"references unknown operationId(s): {unknown_ops}",
            })
            continue
        if len(scenario.sequence) < 1:
            rejected.append({"scenario": scenario, "reason": "empty sequence"})
            continue
        valid.append(scenario)

    return GroundingReport(valid=valid, rejected=rejected)
