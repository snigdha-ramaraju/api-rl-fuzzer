
"""
Every LLM-generated scenario must be checked against the actual
parsed OpenAPI specification before it can be used.

Checks:
- Operation IDs
- HTTP methods
- Unknown parameters
- Missing required parameters

Invalid scenarios are rejected, never silently repaired.
"""

from __future__ import annotations

from dataclasses import dataclass

from spec.parser import Endpoint
from .scenarios import Scenario, ScenarioBatch


@dataclass
class GroundingReport:
    valid: list[Scenario]
    rejected: list[dict]


def validate_batch(
    batch: ScenarioBatch,
    endpoints: list[Endpoint],
) -> GroundingReport:

    known_endpoints = {
        endpoint.operation_id: endpoint
        for endpoint in endpoints
    }

    valid: list[Scenario] = []
    rejected: list[dict] = []

    for scenario in batch.scenarios:

        rejection_reason = None

        if not scenario.sequence:
            rejection_reason = "empty sequence"

        else:
            for step in scenario.sequence:

                endpoint = known_endpoints.get(step.operation_id)

                if endpoint is None:
                    rejection_reason = (
                        f"unknown operationId: {step.operation_id}"
                    )
                    break

                expected_method = endpoint.method.upper()

                if step.http_method.upper() != expected_method:
                    rejection_reason = (
                        f"HTTP method mismatch for {step.operation_id}: "
                        f"expected {expected_method}, got {step.http_method}"
                    )
                    break

                allowed_parameters = {
                    field.name
                    for field in endpoint.all_input_fields
                }

                required_parameters = {
                    field.name
                    for field in endpoint.all_input_fields
                    if field.required
                }

                supplied_parameters = set(step.parameters.keys())

                unknown_parameters = (
                    supplied_parameters - allowed_parameters
                )

                missing_parameters = (
                    required_parameters - supplied_parameters
                )

                if unknown_parameters:
                    rejection_reason = (
                        f"unknown parameter(s) for {step.operation_id}: "
                        f"{sorted(unknown_parameters)}"
                    )
                    break

                if missing_parameters:
                    rejection_reason = (
                        f"missing required parameter(s) for "
                        f"{step.operation_id}: "
                        f"{sorted(missing_parameters)}"
                    )
                    break

        if rejection_reason:
            rejected.append({
                "scenario": scenario,
                "reason": rejection_reason,
            })
        else:
            valid.append(scenario)

    return GroundingReport(
        valid=valid,
        rejected=rejected,
    )
