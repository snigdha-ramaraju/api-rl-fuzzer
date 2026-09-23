"""
Lightweight structural oracle, derived automatically from the OpenAPI spec.
No hand-written business rules or expected-response tables live here --
just: is the status code one the spec declared, and does the JSON shape
match the declared schema. Deviations are signal for the reward function
and for the LLM anomaly triage step, not hard pass/fail rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from .parser import Endpoint


@dataclass
class OracleResult:
    status_known: bool          # was this status code declared in the spec?
    shape_valid: bool | None    # None if we couldn't check (e.g. no body)
    missing_fields: list[str]
    unexpected_type_fields: list[str]

    @property
    def is_anomalous(self) -> bool:
        return (not self.status_known) or (self.shape_valid is False)


def check_response(endpoint: Endpoint, status_code: int, body: dict | None) -> OracleResult:
    status_str = str(status_code)
    status_known = status_str in endpoint.responses or _matches_range(status_str, endpoint.responses)

    if not status_known or body is None:
        return OracleResult(status_known=status_known, shape_valid=None,
                             missing_fields=[], unexpected_type_fields=[])

    expected_fields = endpoint.responses.get(status_str) or endpoint.responses.get(status_str[0] + "XX", [])
    if not expected_fields:
        return OracleResult(status_known=status_known, shape_valid=None,
                             missing_fields=[], unexpected_type_fields=[])

    missing = [f.name for f in expected_fields if f.required and f.name not in body]
    type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
    bad_types = []
    for f in expected_fields:
        if f.name in body and f.type in type_map:
            if not isinstance(body[f.name], type_map[f.type]):
                bad_types.append(f.name)

    shape_valid = not missing and not bad_types
    return OracleResult(status_known=status_known, shape_valid=shape_valid,
                         missing_fields=missing, unexpected_type_fields=bad_types)


def _matches_range(status_str: str, responses: dict) -> bool:
    # OpenAPI allows "2XX"-style wildcard response keys
    wildcard = status_str[0] + "XX"
    return wildcard in responses
