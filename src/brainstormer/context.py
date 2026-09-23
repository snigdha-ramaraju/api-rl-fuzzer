"""
Builds the prompt handed to the LLM brainstormer: endpoint structure from
the spec, a couple of known-good example payloads from the Phase 2 seed
corpus (for realism), any fields the Phase 1 dependency graph couldn't
confidently link (candidates for the LLM to reason about), and a coverage
summary once training has run for a while. This is the only place free-form
text is assembled -- everything downstream of the LLM call is structured
(scenarios.py) and re-validated (grounding.py).
"""

from __future__ import annotations

from spec.parser import Endpoint
from corpus.seed_dict import SeedCorpus


def build_brainstorm_prompt(endpoints: list[Endpoint], corpus: SeedCorpus,
                             coverage_summary: dict | None = None,
                             unresolved_fields: list[dict] | None = None,
                             api_description: str = "") -> str:
    lines: list[str] = []

    if api_description:
        lines.append(f"API description: {api_description}")

    lines.append("Endpoints:")
    for e in endpoints:
        fields = ", ".join(f"{f.name}:{f.type}" for f in e.all_input_fields) or "none"
        desc = f" -- {e.description}" if e.description else ""
        lines.append(f"- {e.operation_id} [{e.method} {e.path}] fields: {fields}{desc}")

    if unresolved_fields:
        lines.append(
            "\nFields the dependency graph could NOT confidently auto-link "
            "(consider scenarios probing these):"
        )
        for u in unresolved_fields:
            lines.append(f"- {u['producer_op']}.{u['producer_field']}: {u['reason']}")

    examples = []
    for e in endpoints:
        payload = corpus.sample_full_payload(e.operation_id)
        if payload:
            examples.append(f"- {e.operation_id} known-good payload: {payload}")
    if examples:
        lines.append("\nKnown-good example payloads:")
        lines.extend(examples)

    if coverage_summary:
        lines.append(
            f"\nCoverage so far: {coverage_summary.get('covered_endpoints', 0)}/"
            f"{coverage_summary.get('total_endpoints', 0)} endpoints touched, "
            f"{coverage_summary.get('distinct_status_count', 0)} distinct "
            "(endpoint, status) combinations observed. Favor scenarios that "
            "probe interactions between endpoints rather than repeating "
            "single-endpoint coverage."
        )

    return "\n".join(lines)
