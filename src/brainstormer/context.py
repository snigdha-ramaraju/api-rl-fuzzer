
"""
Builds the structured API context supplied to the local LLM brainstormer.

Includes endpoint metadata, required fields, input constraints,
known-good payloads, unresolved dependencies and coverage information.

The prompt explicitly guides the LLM to generate valid, boundary and
invalid scenarios without inventing API parameter names or constraints.
"""

from __future__ import annotations

import json

from spec.parser import Endpoint
from corpus.seed_dict import SeedCorpus


def build_brainstorm_prompt(
    endpoints: list[Endpoint],
    corpus: SeedCorpus,
    coverage_summary: dict | None = None,
    unresolved_fields: list[dict] | None = None,
    api_description: str = "",
) -> str:

    lines: list[str] = []

    if api_description:
        lines.append(f"API description: {api_description}")

    lines.extend([
        "\nROLE:",
        "You are an expert REST API security testing engineer.",
        "Generate useful, specification-grounded API testing scenarios.",
        "\nAVAILABLE ENDPOINTS:",
    ])

    for endpoint in endpoints:
        lines.append(f"\nOperation ID: {endpoint.operation_id}")
        lines.append(f"HTTP Method: {endpoint.method}")
        lines.append(f"Path: {endpoint.path}")

        if endpoint.description:
            lines.append(f"Description: {endpoint.description}")

        field_groups = [
            ("PATH PARAMETERS", endpoint.path_params),
            ("QUERY PARAMETERS", endpoint.query_params),
            ("HEADER PARAMETERS", endpoint.header_params),
            ("REQUEST BODY FIELDS", endpoint.request_body),
        ]

        for group_name, fields in field_groups:
            if not fields:
                continue

            lines.append(f"\n{group_name}:")

            for field in fields:
                details = [
                    f"name={field.name}",
                    f"type={field.type}",
                    f"required={field.required}",
                ]

                if field.enum:
                    details.append(f"enum={field.enum}")

                if field.format:
                    details.append(f"format={field.format}")

                if field.example is not None:
                    details.append(f"example={field.example}")

                lines.append("- " + ", ".join(details))

        required_fields = [
            field.name
            for field in endpoint.all_input_fields
            if field.required
        ]

        lines.append(
            "REQUIRED FIELDS: "
            + (
                ", ".join(required_fields)
                if required_fields
                else "None"
            )
        )

        payload = corpus.sample_full_payload(endpoint.operation_id)

        if payload:
            lines.append(
                "Known-good example payload: "
                + json.dumps(payload, default=str)
            )

        # Repeat the exact operation IDs in a compact authoritative allowlist.
    operation_ids = [endpoint.operation_id for endpoint in endpoints]

    lines.extend([
        "\nAUTHORITATIVE OPERATION ID ALLOWLIST:",
        "The following strings are the ONLY legal operation_id values:",
        *[f"- {operation_id}" for operation_id in operation_ids],
        "Copy operation_id values EXACTLY, character by character.",
        "Never create operation IDs from endpoint descriptions or paths.",
        "For example, if the supplied ID is 'login', do NOT write "
        "'user-login' or 'user_login'.",
        "If you cannot identify a suitable endpoint, do not invent one.",
        "\nSCENARIO GENERATION RULES:",
        "\nVALID:",
        "- Use correctly named fields and plausible valid values.",
        "- Include all required fields.",
        "- Prefer realistic multi-step API workflows.",
        "- Reuse matching credentials or identifiers across dependent steps.",
        "\nBOUNDARY:",
        "- Explore edge-case values using the correct parameter names.",
        "- Examples include empty strings, unusually long strings, "
        "and values near plausible input boundaries.",
        "- When the specification defines no explicit boundary, "
        "describe the case as exploratory boundary testing.",
        "- Do not claim an undocumented limit exists.",
        "\nINVALID:",
        "- Use deliberately malformed, empty or wrong-type values "
        "to investigate input validation.",
        "- Preserve the correct parameter names and required keys.",
        "- Explain which input is intentionally invalid and why.",
        "\nSCENARIO TAG:",
        "Use exactly ONE tag from this list:",
        "- state-inconsistency",
        "- auth-boundary",
        "- ordering-effect",
        "- race-condition",
        "- other",
        "\nOUTPUT QUALITY:",
        "- Each scenario must have a meaningful hypothesis and purpose.",
        "- Every sequence step must reference a real supplied endpoint.",
        "- Each sequence must contain one or two operations.",
        "- Avoid duplicate scenarios that test the same behaviour.",
        "- Prefer scenarios that explore useful endpoint interactions.",
        "\nIMPORTANT:",
        "HTTP methods, operation IDs and parameter names must exactly "
        "match the supplied OpenAPI specification.",
        "Required fields must never be silently omitted.",
    ])

    if unresolved_fields:
        lines.append("\nUNRESOLVED DEPENDENCIES:")

        for item in unresolved_fields:
            lines.append(
                f"- {item['producer_op']}."
                f"{item['producer_field']}: {item['reason']}"
            )

    if coverage_summary:
        lines.append("\nCURRENT COVERAGE:")

        lines.append(
            f"Endpoints covered: "
            f"{coverage_summary.get('covered_endpoints', 0)}/"
            f"{coverage_summary.get('total_endpoints', 0)}"
        )

        lines.append(
            "Distinct endpoint/status combinations: "
            f"{coverage_summary.get('distinct_status_count', 0)}"
        )

        lines.append(
            "Prioritize useful scenarios exploring new endpoint "
            "interactions and under-tested behaviour."
        )

    return "\n".join(lines)
