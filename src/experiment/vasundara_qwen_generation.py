
from __future__ import annotations

import json
import time
from pathlib import Path

from spec.parser import load_spec, parse_endpoints
from corpus.seed_dict import build_seed_corpus
from brainstormer.llm_client import LLMClient
from brainstormer.context import build_brainstorm_prompt
from brainstormer.grounding import validate_batch


ROOT = Path(__file__).resolve().parents[2]

SPEC_PATH = ROOT / "datasets" / "openapi" / "crapi-openapi-spec.json"

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "vasundara_qwen_generation_results.json"

def main():
    print("=" * 65)
    print("QWEN CRAPI SCENARIO GENERATION EXPERIMENT")
    print("=" * 65)

    # 1. Load OpenAPI specification.
    spec = load_spec(SPEC_PATH)
    endpoints = parse_endpoints(spec)
    print(f"\nAPI operations loaded: {len(endpoints)}")

    # 2. Build seed corpus.
    corpus = build_seed_corpus(endpoints)

    # 3. Select the actual login endpoint.
    selected_endpoint = next(
        (ep for ep in endpoints if ep.operation_id == "login"),
        None,
    )

    if selected_endpoint is None:
        raise RuntimeError("Login operation not found.")

    prompt = build_brainstorm_prompt(
        endpoints=[selected_endpoint],
        corpus=corpus,
        api_description="OWASP crAPI login endpoint only",
    )

    print(f"Focused endpoint: {selected_endpoint.operation_id}")

    # 4. Initialize Qwen.
    client = LLMClient(
        host="http://localhost:11434",
        model_tier_setting="small",
        max_retries=0,
        retry_delay=1.0,
    )

    if not client.available:
        raise RuntimeError("Ollama/Qwen is unavailable.")

    # 5. Generate scenarios.
    start = time.perf_counter()
    batch = client.brainstorm(prompt=prompt, max_scenarios=3)
    elapsed = time.perf_counter() - start

    # 6. Validate against OpenAPI.
    report = validate_batch(batch, [selected_endpoint])

    accepted = report.valid
    rejected = report.rejected

    total = len(batch.scenarios)
    accepted_count = len(accepted)
    rejected_count = len(rejected)

    rejection_rate = (
        rejected_count / total * 100 if total else 0.0
    )

    # 7. Save results.
    result = {
        "api": "OWASP crAPI",
        "scope": "login endpoint only",
        "model": client.model,
        "generation_method": "local Ollama structured JSON",
        "requested_scenario_types": ["valid", "boundary", "invalid"],
        "generation_time_seconds": round(elapsed, 3),
        "generated_count": total,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "rejection_rate_percent": round(rejection_rate, 2),
        "accepted_scenario_types": [
            item.scenario_type.value for item in accepted
        ],
        "accepted_scenarios": [
            item.model_dump(mode="json") for item in accepted
        ],
        "rejected_scenarios": [
            {
                "scenario": item["scenario"].model_dump(mode="json"),
                "reason": item["reason"],
            }
            for item in rejected
        ],
        "limitations": [
            "Only the login endpoint was tested.",
            "No live crAPI backend was used.",
            "Grounding checks specification consistency, not runtime behaviour.",
            "Scenario type coverage reflects actual generated output.",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=4),
        encoding="utf-8",
    )

    # 8. Display results.
    print("\n" + "=" * 65)
    print("EXPERIMENT RESULTS")
    print("=" * 65)
    print(f"Model: {client.model}")
    print(f"Generation time: {elapsed:.3f} seconds")
    print(f"Generated: {total}")
    print(f"Accepted: {accepted_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Rejection rate: {rejection_rate:.2f}%")

    print("\nAccepted scenario types:")
    for item in accepted:
        print(f"- {item.scenario_type.value}")

    print("\nRejection reasons:")
    for item in rejected:
        print(f"- {item['reason']}")

    print(f"\nResults saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()