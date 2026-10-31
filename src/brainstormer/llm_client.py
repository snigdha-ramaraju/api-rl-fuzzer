
"""
Local-only Ollama client for structured API testing scenarios.

Generates valid, boundary and invalid scenarios through separate
LLM calls to reduce JSON truncation on resource-constrained GPUs.
"""

from __future__ import annotations

import time
import ollama

from .hardware import resolve_model
from copy import deepcopy

from .scenarios import ScenarioBatch, SCENARIO_BATCH_JSON_SCHEMA

SETUP_HELP = """
Ollama is required but not reachable.
Install Ollama from https://ollama.com/download
Start it with: ollama serve
Pull the configured model with: ollama pull {model}
"""


SYSTEM_PROMPT = """
You generate compact REST API testing scenarios.

STRICT RULES:
- Generate exactly ONE scenario.
- Use only exact operationIds and HTTP methods from the API context.
- Never invent operationIds, endpoints or parameter names.
- Use exactly ONE API step per scenario.
- Include every required parameter for that endpoint.
- Use only parameters belonging to that endpoint.
- Keep all text fields short and meaningful.
- Do not invent undocumented API constraints.
- Do not predict guaranteed HTTP status codes.

SCENARIO TYPES:

VALID:
- Use realistic values from known-good examples.
- Include every required parameter.

BOUNDARY:
- Use concrete edge-case values, not placeholders.
- Examples: empty strings, single-character strings,
  or unusually long strings.
- Do not claim undocumented limits exist.
- Explain that the test is exploratory.

INVALID:
- Use correct parameter names.
- Prefer a value whose JSON type conflicts with
  the declared OpenAPI parameter type.
- Example: use a number for a parameter declared as string.
- Keep all required parameters present.
- Explain the intentional invalid value in the hint.
- Never describe a valid example as invalid.
- Never invent undocumented constraints.

Allowed tags:
state-inconsistency, auth-boundary, ordering-effect,
race-condition, other.

Return only JSON matching the supplied schema.
No Markdown or explanations.
"""
SINGLE_SCENARIO_JSON_SCHEMA = deepcopy(
    SCENARIO_BATCH_JSON_SCHEMA
)

SINGLE_SCENARIO_JSON_SCHEMA["properties"]["scenarios"]["minItems"] = 1
SINGLE_SCENARIO_JSON_SCHEMA["properties"]["scenarios"]["maxItems"] = 1

SINGLE_SCENARIO_JSON_SCHEMA["$defs"]["Scenario"]["properties"][
    "sequence"
]["maxItems"] = 1

class LLMClient:

    def __init__(
        self,
        host: str,
        model_tier_setting: str = "auto",
        max_retries: int = 0,
        retry_delay: float = 1.0,
    ):
        self.host = host
        self.model = resolve_model(model_tier_setting)
        self.max_retries = max(0, max_retries)
        self.retry_delay = max(0.0, retry_delay)

        self.available = False
        self._client = ollama.Client(host=host, timeout=90.0)

        self._verify_ready()

    def _verify_ready(self) -> None:
        try:
            models = {
                item["model"]
                for item in self._client.list().get("models", [])
            }
        except Exception:
            print(
                f"[LLM] Ollama unavailable at {self.host}. "
                "Continuing without LLM-generated scenarios."
            )
            return

        if self.model not in models:
            print(
                f"[LLM] Model '{self.model}' is not installed. "
                "Continuing without LLM-generated scenarios."
            )
            return

        self.available = True
        print(f"[LLM] Ready: {self.model}")

    def _generate_one(
        self,
        prompt: str,
        scenario_type: str,
    ) -> ScenarioBatch:
        instruction = (
            f"Generate exactly ONE {scenario_type.upper()} scenario. "
            f"Set scenario_type to '{scenario_type}'. "
            "Use exactly ONE API step. "
            "Copy one operationId and its HTTP method exactly "
            "from the supplied API context. "
            "Include EVERY required parameter listed in the API context. "
            "NEVER use an empty parameters object when required fields exist. "
            "For valid scenarios, use realistic example values. "
            "For boundary scenarios, use concrete edge-case values. "
            "For invalid scenarios, KEEP every required parameter key "
            "and change the value of at least one field to an invalid "
            "JSON type or malformed value. "
            "Never remove required keys to create an invalid scenario. "
            "Keep hypothesis, purpose and hint concise. "
            "Return exactly one item in the scenarios array."
        )

        for attempt in range(1, self.max_retries + 2):

            try:
                response = self._client.chat(
                    model=self.model,
                    messages=[
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": (
            instruction
            + "\n\nAPI context:\n"
            + prompt
        ),
    },
],
                    format=SINGLE_SCENARIO_JSON_SCHEMA,
                    options={
                        "num_predict": 1024,
                        "temperature": 0,
                    },
                )

                content = response["message"]["content"]

                batch = ScenarioBatch.model_validate_json(content)

                if len(batch.scenarios) != 1:
                    raise ValueError(
                        "Expected exactly one scenario, received "
                        f"{len(batch.scenarios)}"
                    )

                scenario = batch.scenarios[0]

                if scenario.scenario_type.value != scenario_type:
                    raise ValueError(
                        f"Expected {scenario_type}, received "
                        f"{scenario.scenario_type.value}"
                    )

                scenario.sequence = scenario.sequence[:2]

                return batch

            except Exception as exc:
                print(
                    f"[LLM] {scenario_type} attempt "
                    f"{attempt}/{self.max_retries + 1} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                if attempt <= self.max_retries:
                    time.sleep(self.retry_delay)

        print(
            f"[LLM] Could not generate {scenario_type} scenario."
        )

        return ScenarioBatch(scenarios=[])

    def brainstorm(
        self,
        prompt: str,
        max_scenarios: int = 1,
    ) -> ScenarioBatch:

        if not self.available:
            return ScenarioBatch(scenarios=[])

        count = max(0, min(max_scenarios, 3))

        requested_types = [
            "valid",
            "boundary",
            "invalid",
        ][:count]

        generated = []

        for scenario_type in requested_types:

            print(
                f"\n[LLM] Generating {scenario_type} scenario..."
            )

            batch = self._generate_one(
                prompt=prompt,
                scenario_type=scenario_type,
            )

            generated.extend(batch.scenarios)

        print(
            f"\n[LLM] Generation completed: "
            f"{len(generated)}/{count} scenarios."
        )

        return ScenarioBatch(scenarios=generated)
