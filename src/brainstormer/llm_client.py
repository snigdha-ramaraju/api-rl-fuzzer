
from __future__ import annotations

import ollama

from .hardware import resolve_model
from .scenarios import ScenarioBatch, SCENARIO_BATCH_JSON_SCHEMA

SETUP_HELP = """
Ollama is required but not reachable.
Install Ollama from https://ollama.com/download
Start it with: ollama serve
Pull the configured model with: ollama pull {model}
"""

SYSTEM_PROMPT = """
You are a REST API testing assistant.
The API context lists valid operationIds.
If at least one operationId is provided, generate exactly one scenario
with one step using an operationId copied exactly from the context.
Do not invent endpoints, parameters, or payloads.
For GET /health, generate one step calling GET /health.
Return only valid JSON matching the supplied schema.
"""


class LLMClient:
    def __init__(self, host: str, model_tier_setting: str = "auto"):
        self.host = host
        self.model = resolve_model(model_tier_setting)
        self._client = ollama.Client(host=host, timeout=90.0)
        self._verify_ready()

    def _verify_ready(self) -> None:
        try:
            models = {
                item["model"]
                for item in self._client.list().get("models", [])
            }
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}.\n"
                f"{SETUP_HELP.format(model=self.model)}"
            ) from exc

        if self.model not in models:
            raise RuntimeError(
                f"Model '{self.model}' is not pulled.\n"
                f"{SETUP_HELP.format(model=self.model)}"
            )

    def brainstorm(
        self, prompt: str, max_scenarios: int = 1
    ) -> ScenarioBatch:
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
                            f"Return at most {min(max_scenarios, 1)} scenario. "
                            "Use at most two steps.\n\n"
                            f"API context:\n{prompt}"
                        ),
                    },
                ],
                format=SCENARIO_BATCH_JSON_SCHEMA,
                options={
                    "num_predict": 512,
                    "temperature": 0,
                },
            )

            content = response["message"]["content"]
            batch = ScenarioBatch.model_validate_json(content)

            # Enforce the requested scenario and step limits locally.
            batch.scenarios = batch.scenarios[:max_scenarios]
            for scenario in batch.scenarios:
                scenario.sequence = scenario.sequence[:2]

            return batch

        except Exception as exc:
            print(
                f"[LLM] Generation failed ({type(exc).__name__}); "
                "continuing with existing seeds."
            )
            return ScenarioBatch(scenarios=[])
