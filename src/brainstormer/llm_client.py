"""
Local-only LLM client. This is the ONLY place in the codebase that talks
to an LLM, and it only ever talks to a locally-running Ollama instance --
there is no paid-API code path anywhere in this project.

If Ollama isn't installed/running, this fails loudly with setup
instructions rather than silently degrading to a cloud fallback, per the
project constraint: the brainstormer is mandatory, but must always be
Rs0-cost and local.
"""

from __future__ import annotations

import ollama

from .hardware import resolve_model
from .scenarios import ScenarioBatch, SCENARIO_BATCH_JSON_SCHEMA

SETUP_HELP = """
Ollama is required but not reachable.

Setup (all free, local, one-time):
  1. Install Ollama: https://ollama.com/download
  2. Start it:        ollama serve
  3. Pull a model:     ollama pull {model}

No account, no API key, no cost.
"""

SYSTEM_PROMPT = (
    "You are a security/QA test-design assistant. Given an API's structure, "
    "propose multi-step, stateful test scenarios that a simple mutation "
    "fuzzer would likely miss -- ordering effects, auth-boundary crossings, "
    "race conditions, state inconsistencies. Only reference operationIds "
    "that actually appear in the provided context. Propose at most "
    "{max_scenarios} scenarios."
)


class LLMClient:
    def __init__(self, host: str, model_tier_setting: str = "auto"):
        self.host = host
        self.model = resolve_model(model_tier_setting)
        self._client = ollama.Client(host=host)
        self._verify_ready()

    def _verify_ready(self) -> None:
        try:
            models = {m["model"] for m in self._client.list().get("models", [])}
        except Exception as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}.\n{SETUP_HELP.format(model=self.model)}"
            ) from e

        if self.model not in models:
            raise RuntimeError(
                f"Model '{self.model}' not pulled yet.\n{SETUP_HELP.format(model=self.model)}"
            )

    def brainstorm(self, prompt: str, max_scenarios: int = 8) -> ScenarioBatch:
        """
        `prompt` should already contain the API context (spec summary,
        sample inputs, coverage-so-far) -- see brainstormer/context.py.
        Returns a validated ScenarioBatch; raw model output never leaks
        past this method unparsed.
        """
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(max_scenarios=max_scenarios)},
                {"role": "user", "content": prompt},
            ],
            format=SCENARIO_BATCH_JSON_SCHEMA,
        )
        return ScenarioBatch.model_validate_json(response["message"]["content"])
