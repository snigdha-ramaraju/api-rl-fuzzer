"""
Structured, executable scenario objects. The LLM is never allowed to return
free-form prose that a downstream component has to re-interpret -- Ollama's
structured-output mode enforces this schema directly on generation, and the
grounding validator (grounding.py) re-checks it against the real parsed
spec before a scenario is allowed anywhere near episode seeding.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScenarioStep(BaseModel):
    operation_id: str = Field(..., description="Must match an operationId from the OpenAPI spec")
    hint: str = Field(default="", description="Short natural-language hint, e.g. 'use an expired token'")


class Scenario(BaseModel):
    sequence: list[ScenarioStep]
    hypothesis: str = Field(..., description="What business-logic or state issue this sequence might expose")
    tag: str = Field(..., description="One of: state-inconsistency | auth-boundary | ordering-effect | race-condition | other")


class ScenarioBatch(BaseModel):
    scenarios: list[Scenario]


# JSON schema handed to Ollama's `format` parameter so the model is
# constrained to emit valid structured output directly -- no prompt-and-pray
# parsing of free text anywhere in this pipeline.
SCENARIO_BATCH_JSON_SCHEMA = ScenarioBatch.model_json_schema()
