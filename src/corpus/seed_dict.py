"""
Builds a seed corpus from two sources:

1. The OpenAPI spec itself -- every declared `example` and every `enum`
   option is a free, always-in-sync seed. No maintenance burden: it comes
   straight from spec/parser.py's already-parsed Field objects.
2. An optional user-supplied sample-inputs file -- real recorded requests
   (operation_id + the params/body that were sent). These are the highest
   quality seeds since they're known-valid, real-world combinations, but
   they are a WARM START, not a constraint: the RL agent is free to (and
   expected to) mutate past them. See fallback_generator.py for what fills
   in fields the corpus has nothing for.

Two views are exposed:
- field_values: dict[(operation_id, field_name)] -> list of candidate values,
  for independent per-field sampling (used when mutating one field at a time).
- full_examples: dict[operation_id] -> list of complete request payloads,
  for seeding a whole call at once with a known-good combination.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

from spec.parser import Endpoint


@dataclass
class SampleInput:
    operation_id: str
    payload: dict[str, Any]  # merged path/query/header/body values as sent


def load_sample_inputs(path: str | Path) -> list[SampleInput]:
    """
    Expects a JSON file shaped like:
    [
      {"operation_id": "createOrder", "payload": {"item_id": "sku-123", "quantity": 2}},
      {"operation_id": "getOrder", "payload": {"orderId": "ord-9"}}
    ]
    """
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("sample inputs file must contain a JSON array")

    samples = []
    for entry in raw:
        if "operation_id" not in entry or "payload" not in entry:
            raise ValueError(f"malformed sample input entry (needs operation_id + payload): {entry}")
        samples.append(SampleInput(operation_id=entry["operation_id"], payload=entry["payload"]))
    return samples


@dataclass
class SeedCorpus:
    field_values: dict[tuple[str, str], list[Any]] = dc_field(default_factory=dict)
    full_examples: dict[str, list[dict]] = dc_field(default_factory=dict)

    def values_for(self, operation_id: str, field_name: str) -> list[Any]:
        return self.field_values.get((operation_id, field_name), [])

    def sample_value(self, operation_id: str, field_name: str,
                      rng: random.Random | None = None) -> Any | None:
        """Returns a seeded value, or None if the corpus has nothing for this
        field -- caller (e.g. fallback_generator) is expected to handle None."""
        rng = rng or random.Random()
        values = self.values_for(operation_id, field_name)
        return rng.choice(values) if values else None

    def sample_full_payload(self, operation_id: str,
                             rng: random.Random | None = None) -> dict | None:
        rng = rng or random.Random()
        examples = self.full_examples.get(operation_id, [])
        return dict(rng.choice(examples)) if examples else None

    def _add_value(self, operation_id: str, field_name: str, value: Any) -> None:
        if value is None:
            return
        key = (operation_id, field_name)
        bucket = self.field_values.setdefault(key, [])
        if value not in bucket:
            bucket.append(value)


def build_seed_corpus(endpoints: list[Endpoint],
                       sample_inputs: list[SampleInput] | None = None) -> SeedCorpus:
    corpus = SeedCorpus()

    # Source 1: spec-declared examples and enums -- free, always in sync.
    for ep in endpoints:
        for f in ep.all_input_fields:
            if f.example is not None:
                corpus._add_value(ep.operation_id, f.name, f.example)
            if f.enum:
                for opt in f.enum:
                    corpus._add_value(ep.operation_id, f.name, opt)

    # Source 2: recorded sample inputs -- highest-quality seeds, and the
    # only source that populates full_examples (a whole known-good call).
    for sample in sample_inputs or []:
        corpus.full_examples.setdefault(sample.operation_id, []).append(dict(sample.payload))
        for field_name, value in sample.payload.items():
            corpus._add_value(sample.operation_id, field_name, value)

    return corpus
