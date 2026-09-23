"""
Generates a plausible value for a field when the seed corpus has nothing
for it. This is intentionally simple and type/format-aware rather than
purely random-bytes -- the point is to produce schema-valid starting
values so early rollouts aren't wasted on trivially-rejected requests.
Mutation/mutation-engine (later phase) is responsible for pushing values
outside these plausible bounds on purpose; this module just gives a
sane default, it is not a fuzzing strategy itself.
"""

from __future__ import annotations

import random
import string
import uuid
from datetime import datetime, timedelta, timezone

from spec.parser import Field

_FORMAT_GENERATORS = {
    "uuid": lambda rng: str(uuid.UUID(int=rng.getrandbits(128))),
    "email": lambda rng: f"user{rng.randint(1, 99999)}@example.com",
    "date-time": lambda rng: (
        datetime.now(timezone.utc) - timedelta(days=rng.randint(0, 3650))
    ).isoformat(),
    "date": lambda rng: (
        datetime.now(timezone.utc) - timedelta(days=rng.randint(0, 3650))
    ).date().isoformat(),
    "uri": lambda rng: f"https://example.com/{_random_string(rng, 6)}",
}


def _random_string(rng: random.Random, length: int = 8) -> str:
    return "".join(rng.choice(string.ascii_letters) for _ in range(length))


def generate_fallback_value(field: Field, rng: random.Random | None = None) -> object:
    rng = rng or random.Random()

    if field.enum:
        return rng.choice(field.enum)

    if field.format and field.format in _FORMAT_GENERATORS:
        return _FORMAT_GENERATORS[field.format](rng)

    match field.type:
        case "string":
            return _random_string(rng, rng.randint(3, 12))
        case "integer":
            return rng.randint(0, 10_000)
        case "number":
            return round(rng.uniform(0, 10_000), 2)
        case "boolean":
            return rng.choice([True, False])
        case "array":
            return []
        case "object":
            return {}
        case _:
            return _random_string(rng)
