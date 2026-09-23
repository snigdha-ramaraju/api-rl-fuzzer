"""
Single entry point later phases call to get a value for a field: prefer a
seeded value most of the time, but not always -- pure seed-only sampling
would just replay known-good requests forever and defeat the purpose of
using RL at all. `seed_probability` is intentionally the caller's decision
(e.g. the mutation engine or PPO env may want a different ratio, or may
anneal it over training), so this module takes it as a parameter rather
than hard-coding a policy here.
"""

from __future__ import annotations

import random

from spec.parser import Field
from .seed_dict import SeedCorpus
from .fallback_generator import generate_fallback_value


def resolve_value(operation_id: str, field: Field, corpus: SeedCorpus,
                   rng: random.Random | None = None,
                   seed_probability: float = 0.7) -> object:
    rng = rng or random.Random()

    seeded = corpus.sample_value(operation_id, field.name, rng=rng)
    if seeded is not None and rng.random() < seed_probability:
        return seeded

    return generate_fallback_value(field, rng=rng)
