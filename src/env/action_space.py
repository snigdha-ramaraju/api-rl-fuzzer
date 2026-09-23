"""
Action space: one discrete action per endpoint operation. Field VALUES are
not separate RL actions in this phase -- they're resolved automatically:
1. From the dependency graph, when a prior call earlier in THIS episode
   already produced a value for that field (e.g. createOrder's 'id' feeds
   getOrder's 'orderId'). This is what lets a chained sequence like
   create -> read -> cancel actually work end to end.
2. Otherwise from the Phase 2 seed corpus / fallback generator.

This keeps the action space small and lets PPO focus on *sequencing*
calls -- exactly the state-dependent behaviour a plain mutation engine
can't learn. Per-field mutation as its own action is a natural extension
for a later phase, not required here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from spec.parser import Endpoint, Field
from spec.graph import DependencyEdge
from corpus.seed_dict import SeedCorpus
from corpus.resolver import resolve_value


@dataclass
class ResolvedRequest:
    path_values: dict
    query_values: dict
    header_values: dict
    body_values: dict
    schema_valid: bool  # False if any required field resolved to None


def build_action_space(endpoints: list[Endpoint]) -> list[Endpoint]:
    """Fixed ordering -- action index i always refers to endpoints[i]."""
    return list(endpoints)


def build_dependency_index(edges: list[DependencyEdge]) -> dict[tuple[str, str], list[DependencyEdge]]:
    index: dict[tuple[str, str], list[DependencyEdge]] = {}
    for edge in edges:
        index.setdefault((edge.consumer_op, edge.consumer_field), []).append(edge)
    return index


def resolve_request(endpoint: Endpoint, corpus: SeedCorpus,
                     known_values: dict[tuple[str, str], object],
                     dependency_index: dict[tuple[str, str], list[DependencyEdge]],
                     rng: random.Random, seed_probability: float = 0.7) -> ResolvedRequest:

    def resolve_field(f: Field) -> object:
        for edge in dependency_index.get((endpoint.operation_id, f.name), []):
            key = (edge.producer_op, edge.producer_field)
            if key in known_values:
                return known_values[key]
        return resolve_value(endpoint.operation_id, f, corpus, rng=rng, seed_probability=seed_probability)

    path_values = {f.name: resolve_field(f) for f in endpoint.path_params}
    query_values = {f.name: resolve_field(f) for f in endpoint.query_params}
    header_values = {f.name: resolve_field(f) for f in endpoint.header_params}
    body_values = {f.name: resolve_field(f) for f in endpoint.request_body}

    all_values = {**path_values, **query_values, **header_values, **body_values}
    required_fields = [f for f in endpoint.all_input_fields if f.required]
    schema_valid = all(all_values.get(f.name) is not None for f in required_fields)

    return ResolvedRequest(path_values, query_values, header_values, body_values, schema_valid)
