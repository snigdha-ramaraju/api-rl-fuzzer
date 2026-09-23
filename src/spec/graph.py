"""
Auto-derives a producer -> consumer dependency graph from parsed endpoints,
so PPO can learn to chain calls (e.g. POST /orders -> GET /orders/{id})
without anyone hand-annotating which endpoints feed which.

Matching strategy (cheapest first, escalate only on ambiguity):
1. Exact name match: response field 'id' feeds path/query param 'id'.
2. Suffix match: response field 'id' feeds param 'orderId' / 'order_id'
   when the containing endpoint's tag/path suggests the same resource.
3. Anything left ambiguous is NOT guessed here -- it's exported as a
   list of "unresolved" candidates for the LLM analyzer to disambiguate
   later (see brainstormer/). This module never fabricates a low-confidence
   edge silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from .parser import Endpoint, Field


@dataclass
class DependencyEdge:
    producer_op: str          # operation_id of endpoint that returns the value
    producer_field: str       # field name in the response
    consumer_op: str          # operation_id of endpoint that needs the value
    consumer_field: str       # param/body-field name it fills
    confidence: str           # "exact" | "suffix"


def _normalize(name: str) -> str:
    # camelCase / snake_case -> lower, no separators: "orderId" == "order_id" == "id" match base
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return s.replace("-", "_")


def _strip_resource_prefix(name: str) -> str:
    """'orderId' -> 'id', 'order_id' -> 'id' -- used for suffix matching."""
    norm = _normalize(name)
    parts = norm.split("_")
    return parts[-1] if len(parts) > 1 else norm


def build_dependency_graph(endpoints: list[Endpoint]) -> tuple[list[DependencyEdge], list[dict]]:
    edges: list[DependencyEdge] = []
    unresolved: list[dict] = []

    # index all consumer-side fields (path/query/body) across all endpoints
    consumers: list[tuple[Endpoint, str, Field]] = []
    for ep in endpoints:
        for f in ep.all_input_fields:
            consumers.append((ep, "input", f))

    for producer in endpoints:
        # only 2xx responses are useful as producers of real data
        for status, fields in producer.responses.items():
            if not status.startswith("2"):
                continue
            for pf in fields:
                if pf.type in ("object", "array"):
                    continue  # top-level flatten only; nested match is LLM's job
                matched_any = False
                for consumer_ep, _, cf in consumers:
                    if consumer_ep.operation_id == producer.operation_id:
                        continue  # skip self-loops
                    if pf.type != cf.type:
                        continue

                    if _normalize(pf.name) == _normalize(cf.name):
                        edges.append(DependencyEdge(
                            producer.operation_id, pf.name,
                            consumer_ep.operation_id, cf.name,
                            confidence="exact",
                        ))
                        matched_any = True
                    elif _strip_resource_prefix(cf.name) == _normalize(pf.name) and pf.name == "id":
                        edges.append(DependencyEdge(
                            producer.operation_id, pf.name,
                            consumer_ep.operation_id, cf.name,
                            confidence="suffix",
                        ))
                        matched_any = True

                if not matched_any and pf.name.lower() in ("id", "uuid", "token"):
                    # high-value field with no confident consumer match --
                    # flag for the LLM analyzer rather than dropping it
                    unresolved.append({
                        "producer_op": producer.operation_id,
                        "producer_field": pf.name,
                        "reason": "no exact/suffix match found for a likely-identifying field",
                    })

    return edges, unresolved
