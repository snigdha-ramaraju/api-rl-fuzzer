"""
Parses an OpenAPI 3.x spec into a flat, RL-friendly representation:
a list of Endpoint objects with resolved parameters, request body schema,
and response schemas per status code.

This is the single source of truth the rest of the system builds on:
- action space (env/action_space.py) enumerates Endpoint.operation_id
- dependency graph (spec/graph.py) matches Endpoint.produces / .consumes
- structural oracle (spec/oracle.py) validates against Endpoint.responses
"""

from __future__ import annotations

import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from openapi_spec_validator import validate


@dataclass
class Field:
    name: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    required: bool = False
    enum: list[Any] | None = None
    format: str | None = None  # e.g. "date-time", "uuid", "email"
    example: Any = None


@dataclass
class Endpoint:
    operation_id: str
    method: str  # GET, POST, PUT, DELETE, PATCH
    path: str
    path_params: list[Field] = field(default_factory=list)
    query_params: list[Field] = field(default_factory=list)
    header_params: list[Field] = field(default_factory=list)
    request_body: list[Field] = field(default_factory=list)
    responses: dict[str, list[Field]] = field(default_factory=dict)  # status_code -> fields
    tags: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def all_input_fields(self) -> list[Field]:
        return self.path_params + self.query_params + self.header_params + self.request_body


class SpecParseError(Exception):
    pass


def load_spec(spec_path: str | Path) -> dict:
    spec_path = Path(spec_path)
    raw = spec_path.read_text()
    if spec_path.suffix in (".yaml", ".yml"):
        spec = yaml.safe_load(raw)
    else:
        spec = json.loads(raw)

    try:
        validate(spec)
    except Exception as e:
        raise SpecParseError(f"Invalid OpenAPI spec: {e}") from e

    return spec


def _resolve_ref(spec: dict, ref: str) -> dict:
    """Resolve a local $ref like '#/components/schemas/Order'."""
    parts = ref.lstrip("#/").split("/")
    node = spec
    for p in parts:
        node = node[p]
    return node


def _schema_to_fields(spec: dict, schema: dict, path: str = "") -> list[Field]:
    """Flatten a JSON schema object into a list of Field objects (one level;
    nested objects are kept as type='object' rather than recursively flattened,
    since RL actions operate on top-level params/body fields)."""
    if "$ref" in schema:
        schema = _resolve_ref(spec, schema["$ref"])

    fields: list[Field] = []
    if schema.get("type") == "object" or "properties" in schema:
        required = set(schema.get("required", []))
        for name, prop in schema.get("properties", {}).items():
            if "$ref" in prop:
                prop = _resolve_ref(spec, prop["$ref"])
            fields.append(Field(
                name=name,
                type=prop.get("type", "object"),
                required=name in required,
                enum=prop.get("enum"),
                format=prop.get("format"),
                example=prop.get("example"),
            ))
    return fields


def _params_by_location(spec: dict, params: list[dict], location: str) -> list[Field]:
    out = []
    for p in params:
        if "$ref" in p:
            p = _resolve_ref(spec, p["$ref"])
        if p.get("in") != location:
            continue
        schema = p.get("schema", {})
        out.append(Field(
            name=p["name"],
            type=schema.get("type", "string"),
            required=p.get("required", False),
            enum=schema.get("enum"),
            format=schema.get("format"),
            example=schema.get("example") or p.get("example"),
        ))
    return out


def parse_endpoints(spec: dict) -> list[Endpoint]:
    """Main entry point: OpenAPI dict -> list[Endpoint]."""
    endpoints: list[Endpoint] = []

    for path, path_item in spec.get("paths", {}).items():
        # parameters defined at the path-item level apply to all operations
        shared_params = path_item.get("parameters", [])

        for method in ("get", "post", "put", "patch", "delete"):
            if method not in path_item:
                continue
            op = path_item[method]
            all_params = shared_params + op.get("parameters", [])

            request_body_fields: list[Field] = []
            rb = op.get("requestBody", {})
            content = rb.get("content", {})
            json_body = content.get("application/json", {})
            if "schema" in json_body:
                request_body_fields = _schema_to_fields(spec, json_body["schema"])

            responses: dict[str, list[Field]] = {}
            for status_code, resp in op.get("responses", {}).items():
                resp_content = resp.get("content", {}).get("application/json", {})
                if "schema" in resp_content:
                    responses[status_code] = _schema_to_fields(spec, resp_content["schema"])
                else:
                    responses[status_code] = []

            operation_id = op.get("operationId") or f"{method}_{path}".replace("/", "_")

            endpoints.append(Endpoint(
                operation_id=operation_id,
                method=method.upper(),
                path=path,
                path_params=_params_by_location(spec, all_params, "path"),
                query_params=_params_by_location(spec, all_params, "query"),
                header_params=_params_by_location(spec, all_params, "header"),
                request_body=request_body_fields,
                responses=responses,
                tags=op.get("tags", []),
                description=op.get("description", "") or op.get("summary", ""),
            ))

    return endpoints
