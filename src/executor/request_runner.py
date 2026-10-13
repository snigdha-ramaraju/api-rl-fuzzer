"""
Executes real HTTP requests against the target API for a resolved call.
Enforces the rate limit and total-request budget from config/target.yaml
so nothing can run away against any target, paid or not -- this is the
one place in Phase 3 where the cost-safety cap actually lives.

Uses httpx.Client, which accepts an injectable `transport`. Tests use
httpx.MockTransport, so this module (and the env built on top of it) is
fully testable with zero network access and no live target server needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from spec.parser import Endpoint


@dataclass
class ExecutionResult:
    status_code: int
    body: dict | None
    error: str | None = None


class BudgetExceededError(Exception):
    pass


class RequestRunner:
    def __init__(self, base_url: str, rate_limit_per_sec: float = 5,
                 max_requests_per_run: int = 5000,
                 transport: httpx.BaseTransport | None = None,
                 auth_header: dict | None = None):
        self.base_url = base_url.rstrip("/")
        # rate_limit_per_sec <= 0 disables throttling entirely
        self.min_interval = (1.0 / rate_limit_per_sec) if rate_limit_per_sec and rate_limit_per_sec > 0 else 0.0
        self.max_requests = max_requests_per_run
        self.request_count = 0
        self._last_call_ts = 0.0
        self._transport = transport
        self._client = httpx.Client(base_url=self.base_url, transport=transport, headers=auth_header or {})

    def close(self) -> None:
        self._client.close()

    def reset_state(self) -> None:
        transport = self._transport if self._transport is not None else getattr(self._client, "_transport", None)
        if transport is not None and hasattr(transport, "reset_state"):
            try:
                transport.reset_state(reset_base=True)
            except TypeError:
                transport.reset_state()
        base = getattr(transport, "base_transport", None)
        if base is not None and hasattr(base, "reset_state"):
            base.reset_state()

    def execute(self, endpoint: Endpoint, path_values: dict, query_values: dict,
                header_values: dict, body_values: dict) -> ExecutionResult:
        if self.request_count >= self.max_requests:
            raise BudgetExceededError(
                f"Request budget of {self.max_requests} exhausted for this run."
            )

        self._throttle()

        path = endpoint.path
        for name, value in path_values.items():
            path = path.replace("{" + name + "}", str(value))

        try:
            resp = self._client.request(
                endpoint.method, path,
                params={k: v for k, v in query_values.items() if v is not None} or None,
                headers={k: str(v) for k, v in header_values.items() if v is not None} or None,
                json=body_values or None,
            )
            self.request_count += 1
            try:
                body = resp.json()
            except Exception:
                body = None
            return ExecutionResult(status_code=resp.status_code, body=body)
        except httpx.HTTPError as e:
            self.request_count += 1
            return ExecutionResult(status_code=0, body=None, error=str(e))

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call_ts = time.monotonic()
