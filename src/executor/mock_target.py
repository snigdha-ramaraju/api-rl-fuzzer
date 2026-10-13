"""
In-memory mock transport simulating the sample Orders API
(tests/fixtures/sample_api.yaml). This is the SAME mock used by Phase 3's
`tests/test_phase3_env.py::make_mock_transport` -- factored out here so
Phase 4's training script and tests can validate the PPO training loop
end-to-end with zero network access and zero cost, before anyone points
`train_ppo.py` at a real/live target.

Not a general-purpose fixture: it hard-codes the three Orders routes
(create/get/cancel), matching the fixture spec's paths exactly. It exists
purely for local dry-runs and tests, never for production use.
"""

from __future__ import annotations

import httpx


def make_mock_transport() -> httpx.MockTransport:
    """Simulates the sample Orders API in-memory. No network, no live server."""
    store: dict[str, str] = {}
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        if method == "POST" and path == "/orders":
            counter["n"] += 1
            oid = f"ord-{counter['n']}"
            store[oid] = "created"
            return httpx.Response(201, json={"id": oid, "status": "created"})

        if method == "GET" and path.startswith("/orders/"):
            oid = path.rsplit("/", 1)[-1]
            if oid in store:
                return httpx.Response(200, json={"id": oid, "status": store[oid]})
            return httpx.Response(404, json={"error": "not found"})

        if method == "POST" and path.startswith("/orders/"):
            oid = path.rsplit("/", 1)[-1]
            if oid in store:
                store[oid] = "cancelled"
                return httpx.Response(200, json={"id": oid, "status": "cancelled"})
            return httpx.Response(404, json={"error": "not found"})

        return httpx.Response(404, json={"error": "unmatched route"})

    def reset_state() -> None:
        store.clear()
        counter["n"] = 0

    transport = httpx.MockTransport(handler)
    transport.reset_state = reset_state
    return transport
