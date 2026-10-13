"""
Phase 6: Planted-bug mock target.

Wraps the base in-memory Orders API (executor/mock_target.py) and injects
three intentional bugs so anomaly DISCOVERY can be measured against known
ground truth. Every bug is reachable from the real action space: each
trigger is expressed in terms of request shapes that ApiFuzzEnv actually
produces against tests/fixtures/sample_api.yaml (path ids are "ord-<n>"
strings from createOrder or random letter strings; createOrder bodies
always contain both item_id and quantity; quantity is an int in
0..10000 when no seed is supplied).

All three return a status the spec does NOT declare for that operation, so
the structural oracle (spec/oracle.py) flags them as anomalies:

  bug_getorder_malformed_id   GET  /orders/{id}  id is not "ord-<digits>"  -> 400
      Single call (e.g. an uncorrelated getOrder with no prior create).
  bug_cancelorder_double      POST /orders/{id}  second successful cancel   -> 409
      Needs a sequence: createOrder -> cancelOrder -> cancelOrder, with the
      id piped through the dependency graph.
  bug_createorder_big_qty     POST /orders       quantity > 9000            -> 422
      Value-dependent: ~10% of unseeded createOrder calls.

State (which orders were already cancelled) lives on the instance, so each
environment / experiment gets a fresh, independent target.
"""

from __future__ import annotations

import json
import re

import httpx

_ORDER_ID_RE = re.compile(r"^ord-\d+$")
_ORDER_PATH_RE = re.compile(r"^/orders/([^/]+)$")

BIG_QUANTITY_THRESHOLD = 9000

# Ground truth used to classify an observed anomaly as a planted bug.
# (operation_id, status_code) is unique per bug; the base mock never
# returns these statuses on its own (it only returns 200/201/404).
_BUG_DEFINITIONS = {
    "bug_getorder_malformed_id": {
        "name": "getOrder_malformed_id",
        "operation_id": "getOrder",
        "status": 400,
        "description": "GET /orders/<malformed-id> returns undeclared 400 Bad Request",
    },
    "bug_cancelorder_double": {
        "name": "cancelOrder_already_canceled",
        "operation_id": "cancelOrder",
        "status": 409,
        "description": "Cancelling an already-cancelled order returns undeclared 409 Conflict",
    },
    "bug_createorder_big_qty": {
        "name": "createOrder_quantity_too_large",
        "operation_id": "createOrder",
        "status": 422,
        "description": (
            f"POST /orders with quantity > {BIG_QUANTITY_THRESHOLD} returns "
            "undeclared 422 Unprocessable Entity"
        ),
    },
}


def classify_planted_bug(operation_id: str, status_code: int | None) -> str | None:
    """Returns the planted-bug key an observed (operation, status) belongs to,
    or None if it is not a planted bug (i.e. a false positive / other anomaly)."""
    for key, bug in _BUG_DEFINITIONS.items():
        if bug["operation_id"] == operation_id and bug["status"] == status_code:
            return key
    return None


class PlantedBugsMockTarget(httpx.BaseTransport):
    """HTTPX-compatible transport that wraps a base transport and injects
    the planted bugs above."""

    def __init__(self, base_transport):
        self.base_transport = base_transport
        self.bug_definitions = {k: dict(v) for k, v in _BUG_DEFINITIONS.items()}
        self._cancelled: set[str] = set()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        method = request.method.upper()
        path = request.url.path

        # Bug 1: GET /orders/<malformed id>
        if method == "GET":
            m = _ORDER_PATH_RE.match(path)
            if m and not _ORDER_ID_RE.match(m.group(1)):
                return self._make_response(request, 400, {"error": "Invalid order ID"})

        # Bug 3: POST /orders with an oversized quantity
        if method == "POST" and path == "/orders":
            quantity = self._json_body(request).get("quantity")
            if isinstance(quantity, int) and not isinstance(quantity, bool) \
                    and quantity > BIG_QUANTITY_THRESHOLD:
                return self._make_response(
                    request, 422, {"error": "Quantity exceeds allowed maximum"}
                )

        response = self.base_transport.handle_request(request)

        # Bug 2: a SECOND successful cancel of the same order. Judged on the
        # base target's own verdict (200 == the order exists and was cancelled)
        # so cancelling a nonexistent order is still a plain 404.
        if method == "POST" and response.status_code == 200:
            m = _ORDER_PATH_RE.match(path)
            if m:
                order_id = m.group(1)
                if order_id in self._cancelled:
                    return self._make_response(request, 409, {"error": "Order already canceled"})
                self._cancelled.add(order_id)

        return response

    # Alias for test compatibility and potential alternative interface
    handle = handle_request

    @staticmethod
    def _json_body(request: httpx.Request) -> dict:
        try:
            body = json.loads(request.content.decode("utf-8", errors="ignore")) if request.content else {}
        except (json.JSONDecodeError, ValueError):
            return {}
        return body if isinstance(body, dict) else {}

    @staticmethod
    def _make_response(request: httpx.Request, status_code: int, body: dict) -> httpx.Response:
        """Create a real httpx.Response."""
        return httpx.Response(
            status_code=status_code,
            headers={"content-type": "application/json"},
            content=json.dumps(body).encode("utf-8"),
            request=request,
        )

    def reset_state(self, reset_base: bool = False):
        """Forget which orders were cancelled (new experiment on the same target)."""
        self._cancelled.clear()
        if reset_base and hasattr(self.base_transport, "reset_state"):
            self.base_transport.reset_state()

    def close(self):
        """Close the wrapped transport."""
        if hasattr(self.base_transport, "close"):
            self.base_transport.close()
