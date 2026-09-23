"""
Phase 6: Planted-bug mock target.
Extends the base mock Orders API with intentional bugs to test anomaly discovery.
Bugs are genuinely triggerable through normal API usage.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


# Global state to track cancel attempts per order
_cancel_attempts: dict[str, int] = {}


class PlantedBugsMockTarget(httpx.BaseTransport):
    """
    HTTPX-compatible transport that wraps a base MockTransport
    and injects intentionally planted bugs.
    """

    def __init__(self, base_transport):
        self.base_transport = base_transport

        self.bug_definitions = {
            "bug_getorder_negative": {
                "name": "getOrder_negative_id",
                "description": (
                    "GET /orders/<negative-id> returns undeclared 400 Bad Request"
                ),
                "status": 400,
                "body": {"error": "Invalid order ID"},
            },
            "bug_cancelorder_double": {
                "name": "cancelOrder_already_canceled",
                "description": (
                    "POST /orders/<id>/cancel twice returns undeclared 409 Conflict"
                ),
                "status": 409,
                "body": {"error": "Order already canceled"},
            },
            "bug_createorder_missing": {
                "name": "createOrder_missing_field",
                "description": (
                    "POST /orders with missing item_id returns undeclared "
                    "422 Unprocessable Entity"
                ),
                "status": 422,
                "body": {"error": "Missing required field: item_id"},
            },
        }

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Intercept requests and inject planted bugs when triggered."""

        # Bug 1: GET /orders/<negative-id>
        if self._is_get_order_negative(request):
            return self._make_response(
                request,
                400,
                {"error": "Invalid order ID"},
            )

        # Bug 2: POST /orders/<id>/cancel twice
        if self._is_cancel_order_duplicate(request):
            return self._make_response(
                request,
                409,
                {"error": "Order already canceled"},
            )

        # Bug 3: POST /orders with missing item_id
        if self._is_create_order_missing_field(request):
            return self._make_response(
                request,
                422,
                {"error": "Missing required field: item_id"},
            )

        # Normal request -> base mock API
        return self.base_transport.handle_request(request)

    # Alias for test compatibility and potential alternative interface
    handle = handle_request

    def _is_get_order_negative(self, request: httpx.Request) -> bool:
        """Detect GET /orders/<negative-id>."""
        try:
            if request.method.upper() != "GET":
                return False

            url = str(request.url)

            if "/orders/" not in url:
                return False

            parts = url.split("/orders/")

            if len(parts) < 2:
                return False

            id_part = parts[1].split("?")[0].strip("/")

            try:
                order_id = int(id_part)
                return order_id < 0
            except (ValueError, IndexError):
                return False

        except (AttributeError, TypeError):
            return False

    def _is_cancel_order_duplicate(self, request: httpx.Request) -> bool:
        """Detect POST /orders/<id>/cancel called more than once."""
        try:
            if request.method.upper() != "POST":
                return False

            url = str(request.url)

            if "/cancel" not in url:
                return False

            parts = url.split("/orders/")

            if len(parts) < 2:
                return False

            id_part = parts[1].split("/")[0].strip()

            if not id_part:
                return False

            if id_part not in _cancel_attempts:
                _cancel_attempts[id_part] = 0

            _cancel_attempts[id_part] += 1

            return _cancel_attempts[id_part] > 1

        except (AttributeError, TypeError, IndexError):
            return False

    def _is_create_order_missing_field(self, request: httpx.Request) -> bool:
        """Detect POST /orders with quantity but missing item_id."""
        try:
            if request.method.upper() != "POST":
                return False

            url = str(request.url)

            if "/orders" not in url:
                return False

            body_bytes = request.content

            if not body_bytes:
                return False

            body_str = body_bytes.decode("utf-8", errors="ignore")

            try:
                body_obj = json.loads(body_str)

                return (
                    isinstance(body_obj, dict)
                    and "item_id" not in body_obj
                    and "quantity" in body_obj
                )

            except (json.JSONDecodeError, ValueError):
                return (
                    "item_id" not in body_str
                    and "quantity" in body_str
                )

        except (AttributeError, TypeError):
            return False

    def _make_response(
        self,
        request: httpx.Request,
        status_code: int,
        body: dict[str, Any],
    ) -> httpx.Response:
        """Create a real httpx.Response."""
        return httpx.Response(
            status_code=status_code,
            headers={"content-type": "application/json"},
            content=json.dumps(body).encode("utf-8"),
            request=request,
        )

    def reset_state(self):
        """Reset planted-bug state for a new experiment/episode."""
        global _cancel_attempts
        _cancel_attempts.clear()

    def close(self):
        """Close the wrapped transport."""
        if hasattr(self.base_transport, "close"):
            self.base_transport.close()