"""
In-memory mock target for the E-Commerce Benchmark API
(tests/fixtures/ecommerce_api.yaml).

Simulates multi-resource operations (users, carts, items, orders, payments)
with multi-step state transitions and optional reachable planted bugs.
"""

from __future__ import annotations

import json
import re
import httpx

_USER_ID_RE = re.compile(r"^usr-\d+$")
_USER_PATH_RE = re.compile(r"^/users/([^/]+)$")
_USER_CART_RE = re.compile(r"^/users/([^/]+)/cart$")
_CART_ITEMS_RE = re.compile(r"^/carts/([^/]+)/items$")
_CART_CHECKOUT_RE = re.compile(r"^/carts/([^/]+)/checkout$")
_ORDER_PAY_RE = re.compile(r"^/orders/([^/]+)/pay$")

ECOMMERCE_BIG_QTY_THRESHOLD = 9000

ECOMMERCE_PLANTED_BUGS = {
    "bug_ecommerce_invalid_user_id": {
        "name": "getUser_malformed_user_id",
        "operation_id": "getUser",
        "status": 400,
        "description": "GET /users/<malformed-id> returns undeclared 400 Bad Request",
    },
    "bug_ecommerce_double_checkout": {
        "name": "checkoutCart_already_checked_out",
        "operation_id": "checkoutCart",
        "status": 409,
        "description": "Checking out an already-checked-out cart returns undeclared 409 Conflict",
    },
    "bug_ecommerce_oversized_item_qty": {
        "name": "addItemToCart_quantity_too_large",
        "operation_id": "addItemToCart",
        "status": 422,
        "description": f"POST /carts/{{id}}/items with quantity > {ECOMMERCE_BIG_QTY_THRESHOLD} returns undeclared 422 Unprocessable Entity",
    },
}


def classify_planted_bug_ecommerce(operation_id: str, status_code: int | None) -> str | None:
    """Classifies an anomaly on the ecommerce target as a planted bug or false positive."""
    for key, bug in ECOMMERCE_PLANTED_BUGS.items():
        if bug["operation_id"] == operation_id and bug["status"] == status_code:
            return key
    return None


class EcommerceMockTransport(httpx.BaseTransport):
    """Simulates the E-Commerce benchmark API in-memory."""

    def __init__(self, use_planted_bugs: bool = False):
        self.use_planted_bugs = use_planted_bugs
        self.users: dict[str, dict] = {}
        self.carts: dict[str, dict] = {}
        self.orders: dict[str, dict] = {}
        self.user_counter = 0
        self.cart_counter = 0
        self.order_counter = 0

    def reset_state(self) -> None:
        self.users.clear()
        self.carts.clear()
        self.orders.clear()
        self.user_counter = 0
        self.cart_counter = 0
        self.order_counter = 0

    def _json_body(self, request: httpx.Request) -> dict:
        try:
            return json.loads(request.content.decode("utf-8")) if request.content else {}
        except Exception:
            return {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        method = request.method.upper()
        path = request.url.path

        # 1. POST /users
        if method == "POST" and path == "/users":
            body = self._json_body(request)
            self.user_counter += 1
            user_id = f"usr-{self.user_counter}"
            name = body.get("name", "User")
            email = body.get("email", f"{user_id}@example.com")
            self.users[user_id] = {"userId": user_id, "name": name, "email": email}
            return httpx.Response(201, json=self.users[user_id])

        # 2. GET /users/{userId}
        m_user = _USER_PATH_RE.match(path)
        if method == "GET" and m_user:
            user_id = m_user.group(1)
            # Planted bug 1: malformed user ID
            if self.use_planted_bugs and not _USER_ID_RE.match(user_id):
                return httpx.Response(400, json={"error": "Malformed user ID"})
            if user_id in self.users:
                return httpx.Response(200, json=self.users[user_id])
            return httpx.Response(404, json={"error": "User not found"})

        # 3. POST /users/{userId}/cart
        m_cart = _USER_CART_RE.match(path)
        if method == "POST" and m_cart:
            user_id = m_cart.group(1)
            if user_id in self.users:
                self.cart_counter += 1
                cart_id = f"cart-{self.cart_counter}"
                self.carts[cart_id] = {
                    "cartId": cart_id,
                    "userId": user_id,
                    "items": [],
                    "status": "open",
                }
                return httpx.Response(201, json={"cartId": cart_id, "userId": user_id, "status": "open"})
            return httpx.Response(404, json={"error": "User not found"})

        # 4. POST /carts/{cartId}/items
        m_items = _CART_ITEMS_RE.match(path)
        if method == "POST" and m_items:
            cart_id = m_items.group(1)
            body = self._json_body(request)
            quantity = body.get("quantity", 1)

            # Planted bug 3: oversized or invalid quantity
            if self.use_planted_bugs and isinstance(quantity, int) and quantity > ECOMMERCE_BIG_QTY_THRESHOLD:
                return httpx.Response(422, json={"error": "Quantity exceeds inventory limit"})

            if cart_id in self.carts:
                self.carts[cart_id]["items"].append(body)
                count = len(self.carts[cart_id]["items"])
                return httpx.Response(200, json={"cartId": cart_id, "itemsCount": count})
            return httpx.Response(404, json={"error": "Cart not found"})

        # 5. POST /carts/{cartId}/checkout
        m_checkout = _CART_CHECKOUT_RE.match(path)
        if method == "POST" and m_checkout:
            cart_id = m_checkout.group(1)
            if cart_id in self.carts:
                cart = self.carts[cart_id]
                # Planted bug 2: double checkout
                if self.use_planted_bugs and cart.get("status") == "checked_out":
                    return httpx.Response(409, json={"error": "Cart has already been checked out"})

                cart["status"] = "checked_out"
                self.order_counter += 1
                order_id = f"ord-{self.order_counter}"
                self.orders[order_id] = {
                    "orderId": order_id,
                    "cartId": cart_id,
                    "status": "pending_payment",
                    "total": 49.99,
                }
                return httpx.Response(201, json={"orderId": order_id, "cartId": cart_id, "total": 49.99})
            return httpx.Response(404, json={"error": "Cart not found"})

        # 6. POST /orders/{orderId}/pay
        m_pay = _ORDER_PAY_RE.match(path)
        if method == "POST" and m_pay:
            order_id = m_pay.group(1)
            if order_id in self.orders:
                self.orders[order_id]["status"] = "paid"
                return httpx.Response(200, json={"orderId": order_id, "status": "paid"})
            return httpx.Response(404, json={"error": "Order not found"})

        return httpx.Response(404, json={"error": "Unmatched route"})


def make_ecommerce_mock_transport(use_planted_bugs: bool = False) -> EcommerceMockTransport:
    return EcommerceMockTransport(use_planted_bugs=use_planted_bugs)
