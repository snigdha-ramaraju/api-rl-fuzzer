"""
Tests for the E-Commerce Benchmark API target:
- OpenAPI spec parsing and dependency inference
- Stateful mock transport behavior
- Ground-truth reachable planted bug injection and classification
- ApiFuzzEnv end-to-end integration with the second API
"""

from __future__ import annotations

from pathlib import Path
import httpx
import pytest

from spec.parser import load_spec, parse_endpoints
from spec.graph import build_dependency_graph
from executor.mock_ecommerce_target import (
    EcommerceMockTransport,
    make_ecommerce_mock_transport,
    classify_planted_bug_ecommerce,
    ECOMMERCE_PLANTED_BUGS,
)
from executor.planted_bugs import classify_planted_bug
from agent.train_ppo import build_env

ECOMMERCE_SPEC = Path(__file__).parent / "fixtures" / "ecommerce_api.yaml"


def test_ecommerce_spec_parsing_and_dependency_inference():
    spec = load_spec(ECOMMERCE_SPEC)
    endpoints = parse_endpoints(spec)
    assert len(endpoints) == 6

    op_ids = {e.operation_id for e in endpoints}
    assert op_ids == {
        "createUser",
        "getUser",
        "createCart",
        "addItemToCart",
        "checkoutCart",
        "payOrder",
    }

    edges, unresolved = build_dependency_graph(endpoints)

    # Dependency assertions:
    # getUser depends on createUser (via userId)
    get_user_deps = [edge.producer_op for edge in edges if edge.consumer_op == "getUser"]
    assert "createUser" in get_user_deps

    # createCart depends on createUser (via userId)
    create_cart_deps = [edge.producer_op for edge in edges if edge.consumer_op == "createCart"]
    assert "createUser" in create_cart_deps

    # addItemToCart depends on createCart (via cartId)
    add_item_deps = [edge.producer_op for edge in edges if edge.consumer_op == "addItemToCart"]
    assert "createCart" in add_item_deps

    # checkoutCart depends on createCart (via cartId)
    checkout_deps = [edge.producer_op for edge in edges if edge.consumer_op == "checkoutCart"]
    assert "createCart" in checkout_deps

    # payOrder depends on checkoutCart (via orderId)
    pay_order_deps = [edge.producer_op for edge in edges if edge.consumer_op == "payOrder"]
    assert "checkoutCart" in pay_order_deps


def test_ecommerce_mock_transport_state_transitions():
    transport = make_ecommerce_mock_transport(use_planted_bugs=False)
    client = httpx.Client(transport=transport, base_url="http://mock")

    # 1. Create user
    res1 = client.post("/users", json={"name": "Bob", "email": "bob@test.com"})
    assert res1.status_code == 201
    user_id = res1.json()["userId"]
    assert user_id == "usr-1"

    # 2. Get user
    res2 = client.get(f"/users/{user_id}")
    assert res2.status_code == 200
    assert res2.json()["userId"] == user_id

    # 3. Create cart
    res3 = client.post(f"/users/{user_id}/cart")
    assert res3.status_code == 201
    cart_id = res3.json()["cartId"]
    assert cart_id == "cart-1"

    # 4. Add items
    res4 = client.post(f"/carts/{cart_id}/items", json={"productId": "item-9", "quantity": 2})
    assert res4.status_code == 200
    assert res4.json()["itemsCount"] == 1

    # 5. Checkout cart
    res5 = client.post(f"/carts/{cart_id}/checkout", json={"paymentMethod": "credit_card"})
    assert res5.status_code == 201
    order_id = res5.json()["orderId"]
    assert order_id == "ord-1"

    # 6. Pay order
    res6 = client.post(f"/orders/{order_id}/pay", json={"amount": 49.99})
    assert res6.status_code == 200
    assert res6.json()["status"] == "paid"

    # Reset state
    transport.reset_state()
    res_after = client.get(f"/users/{user_id}")
    assert res_after.status_code == 404


def test_ecommerce_planted_bugs_reachability_and_classification():
    transport = make_ecommerce_mock_transport(use_planted_bugs=True)
    client = httpx.Client(transport=transport, base_url="http://mock")

    # Bug 1: malformed user ID
    res1 = client.get("/users/invalid_id_format")
    assert res1.status_code == 400
    bug1 = classify_planted_bug("getUser", 400)
    assert bug1 == "bug_ecommerce_invalid_user_id"
    assert classify_planted_bug_ecommerce("getUser", 400) == "bug_ecommerce_invalid_user_id"

    # Setup user & cart
    client.post("/users", json={"name": "Alice", "email": "alice@test.com"})
    res_cart = client.post("/users/usr-1/cart")
    cart_id = res_cart.json()["cartId"]

    # Bug 3: oversized item quantity
    res_qty = client.post(f"/carts/{cart_id}/items", json={"productId": "item-1", "quantity": 9500})
    assert res_qty.status_code == 422
    bug3 = classify_planted_bug("addItemToCart", 422)
    assert bug3 == "bug_ecommerce_oversized_item_qty"

    # First checkout (legal)
    res_chk1 = client.post(f"/carts/{cart_id}/checkout")
    assert res_chk1.status_code == 201

    # Bug 2: double checkout
    res_chk2 = client.post(f"/carts/{cart_id}/checkout")
    assert res_chk2.status_code == 409
    bug2 = classify_planted_bug("checkoutCart", 409)
    assert bug2 == "bug_ecommerce_double_checkout"


def test_ecommerce_env_build_and_step():
    env = build_env(
        ECOMMERCE_SPEC,
        use_mock=True,
        max_steps_per_episode=10,
        seed=42,
        use_planted_bugs=True,
    )
    assert env.action_space.n == 6
    assert env.observation_space.shape == (7,)  # 6 operations + progress

    obs, info = env.reset(seed=42)
    assert len(obs) == 7

    action = 0  # createUser
    obs, reward, done, truncated, step_info = env.step(action)
    assert "operation_id" in step_info
    assert "status_code" in step_info
    env.close()


def test_ecommerce_runner_integration(tmp_path):
    from experiment.runner import _run_greedy_policy
    from experiment.tracker import ExperimentTracker

    db_path = tmp_path / "ecom_exp.db"
    exp_id = _run_greedy_policy(
        spec_path=ECOMMERCE_SPEC,
        n_episodes=2,
        max_steps_per_episode=5,
        use_planted_bugs=True,
        trial_seed=42,
        db_path=db_path,
        trial_id=0,
    )

    tracker = ExperimentTracker(db_path)
    metrics = tracker.compute_final_metrics(exp_id)
    assert metrics is not None
    assert metrics["policy"] == "greedy"
    assert metrics["eval_requests"] == 10
