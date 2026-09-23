import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spec.parser import load_spec, parse_endpoints
from spec.graph import build_dependency_graph
from spec.oracle import check_response

FIXTURE = Path(__file__).parent / "fixtures" / "sample_api.yaml"


def test_parse_endpoints():
    spec = load_spec(FIXTURE)
    endpoints = parse_endpoints(spec)
    op_ids = {e.operation_id for e in endpoints}
    assert op_ids == {"createOrder", "getOrder", "cancelOrder"}

    create = next(e for e in endpoints if e.operation_id == "createOrder")
    assert create.method == "POST"
    field_names = {f.name for f in create.request_body}
    assert field_names == {"item_id", "quantity"}
    assert "201" in create.responses


def test_dependency_graph_links_create_to_get():
    spec = load_spec(FIXTURE)
    endpoints = parse_endpoints(spec)
    edges, unresolved = build_dependency_graph(endpoints)

    # createOrder returns 'id' -> should feed getOrder's/cancelOrder's 'orderId' path param
    linked_consumers = {(e.consumer_op, e.consumer_field) for e in edges if e.producer_op == "createOrder"}
    assert ("getOrder", "orderId") in linked_consumers
    assert ("cancelOrder", "orderId") in linked_consumers


def test_oracle_flags_missing_required_field():
    spec = load_spec(FIXTURE)
    endpoints = parse_endpoints(spec)
    get_order = next(e for e in endpoints if e.operation_id == "getOrder")

    good = check_response(get_order, 200, {"id": "abc", "status": "created"})
    assert good.is_anomalous is False

    bad = check_response(get_order, 200, {"id": "abc"})  # missing 'status'
    assert bad.is_anomalous is True
    assert "status" in bad.missing_fields

    unknown_status = check_response(get_order, 500, {"id": "abc", "status": "created"})
    assert unknown_status.is_anomalous is True
