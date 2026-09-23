import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spec.parser import load_spec, parse_endpoints, Field
from corpus.seed_dict import build_seed_corpus, load_sample_inputs, SeedCorpus
from corpus.fallback_generator import generate_fallback_value
from corpus.resolver import resolve_value

FIXTURE_SPEC = Path(__file__).parent / "fixtures" / "sample_api.yaml"
FIXTURE_SAMPLES = Path(__file__).parent / "fixtures" / "sample_inputs.json"


def _endpoints():
    spec = load_spec(FIXTURE_SPEC)
    return parse_endpoints(spec)


def test_corpus_picks_up_spec_examples_and_enums():
    corpus = build_seed_corpus(_endpoints())

    # 'item_id' has a spec-declared example
    assert "sku-example-1" in corpus.values_for("createOrder", "item_id")
    # 'priority' is an enum query param -- both options should be seeded
    priority_values = corpus.values_for("createOrder", "priority")
    assert set(priority_values) == {"standard", "expedited"}


def test_load_sample_inputs_parses_fixture():
    samples = load_sample_inputs(FIXTURE_SAMPLES)
    assert len(samples) == 3
    assert samples[0].operation_id == "createOrder"
    assert samples[0].payload == {"item_id": "sku-123", "quantity": 2}


def test_corpus_merges_sample_inputs_into_field_values_and_full_examples():
    samples = load_sample_inputs(FIXTURE_SAMPLES)
    corpus = build_seed_corpus(_endpoints(), samples)

    # per-field values from recorded samples
    item_ids = corpus.values_for("createOrder", "item_id")
    assert "sku-123" in item_ids and "sku-456" in item_ids
    # spec example should ALSO still be present alongside recorded values
    assert "sku-example-1" in item_ids

    # full known-good payloads preserved for whole-request seeding
    full = corpus.full_examples["createOrder"]
    assert {"item_id": "sku-123", "quantity": 2} in full
    assert {"item_id": "sku-456", "quantity": 1} in full


def test_corpus_deduplicates_values():
    samples = load_sample_inputs(FIXTURE_SAMPLES)
    corpus = build_seed_corpus(_endpoints(), samples)
    order_ids = corpus.values_for("getOrder", "orderId")
    assert order_ids.count("ord-9") == 1


def test_sample_value_returns_none_when_unseeded():
    corpus = SeedCorpus()
    assert corpus.sample_value("nonexistent_op", "nonexistent_field") is None


def test_fallback_generator_respects_enum():
    field = Field(name="priority", type="string", enum=["standard", "expedited"])
    rng = random.Random(42)
    for _ in range(20):
        assert generate_fallback_value(field, rng=rng) in ("standard", "expedited")


def test_fallback_generator_respects_format():
    field = Field(name="id", type="string", format="uuid")
    val = generate_fallback_value(field, rng=random.Random(1))
    assert len(val) == 36 and val.count("-") == 4


def test_fallback_generator_type_coverage():
    rng = random.Random(7)
    assert isinstance(generate_fallback_value(Field(name="n", type="integer"), rng=rng), int)
    assert isinstance(generate_fallback_value(Field(name="n", type="number"), rng=rng), float)
    assert isinstance(generate_fallback_value(Field(name="n", type="boolean"), rng=rng), bool)
    assert isinstance(generate_fallback_value(Field(name="n", type="string"), rng=rng), str)


def test_resolver_prefers_seed_when_seed_probability_is_one():
    samples = load_sample_inputs(FIXTURE_SAMPLES)
    corpus = build_seed_corpus(_endpoints(), samples)
    field = Field(name="item_id", type="string")
    rng = random.Random(0)

    for _ in range(20):
        val = resolve_value("createOrder", field, corpus, rng=rng, seed_probability=1.0)
        assert val in corpus.values_for("createOrder", "item_id")


def test_resolver_falls_back_when_seed_probability_is_zero():
    samples = load_sample_inputs(FIXTURE_SAMPLES)
    corpus = build_seed_corpus(_endpoints(), samples)
    field = Field(name="item_id", type="string")
    rng = random.Random(0)

    seeded_values = set(corpus.values_for("createOrder", "item_id"))
    fallback_seen = False
    for _ in range(20):
        val = resolve_value("createOrder", field, corpus, rng=rng, seed_probability=0.0)
        if val not in seeded_values:
            fallback_seen = True
    assert fallback_seen  # with prob 0, should never take the seeded path


def test_resolver_falls_back_gracefully_when_no_seed_exists_at_all():
    corpus = SeedCorpus()
    field = Field(name="mystery_field", type="integer")
    rng = random.Random(3)
    val = resolve_value("unknown_op", field, corpus, rng=rng, seed_probability=1.0)
    assert isinstance(val, int)
