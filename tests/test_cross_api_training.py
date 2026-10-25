"""
Tests for Cross-API Environment, Multi-API Training, and Held-Out Generalization.

Validates:
1. Variable endpoint counts mapping to identical observation and action spaces.
2. Strict action masking (never remapping invalid actions; inactive actions raise ValueError).
3. Semantic slot feature vector extraction (active, covered, method, deps, producer, frequency).
4. Multi-API environment with randomized API selection and strict state isolation on reset.
5. Multi-API MaskablePPO training and saved model persistence.
6. Zero-shot evaluation on a completely unseen, held-out API.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import numpy as np

from spec.parser import load_spec, parse_endpoints
from corpus.seed_dict import build_seed_corpus
from executor.request_runner import RequestRunner
from env.cross_api_env import (
    CrossApiFuzzEnv,
    MultiApiFuzzEnv,
    DEFAULT_MAX_SLOTS,
    OBSERVATION_DIM,
    make_tiny_mock_transport,
    resolve_mock_transport_for_spec,
)
from agent.train_ppo import (
    build_cross_api_env,
    build_multi_api_env,
    build_maskable_model,
    train_cross_api,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SPEC = REPO_ROOT / "tests" / "fixtures" / "sample_api.yaml"        # 3 endpoints
ECOMMERCE_SPEC = REPO_ROOT / "tests" / "fixtures" / "ecommerce_api.yaml"  # 6 endpoints
TINY_SPEC = REPO_ROOT / "tests" / "fixtures" / "tiny_api.yaml"            # 2 endpoints (held-out)
CRAPI_SPEC = REPO_ROOT / "datasets" / "openapi" / "crapi-openapi-spec.json"  # 44 endpoints


# =========================================================================
# 1. Variable Endpoint Counts & Space Invariance Tests
# =========================================================================

def test_variable_endpoint_counts_share_identical_spaces():
    """Verify that APIs with 2, 3, 6, and 44 endpoints share identical fixed spaces."""
    env_tiny = build_cross_api_env(TINY_SPEC, max_slots=64)
    env_sample = build_cross_api_env(SAMPLE_SPEC, max_slots=64)
    env_ecom = build_cross_api_env(ECOMMERCE_SPEC, max_slots=64)
    env_crapi = build_cross_api_env(CRAPI_SPEC, max_slots=64)

    # All must have identical Discrete(64) and Box(386,)
    for env in (env_tiny, env_sample, env_ecom, env_crapi):
        assert env.action_space.n == 64
        assert env.observation_space.shape == (OBSERVATION_DIM,)
        assert env.observation_space.shape == (386,)
        env.close()


def test_crapi_spec_44_endpoints_loads_in_cross_api_env():
    """Verify OWASP crAPI spec: all 44 endpoints fit, exactly 44 action-mask entries active,
    observation space shape is (386,) and action space is Discrete(64)."""
    env = build_cross_api_env(CRAPI_SPEC, max_slots=64)
    obs, info = env.reset()

    # Space dimensions
    assert env.action_space.n == 64
    assert env.observation_space.shape == (386,)
    assert obs.shape == (386,)

    # Exactly 44 endpoints active
    assert len(env.unwrapped.endpoints) == 44
    mask = env.unwrapped.action_masks()
    assert len(mask) == 64
    assert mask.sum() == 44
    assert mask[:44].all()
    assert not mask[44:].any()

    # Step on valid action
    obs_next, reward, done, trunc, step_info = env.step(0)
    assert not step_info.get("budget_exceeded")

    # Stepping inactive slot 44 strictly raises ValueError (never remaps)
    with pytest.raises(ValueError, match="is inactive"):
        env.step(44)

    env.close()


# =========================================================================
# 2. Strict Action Masking & Rejection (Never Remap)
# =========================================================================

def test_action_masking_matches_active_endpoint_count():
    """Verify action_masks() returns True only for active slots."""
    env_sample = build_cross_api_env(SAMPLE_SPEC, max_slots=64)  # 3 endpoints
    mask = env_sample.unwrapped.action_masks()
    assert mask.shape == (64,)
    assert mask[:3].all()
    assert not mask[3:].any()
    env_sample.close()

    env_ecom = build_cross_api_env(ECOMMERCE_SPEC, max_slots=64)  # 6 endpoints
    mask_ecom = env_ecom.unwrapped.action_masks()
    assert mask_ecom.shape == (64,)
    assert mask_ecom[:6].all()
    assert not mask_ecom[6:].any()
    env_ecom.close()


def test_stepping_inactive_slot_raises_value_error_never_remaps():
    """Verify that selecting an inactive slot raises ValueError and never remaps to another endpoint."""
    env = build_cross_api_env(SAMPLE_SPEC, max_slots=64)  # valid: 0, 1, 2
    env.reset()

    # Step on valid action
    obs, reward, done, trunc, info = env.step(0)
    assert info["operation_id"] == "createOrder"

    # Step on inactive slot (slot 3, 10, 44, or 63)
    for invalid_action in (3, 10, 44, 63):
        with pytest.raises(ValueError, match="is inactive"):
            env.step(invalid_action)

    env.close()


# =========================================================================
# 3. Semantic Observation Feature Extraction
# =========================================================================

def test_semantic_observation_features_and_padding():
    """Verify feature encoding for active slots and zeros for padding slots."""
    env = build_cross_api_env(SAMPLE_SPEC, max_slots=64)
    obs, _ = env.reset()

    # 3 endpoints * 6 features = 18 slot features
    # Slot 0: createOrder (POST, active=1, covered=0, method=POST (0.4), is_producer=1)
    slot_0 = obs[:6]
    assert slot_0[0] == 1.0  # is_active
    assert slot_0[1] == 0.0  # is_covered (fresh reset)
    assert slot_0[2] == pytest.approx(0.4)  # POST method code
    assert slot_0[4] == 1.0  # is_producer

    # Slot 3..63 must be strictly padded with 0.0
    padding_features = obs[3 * 6 : 64 * 6]
    assert len(padding_features) == 61 * 6
    assert np.all(padding_features == 0.0)

    # Global features (progress, coverage_ratio)
    global_features = obs[64 * 6 :]
    assert len(global_features) == 2
    assert global_features[0] == 0.0  # progress at reset
    assert global_features[1] == 0.0  # coverage ratio at reset

    # Step once on slot 0 and verify coverage updates
    obs_next, _, _, _, _ = env.step(0)
    assert obs_next[1] == 1.0  # slot 0 is now covered!
    assert obs_next[-1] > 0.0  # overall coverage ratio increased
    env.close()


# =========================================================================
# 4. Multi-API Environment & Clean State Isolation Across Resets
# =========================================================================

def test_multi_api_env_switching_and_state_isolation():
    """Verify MultiApiFuzzEnv switches APIs cleanly and isolates state across episodes."""
    specs = [SAMPLE_SPEC, ECOMMERCE_SPEC]
    env = MultiApiFuzzEnv(spec_paths=specs, max_slots=64, max_steps_per_episode=10)

    # Force reset to API 0 (sample_api, 3 endpoints)
    obs0, info0 = env.reset(options={"spec_idx": 0})
    assert info0["n_endpoints"] == 3
    assert "sample_api" in info0["spec_path"]
    mask0 = env.action_masks()
    assert mask0[:3].all() and not mask0[3:].any()

    # Step in API 0
    obs, r, d, t, info = env.step(0)
    assert env.known_values != {}  # populated order ID
    assert len(env.coverage.hits) > 0

    # Force reset to API 1 (ecommerce_api, 6 endpoints)
    obs1, info1 = env.reset(options={"spec_idx": 1})
    assert info1["n_endpoints"] == 6
    assert "ecommerce_api" in info1["spec_path"]
    mask1 = env.action_masks()
    assert mask1[:6].all() and not mask1[6:].any()

    # Verify state from API 0 did not bleed into API 1
    assert env.known_values == {}  # clean reset
    assert len(env.coverage.hits) == 0  # clean reset
    assert env._step_count == 0

    # Step on slot 5 in ecommerce (cancelOrder in ecommerce)
    obs, r, d, t, info = env.step(5)
    assert info["operation_id"] in [e.operation_id for e in env.endpoints]
    env.close()


def test_multi_api_env_random_selection():
    """Verify that default reset randomly selects across all provided specs."""
    specs = [SAMPLE_SPEC, ECOMMERCE_SPEC]
    env = MultiApiFuzzEnv(spec_paths=specs, max_slots=64, seed=42)

    seen_specs = set()
    for i in range(20):
        _, info = env.reset(seed=100 + i)
        seen_specs.add(info["spec_path"])

    # Both specs must have been chosen across 20 random resets
    assert len(seen_specs) == 2
    env.close()


# =========================================================================
# 5. Multi-API Training, Saving, Loading, and Zero-Shot Generalization
# =========================================================================

def test_multi_api_ppo_training_and_zero_shot_generalization(tmp_path):
    """
    Full pipeline test:
    1. Train MaskablePPO across 2 distinct APIs (Sample + Ecommerce).
    2. Save the trained model to disk.
    3. Load the model from disk.
    4. Evaluate zero-shot on an unseen held-out API (Tiny Auth API, 2 endpoints).
    5. Verify strict action masking prevents sampling inactive slots (2..63).
    """
    from sb3_contrib import MaskablePPO

    training_specs = [SAMPLE_SPEC, ECOMMERCE_SPEC]
    model_output = tmp_path / "cross_api_reusable_policy"

    # 1. Train across both APIs with 64 slots
    model = train_cross_api(
        spec_paths=training_specs,
        output_path=model_output,
        timesteps=256,
        max_slots=64,
    )
    assert model is not None

    # 2. Verify model artifact exists
    saved_zip = model_output.with_suffix(".zip")
    assert saved_zip.exists()
    assert saved_zip.stat().st_size > 0

    # 3. Load saved model
    loaded_model = MaskablePPO.load(str(saved_zip))
    assert isinstance(loaded_model, MaskablePPO)

    # 4. Zero-shot evaluation on completely unseen held-out API
    held_out_env = build_cross_api_env(TINY_SPEC, max_slots=64, max_steps_per_episode=10)
    obs, info = held_out_env.reset()

    # 5. Test 10 consecutive action predictions on the held-out API
    for _ in range(10):
        masks = held_out_env.unwrapped.action_masks()
        action, _ = loaded_model.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)

        # Must only pick active slots 0 or 1, never 2..63
        assert action in (0, 1), f"Model selected masked inactive slot {action}!"

        obs, reward, done, trunc, step_info = held_out_env.step(action)
        if done or trunc:
            obs, info = held_out_env.reset()

    held_out_env.close()
