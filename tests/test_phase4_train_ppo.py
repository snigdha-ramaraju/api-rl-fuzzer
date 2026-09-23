"""
Phase 4 tests: PPO training loop against the existing (unmodified)
ApiFuzzEnv. Everything here runs against the in-memory mock Orders API
(src/executor/mock_target.py) -- no network, no live server, no cost --
per the project rule that the training loop must be validated against
the same mock target Phase 3 uses before it's ever pointed at a real one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from agent.train_ppo import build_env, build_model, train, build_arg_parser

FIXTURE_SPEC = Path(__file__).parent / "fixtures" / "sample_api.yaml"


def _make_mock_env(max_steps=10, seed=None, seed_probability=1.0):
    return build_env(
        FIXTURE_SPEC,
        use_mock=True,
        max_steps_per_episode=max_steps,
        seed_probability=seed_probability,
        seed=seed,
    )


def _mean_episode_reward(policy, n_episodes=20, base_seed=1000, max_steps=10):
    """Rolls out `n_episodes` full episodes on a fresh mock env each time
    and returns the mean total episode reward. policy=None means take
    uniformly random actions (the untrained baseline)."""
    total = 0.0
    for i in range(n_episodes):
        env = _make_mock_env(max_steps=max_steps, seed=base_seed + i)
        obs, _info = env.reset(seed=base_seed + i)
        done = truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            if policy is None:
                action = env.action_space.sample()
            else:
                action, _state = policy.predict(obs, deterministic=True)
                action = int(action)
            obs, reward, done, truncated, _info = env.step(action)
            ep_reward += reward
        env.close()
        total += ep_reward
    return total / n_episodes


def _train_small_model(timesteps, seed=7, max_steps=10) -> PPO:
    env = _make_mock_env(max_steps=max_steps, seed=seed)
    monitored = Monitor(env)
    vec_env = DummyVecEnv([lambda: monitored])
    model = PPO(
        "MlpPolicy", vec_env,
        n_steps=64, batch_size=64, n_epochs=10,
        learning_rate=3e-4, gamma=0.95, seed=seed, verbose=0,
    )
    model.learn(total_timesteps=timesteps)
    vec_env.close()
    return model


# ---------- env wiring reused from Phase 3 ----------

def test_build_env_against_mock_target_matches_phase3_action_space():
    env = _make_mock_env()
    assert env.action_space.n == 3  # createOrder, getOrder, cancelOrder
    assert env.observation_space.shape == (4,)
    env.close()


# ---------- training loop mechanics ----------

def test_training_run_completes_without_error_on_mock_target():
    model = _train_small_model(timesteps=256)
    assert isinstance(model, PPO)


def test_training_run_produces_saved_model_artifact(tmp_path):
    parser = build_arg_parser()
    output = tmp_path / "ppo_api_fuzzer"
    args = parser.parse_args([
        "--mock",
        "--timesteps", "256",
        "--n-steps", "64",
        "--batch-size", "64",
        "--n-epochs", "4",
        "--seed", "3",
        "--output", str(output),
        "--verbose", "0",
    ])
    model = train(args)
    assert isinstance(model, PPO)

    saved_file = output.with_suffix(".zip")
    assert saved_file.exists()
    assert saved_file.stat().st_size > 0


def test_build_model_uses_requested_hyperparameters():
    env = _make_mock_env()
    monitored = Monitor(env)
    vec_env = DummyVecEnv([lambda: monitored])
    parser = build_arg_parser()
    args = parser.parse_args([
        "--mock", "--n-steps", "32", "--batch-size", "32",
        "--n-epochs", "3", "--learning-rate", "0.001", "--gamma", "0.9",
        "--seed", "1", "--verbose", "0",
    ])
    model = build_model(vec_env, args)
    assert model.n_steps == 32
    assert model.batch_size == 32
    assert model.n_epochs == 3
    assert model.gamma == 0.9
    vec_env.close()


# ---------- learning signal ----------

def test_ppo_policy_improves_mean_reward_over_random_baseline_on_mock_target():
    """The mock Orders API's reward is bounded per episode (coverage and
    anomaly bonuses are both novelty-gated per episode -- see env/reward.py
    and env/coverage.py -- so repeating an identical call cannot be farmed
    for reward). The learnable signal here is SEQUENCING: an agent that
    reliably visits the undeclared-404 anomaly on getOrder/cancelOrder
    while cold, then creates an order and chains into it, collects
    everything discoverable in only a handful of steps; a uniformly random
    policy stumbles into that ordering less reliably within the same short
    episode. A short PPO training run in this tiny 3-action environment
    should still beat a uniformly random policy on mean per-episode
    reward -- this is the concrete "policy improves over training" signal
    for Phase 4, now grounded in genuine sequencing skill rather than in
    the reward-hacking exploit an earlier version of this test relied on."""
    random_baseline = _mean_episode_reward(policy=None, n_episodes=20, base_seed=1000)

    model = _train_small_model(timesteps=6000, seed=7)
    trained_mean = _mean_episode_reward(policy=model, n_episodes=20, base_seed=2000)

    assert trained_mean > random_baseline
