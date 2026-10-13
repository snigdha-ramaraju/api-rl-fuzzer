"""
Phase 4 -- PPO training loop using stable-baselines3 against the existing
`ApiFuzzEnv` (Phase 3), with its interface left completely unmodified.

Phase 5 adds optional episode seeding from the local LLM brainstormer
(src/brainstormer/, src/env/scenario_seeding.py). It is wired in here as
an opt-in layer: pass --enable-brainstormer to wrap the environment with
ScenarioSeededEnv; leave it unset and this script behaves exactly as it
did in Phase 4, byte-for-byte. The LLM is never on the step-level decision
path either way -- see scenario_seeding.py and brainstormer/pool.py.

Scope, deliberately narrow:
  - Consumes env/api_env.py::ApiFuzzEnv exactly as Phase 3 left it. The
    brainstormer only ever seeds episodes via a wrapper; it has no hook
    into training mechanics, the reward function, or per-step decisions.
  - No SQLite persistence, no real-target integration beyond "point it at
    a target.yaml if you have one", no per-field mutation actions. The
    action space stays one discrete action per endpoint, as Phase 3
    defined it.

Cost/safety notes (project must stay Rs0):
  - stable-baselines3 and torch are free/open-source/local -- no paid API
    calls are introduced by this file.
  - The brainstormer (--enable-brainstormer) talks ONLY to a locally-running
    Ollama instance. If Ollama isn't reachable, LLMClient fails loudly with
    setup instructions rather than silently falling back to anything paid.
  - Training against a real/live target still goes through Phase 3's
    `RequestRunner`, which enforces the rate limit and hard request budget
    from `config/target.yaml`. This script does not bypass that.
  - Always validate against the bundled mock target first (`--mock`) --
    this is the exact in-memory Orders-API simulation Phase 3's tests use
    (src/executor/mock_target.py), so the training loop itself can be
    proven correct before it is ever pointed at a real network.

Usage:
    # Dry-run: train against the in-memory mock Orders API. No network,
    # no live server, no cost. This is how the training loop is validated.
    python -m agent.train_ppo --mock --timesteps 4000

    # Same, but with LLM-seeded episodes (requires `ollama serve` running
    # locally and a model pulled -- see brainstormer/llm_client.py).
    python -m agent.train_ppo --mock --timesteps 4000 --enable-brainstormer

    # Real target: driven by a real OpenAPI spec + config/target.yaml.
    python -m agent.train_ppo \\
        --spec path/to/openapi.yaml \\
        --target-config config/target.yaml \\
        --timesteps 100000
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Allow running as `python -m agent.train_ppo` or as a plain script from
# anywhere, without requiring the package to be pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# stable_baselines3 (and the torch it pulls in) is imported lazily, inside
# build_model()/train(), rather than at module scope. This means build_env()
# and build_arg_parser() -- which Phase 5's ScenarioSeededEnv tests exercise
# -- can be used and tested without torch installed. Nothing about Phase 4's
# own training behavior changes: build_model()/train() still require and
# import stable_baselines3 exactly as before, just deferred to call time.

from spec.parser import load_spec, parse_endpoints, Endpoint
from spec.graph import build_dependency_graph
from corpus.seed_dict import build_seed_corpus, load_sample_inputs, SeedCorpus
from executor.request_runner import RequestRunner
from executor.config import load_target_config
from executor.mock_target import make_mock_transport
from env.api_env import ApiFuzzEnv
from env.scenario_seeding import ScenarioSeededEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOCK_SPEC = REPO_ROOT / "tests" / "fixtures" / "sample_api.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "ppo_api_fuzzer"


def build_env(
    spec_path: str | Path,
    *,
    target_config_path: str | Path | None = None,
    sample_inputs_path: str | Path | None = None,
    use_mock: bool = False,
    max_steps_per_episode: int = 50,
    seed_probability: float = 0.7,
    seed: int | None = None,
    enable_brainstormer: bool = False,
    ollama_host: str = "http://localhost:11434",
    llm_model_tier: str = "auto",
    brainstorm_interval_episodes: int = 500,
    max_scenarios_per_call: int = 8,
    seed_episode_probability: float = 0.15,
    api_description: str = "",
    use_planted_bugs: bool = False,
    llm_client: object | None = None,
    prepare_at_reset: bool = True,
):
    """Wires up an ApiFuzzEnv exactly the way Phase 3's own tests do,
    just parameterized so this same function serves both the mock
    dry-run path and a real-target run. When enable_brainstormer=False
    (the default -- unchanged from Phase 4), returns a plain ApiFuzzEnv.
    When True, returns that same env wrapped in ScenarioSeededEnv (Phase 5)."""
    spec = load_spec(spec_path)
    endpoints: list[Endpoint] = parse_endpoints(spec)

    sample_inputs = load_sample_inputs(sample_inputs_path) if sample_inputs_path else None
    corpus: SeedCorpus = build_seed_corpus(endpoints, sample_inputs)

    if use_mock:
        from executor.planted_bugs import PlantedBugsMockTarget
        base_transport = make_mock_transport()
        if use_planted_bugs:
            base_transport = PlantedBugsMockTarget(base_transport)
        
        runner = RequestRunner(
            base_url="http://mock",
            rate_limit_per_sec=0,  # no need to throttle an in-memory transport
            max_requests_per_run=10_000_000,
            transport=base_transport,
        )
    else:
        if target_config_path is None:
            raise ValueError(
                "--target-config is required unless --mock is set "
                "(pass config/target.yaml or your own)."
            )
        cfg = load_target_config(target_config_path)
        runner = RequestRunner(
            base_url=cfg.base_url,
            rate_limit_per_sec=cfg.rate_limit_per_sec,
            max_requests_per_run=cfg.max_requests_per_run,
            auth_header=cfg.auth_header,
        )
        print(
            f"[train_ppo] Training against a real/live target: {cfg.base_url} "
            f"(rate_limit={cfg.rate_limit_per_sec}/s, budget={cfg.max_requests_per_run} "
            "requests). Make sure this target is free/self-hosted -- this project "
            "has no paid-API integration anywhere, by design.",
            file=sys.stderr,
        )

    env = ApiFuzzEnv(
        endpoints,
        corpus,
        runner,
        max_steps_per_episode=max_steps_per_episode,
        seed_probability=seed_probability,
        seed=seed,
    )

    if not enable_brainstormer:
        return env

    # Local import: the brainstormer package (and its ollama/psutil/pydantic
    # deps) is only touched when explicitly opted into.
    from brainstormer.llm_client import LLMClient
    from brainstormer.pool import ScenarioPool

    _, unresolved = build_dependency_graph(endpoints)
    if llm_client is None:
        llm_client = LLMClient(host=ollama_host, model_tier_setting=llm_model_tier)
    pool = ScenarioPool(
        llm_client=llm_client,
        endpoints=endpoints,
        corpus=corpus,
        unresolved_fields=unresolved,
        api_description=api_description,
        brainstorm_interval_episodes=brainstorm_interval_episodes,
        max_scenarios_per_call=max_scenarios_per_call,
    )
    return ScenarioSeededEnv(
        env,
        pool,
        seed_episode_probability=seed_episode_probability,
        rng=None if seed is None else random.Random(seed),
        prepare_at_reset=prepare_at_reset,
    )


def build_model(vec_env, args: argparse.Namespace):
    from stable_baselines3 import PPO
    return PPO(
        "MlpPolicy",
        vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        seed=args.seed,
        verbose=args.verbose,
    )


def train(args: argparse.Namespace):
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    spec_path = args.spec or (DEFAULT_MOCK_SPEC if args.mock else None)
    if spec_path is None:
        raise ValueError("--spec is required unless --mock is set.")

    env = build_env(
        spec_path,
        target_config_path=args.target_config,
        sample_inputs_path=args.sample_inputs,
        use_mock=args.mock,
        max_steps_per_episode=args.max_steps_per_episode,
        seed_probability=args.seed_probability,
        seed=args.seed,
        enable_brainstormer=args.enable_brainstormer,
        ollama_host=args.ollama_host,
        llm_model_tier=args.llm_model_tier,
        brainstorm_interval_episodes=args.brainstorm_interval_episodes,
        max_scenarios_per_call=args.max_scenarios_per_call,
        seed_episode_probability=args.seed_episode_probability,
        api_description=args.api_description,
    )

    monitored = Monitor(env)
    vec_env = DummyVecEnv([lambda: monitored])

    model = build_model(vec_env, args)

    callbacks = []
    if args.checkpoint_freq and args.checkpoint_freq > 0:
        checkpoint_dir = Path(args.output).parent / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix="ppo_api_fuzzer",
            )
        )

    model.learn(total_timesteps=args.timesteps, callback=callbacks or None)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"[train_ppo] Saved trained model to {output_path.with_suffix('.zip')}")

    vec_env.close()
    return model


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4: PPO training against ApiFuzzEnv.")
    p.add_argument("--spec", type=str, default=None, help="Path to the OpenAPI spec.")
    p.add_argument("--target-config", type=str, default=None,
                    help="Path to a target.yaml (required unless --mock).")
    p.add_argument("--sample-inputs", type=str, default=None,
                    help="Optional recorded sample-inputs JSON file (Phase 2).")
    p.add_argument("--mock", action="store_true",
                    help="Train against the bundled in-memory mock Orders API "
                         "instead of a real target. Use this to validate the "
                         "training loop before ever touching a real/live target.")
    p.add_argument("--timesteps", type=int, default=20_000)
    p.add_argument("--max-steps-per-episode", type=int, default=50)
    p.add_argument("--seed-probability", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=256,
                    help="PPO rollout length per update (SB3 n_steps).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--checkpoint-freq", type=int, default=0,
                    help="Save a checkpoint every N timesteps (0 = disabled, "
                         "only save the final model).")
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                    help="Path (without extension) to save the final model to.")
    p.add_argument("--verbose", type=int, default=1)

    # Phase 5: local LLM brainstormer (opt-in, off by default -- unset
    # leaves training identical to Phase 4).
    p.add_argument("--enable-brainstormer", action="store_true",
                    help="Wrap the env with LLM-seeded episodes. Requires "
                         "`ollama serve` running locally with a model pulled "
                         "(see src/brainstormer/llm_client.py). No paid API "
                         "involved either way.")
    p.add_argument("--ollama-host", type=str, default="http://localhost:11434")
    p.add_argument("--llm-model-tier", type=str, default="auto",
                    choices=["auto", "small", "medium", "large"])
    p.add_argument("--brainstorm-interval-episodes", type=int, default=500,
                    help="How often (in episodes) to call the LLM for new "
                         "scenario ideas. Never called per-step.")
    p.add_argument("--max-scenarios-per-call", type=int, default=8)
    p.add_argument("--seed-episode-probability", type=float, default=0.15,
                    help="Fraction of episodes that start from an LLM-"
                         "seeded scenario rather than pure agent choice.")
    p.add_argument("--api-description", type=str, default="",
                    help="Optional free-text API description, passed to "
                         "the LLM as extra context.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
