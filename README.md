# api-rl-fuzzer — Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5

RL-based API fuzzing agent. This build covers **Phase 1** (OpenAPI spec
ingestion, auto-derived dependency graph, structural oracle), **Phase 2**
(seed corpus builder), **Phase 3** (Gymnasium RL environment + HTTP
executor), **Phase 4** (PPO training loop via `stable-baselines3` against
the Phase 3 environment, unmodified), and **Phase 5** (mandatory,
strictly-local LLM scenario brainstormer, wired in as opt-in episode
seeding). This is the complete project as currently scoped — no Phase 6
work is included.

## Correctness fix: novelty-gated anomaly reward (reward-hacking closed)

A pre-Phase-6 review found that the anomaly bonus in `env/reward.py` was
paid out on **every** step where the oracle flagged the response as
anomalous, with no novelty check — unlike the coverage bonus, which was
already correctly gated on "have I seen this `(operation, status)` pair
before, this episode?". Since an anomaly is just as easy to trigger
repeatedly as it is once (e.g. calling an endpoint cold, forever, always
gets back the same undeclared-404), an agent could farm the larger
`ANOMALY_BONUS` (2.0, vs. 1.0 for new coverage) without limit by simply
repeating one action — a pure reward-hacking exploit with nothing to do
with discovering new behavior, and one that a short training run could
(and did) learn to exploit instead of learning to sequence calls.

**The fix** (`src/env/coverage.py`, `src/env/reward.py`, `src/env/api_env.py`):
- `CoverageTracker` gained `record_anomaly(operation_id, status_code,
  oracle_result)`, which tracks a per-episode set of anomaly *signatures*
  — `(operation_id, status_code, status_known, shape_valid, missing_fields,
  unexpected_type_fields)` — and returns whether this exact signature is
  new this episode. It's deliberately more specific than the plain
  `(op, status)` coverage key: a genuinely different anomaly on the same
  status code (e.g. a different missing field) still counts as novel,
  while calling the same endpoint over and over and getting the identical
  anomaly back does not. `reset()` clears it alongside `hits`, so novelty
  is correctly scoped per-episode, matching the coverage bonus's existing
  semantics.
- `compute_reward()` gained an `is_new_anomaly` parameter; the anomaly
  bonus is now only paid when `oracle_result.is_anomalous and
  is_new_anomaly`. It defaults to `True` for backward compatibility, but
  that default only ever matters when `oracle_result` is `None` or not
  anomalous, in which case the flag has no effect anyway — real callers
  with an anomalous result must pass the tracker's actual verdict.
- `ApiFuzzEnv.step()` now calls `coverage.record_anomaly(...)` whenever
  the oracle reports an anomaly and threads the result through to
  `compute_reward()`; `info["is_new_anomaly"]` is exposed alongside the
  existing `info["anomalous"]` for observability.

**Nothing else changed.** The action space, observation space, dependency
graph, oracle, corpus, executor, PPO training loop, and LLM brainstormer
are all untouched — this was a scoped fix to one reward-shaping bug, not
an architecture change. `tests/test_phase3_env.py` gained 7 tests directly
covering the fix (signature-level novelty gating, distinguishing different
anomaly reasons, `reset()` clearing both trackers, the reward function's
new parameter in all three states, and a full-stack env-level regression
test proving repeated identical anomalies now earn nothing after the
first). Phase 4's existing "policy improves with training" test still
passes — it turned out to still hold under the fixed reward (the bounded,
novelty-gated reward still rewards good call-sequencing, which is still
learnable), but its docstring has been corrected since it previously
described the exploit this fix removes.

## What's in Phase 1

- `src/spec/parser.py` — parses an OpenAPI 3.x spec into flat `Endpoint`/`Field`
  objects (path/query/header params, request body, per-status response schemas).
  This becomes the action space for the RL agent later.
- `src/spec/graph.py` — auto-derives a producer→consumer dependency graph by
  matching response field names/types to request field names/types (exact +
  suffix matching). No hand-annotated dependencies. Anything ambiguous is
  exported as `unresolved` rather than guessed.
- `src/spec/oracle.py` — a structural response oracle derived directly from
  the spec's response schemas (status code validity, required-field presence,
  type conformance). No hand-written business rules.

## What's in Phase 2

- `src/corpus/seed_dict.py` — builds a **seed corpus** from two sources:
  the spec's own declared `example`/`enum` values (free, always in sync),
  and an optional file of real recorded requests (`load_sample_inputs`).
  Exposes two views: `field_values` (per-field candidate values, for
  mutating one field at a time) and `full_examples` (whole known-good
  request payloads per operation, for seeding an entire call at once).
- `src/corpus/fallback_generator.py` — type/format-aware value generator
  (uuid, email, date-time, enums, bounded ints/floats/strings) used when
  the corpus has nothing seeded for a field. Produces schema-valid
  defaults; it is not a fuzzing strategy, just a sane starting point.
- `src/corpus/resolver.py` — the entry point later phases call:
  `resolve_value(operation_id, field, corpus, seed_probability=0.7)`.
  Prefers a seeded value most of the time but not always, and the ratio
  is a caller-supplied parameter (not hard-coded), since the mutation
  engine / PPO env may want to anneal it over training. Seeds are a warm
  start, never a constraint — the agent can and should mutate past them.

Sample-inputs file format (`tests/fixtures/sample_inputs.json` is a working example):
```json
[
  {"operation_id": "createOrder", "payload": {"item_id": "sku-123", "quantity": 2}},
  {"operation_id": "getOrder", "payload": {"orderId": "ord-9"}}
]
```

## What's in Phase 3

- `src/executor/request_runner.py` — `RequestRunner` executes real HTTP
  calls via `httpx`, enforcing a rate limit and a hard total-request
  **budget cap** so nothing can run away against any target, paid or not.
  Accepts an injectable `transport`, so it's fully testable with
  `httpx.MockTransport` — no live server or network access required.
- `src/executor/config.py` — loads `config/target.yaml` into the values
  `RequestRunner` needs. Kept separate so the runner itself stays file-IO-free
  and trivially testable with explicit constructor args.
- `config/target.yaml` — target base URL, auth header, rate limit, and
  request budget. Points at `http://localhost:8080` by default.
- `src/env/coverage.py` — in-memory tracker of which `(operation_id, status_code)`
  combinations have been observed, feeding both reward novelty and the
  observation vector. No persistence yet (later phase).
- `src/env/reward.py` — reward = coverage novelty + oracle-anomaly bonus −
  invalid-request/transport penalty. No hand-written business rules or
  expected-response tables — "anomalous" is defined purely by Phase 1's
  structural oracle, so RL can find issues nobody wrote down in advance.
- `src/env/action_space.py` — one discrete action per endpoint. Field values
  are resolved automatically: **dependency-graph value from an earlier call
  this episode first**, then Phase 2's seed corpus / fallback generator.
  This is what lets a chained sequence like create → read → cancel work.
- `src/env/api_env.py` — `ApiFuzzEnv(gym.Env)`, the Phase 3 deliverable.
  Wires together the spec parser, dependency graph, oracle, seed corpus,
  and request runner into a standard Gymnasium environment (`reset`/`step`
  with `Discrete` action space and `Box` observation space).

The key integration test (`test_env_chains_create_then_get_via_dependency_graph`)
proves this end to end: calling `createOrder` then `getOrder` in the same
episode returns `200` (found), not `404` — meaning the id from the first
response was automatically piped into the second request's path parameter
via the Phase 1 dependency graph, with zero hand-annotation.

## What's in Phase 4

- `src/executor/mock_target.py` — the Phase 3 test suite's in-memory mock
  Orders API, factored out into its own module so it can be reused (not
  duplicated) as the required "validate before touching a real target"
  target for PPO training.
- `src/agent/train_ppo.py` — the PPO training loop, via `stable-baselines3`,
  against `ApiFuzzEnv` exactly as Phase 3 defined it. No changes to the env
  interface. `build_env()` wires up spec → endpoints → seed corpus →
  `RequestRunner` → `ApiFuzzEnv`, parameterized so the same function drives
  either the bundled mock target (`--mock`) or a real one (`--target-config`).
  No LLM/brainstormer involvement anywhere in this phase — that's Phase 5,
  and even then it only ever seeds episodes, never trains or decides.
  Out of scope here by design: SQLite persistence, real-target integration
  beyond "point it at a target.yaml", and per-field mutation actions (the
  action space stays one discrete action per endpoint).

Run it directly:

```bash
# Validate the training loop against the bundled mock target first —
# no network, no live server, zero cost.
python -m agent.train_ppo --mock --timesteps 20000

# Once validated, point it at a real spec + target (still budget/rate
# limited by config/target.yaml, via the same Phase 3 RequestRunner):
python -m agent.train_ppo --spec path/to/openapi.yaml \
    --target-config config/target.yaml --timesteps 100000
```

`tests/test_phase4_train_ppo.py` (5 tests) all run against the mock target:
a short training run completes without error, produces a saved model
artifact (`model.save(...)` + a non-empty `.zip` file), `build_model()`
correctly wires the requested PPO hyperparameters, and — the actual
learning-signal test — a short-trained policy's mean episode reward on the
mock target reliably beats a uniformly random policy's, because the
mock's reward structure (coverage novelty + oracle anomaly bonus) exposes
a learnable pattern for PPO to find within a few thousand timesteps in
this small, 3-action environment.

## What's in Phase 5

The LLM scenario brainstormer is a **required project component**, but
strictly local and Rs0-cost: it only ever talks to a locally-running
**Ollama** instance (`src/brainstormer/llm_client.py`), never a paid API.
If Ollama isn't reachable, it fails loudly with setup instructions rather
than silently degrading to anything paid. Model tier (`qwen2.5:3b/7b/14b-instruct`)
is auto-selected from detected hardware.

The LLM's role is strictly **propose, not decide**: it brainstorms
multi-step/stateful scenarios that a plain mutation/coverage-guided
exploration would likely miss (ordering effects, auth-boundary crossings,
race conditions, state inconsistencies). PPO remains the actual learner —
the LLM is never on the step-level decision path, never modifies the
reward function, and is invoked only periodically (every
`brainstorm_interval_episodes` episode boundaries), never per-step.

- `src/brainstormer/scenarios.py` — the structured `Scenario`/`ScenarioBatch`
  schema the LLM must emit (enforced directly via Ollama's JSON-schema
  `format` parameter — no prompt-and-pray text parsing anywhere).
- `src/brainstormer/hardware.py` — hardware-based model tier detection
  (RAM / NVIDIA VRAM via `psutil`/`nvidia-smi`) → `qwen2.5:{3b,7b,14b}-instruct`.
- `src/brainstormer/llm_client.py` — the **only** place in the codebase
  that talks to an LLM, and only ever to local Ollama.
- `src/brainstormer/grounding.py` — hard-rejects any scenario referencing
  an `operationId` not present in the real parsed spec. Local 3B–14B models
  hallucinate endpoint names occasionally; this is a filter, not a warning.
- `src/brainstormer/context.py` — builds the LLM's prompt from the spec,
  Phase 2's seed corpus (known-good example payloads), Phase 1's
  `unresolved` dependency-graph fields, and a coverage-so-far summary.
- `src/brainstormer/pool.py` — `ScenarioPool` owns the periodic brainstorm
  cadence and the resulting queue of *grounded* scenarios. The only call
  site that ever invokes the LLM.
- `src/env/scenario_seeding.py` — `ScenarioSeededEnv(gym.Wrapper)`. Wraps
  `ApiFuzzEnv` **without modifying it** (per Phase 3's original design:
  "this module has no dependency on [the brainstormer]"). At each `reset()`,
  with probability `seed_episode_probability` (default 0.15), an episode's
  first N actions are forced to follow an LLM-proposed, grounded scenario's
  operation sequence instead of the agent's own choices; after that
  scripted prefix, control reverts to the agent for the rest of the
  episode. PPO experiences the resulting reward exactly like any other
  rollout — no special-cased reward path.
- `src/agent/train_ppo.py` — extended with **opt-in, backward-compatible**
  flags (`--enable-brainstormer`, `--ollama-host`, `--llm-model-tier`,
  `--brainstorm-interval-episodes`, `--max-scenarios-per-call`,
  `--seed-episode-probability`, `--api-description`). Unset, training
  behaves exactly as Phase 4 left it — `build_env()` returns a plain
  `ApiFuzzEnv`. `stable_baselines3` is now imported lazily inside
  `build_model()`/`train()` rather than at module scope, so `build_env()`
  can be used and tested without torch installed; this is the only change
  to Phase 4's own code, and it does not alter its behavior.

The key integration test
(`test_wrapper_forces_scripted_sequence_and_dependency_graph_still_chains`)
proves LLM-seeded episodes and the Phase 1 dependency graph compose
correctly: a scripted `createOrder → getOrder` sequence still resolves
`getOrder`'s `orderId` from `createOrder`'s response (200, not 404) even
though the actions were forced by the scenario rather than chosen live.

Run it (requires `ollama serve` running locally with a model pulled):

```bash
python -m agent.train_ppo --mock --timesteps 20000 --enable-brainstormer
```

## Requirements

- Python 3.11+
- No paid services, no API keys anywhere in this project.
- Phase 1–4 tests use `httpx.MockTransport` to simulate a target API
  in-memory — no network access or live server needed.
- Phase 5 tests mock `ollama.Client` — **you do not need Ollama installed
  or running to run the test suite.** Ollama is only required if you
  actually run training with `--enable-brainstormer` (see
  `src/brainstormer/llm_client.py` for the one-time local setup).

## Setup

```bash
cd api-rl-fuzzer
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

## Run the tests / smoke test

```bash
pytest tests/ -v
```

Expected output: 56 tests passing —

Phase 1 (`test_phase1_spec.py`, 3 tests):
- `test_parse_endpoints` — confirms the sample spec parses into the three
  expected operations with correct methods/fields.
- `test_dependency_graph_links_create_to_get` — confirms `createOrder`'s
  response `id` field is auto-linked to `getOrder`/`cancelOrder`'s `orderId`
  path parameter, with no manual annotation.
- `test_oracle_flags_missing_required_field` — confirms the oracle correctly
  passes a well-formed response, flags a response missing a required field,
  and flags an undeclared status code.

Phase 2 (`test_phase2_corpus.py`, 11 tests):
- spec-declared examples/enums are picked up automatically into the corpus
- recorded sample inputs merge into both `field_values` and `full_examples`
  alongside spec-derived seeds, with deduplication
- the fallback generator respects enum/format constraints and covers every
  JSON schema type
- the resolver's `seed_probability` correctly controls seeded-vs-fallback
  sampling (tested at both extremes, 1.0 and 0.0) and degrades gracefully
  to fallback generation when a field has no seed at all

Phase 3 (`test_phase3_env.py`, 22 tests):
- the executor executes real calls against a mocked API, correctly enforces
  the request budget, and `config/target.yaml` loads as expected
- the coverage tracker distinguishes new vs. repeat `(op, status)` hits,
  and separately tracks per-episode ANOMALY novelty (`record_anomaly`) —
  the exact same anomaly reported once is not novel again, a genuinely
  different anomaly reason on the same status code still is, and
  `reset()` clears both
- the reward function's novelty bonus, anomaly bonus, and penalties combine
  as designed, including the anomaly bonus's novelty gate (`is_new_anomaly`):
  paid when novel, withheld when not, and inert (no-op) when the response
  isn't anomalous at all
- dependency-graph value resolution takes priority over the seed corpus
  when a prior in-episode response can supply a field, and falls back
  gracefully when it can't
- full environment episodes: correct action/observation space shapes, a
  create→get chain correctly resolving via the dependency graph (200, not
  404), a cold `getOrder` call correctly flagged anomalous (undeclared 404)
  with the bonus paid on first occurrence, episode truncation at
  `max_steps`, graceful episode end on budget exhaustion, and — the
  reward-hacking regression test — repeating that identical cold call
  several more times earns exactly zero additional reward each time,
  proving the exploit is closed end to end

Phase 4 (`test_phase4_train_ppo.py`, 5 tests):
- `build_env()` against the mock target reproduces the same action/observation
  space shapes as Phase 3
- a short PPO training run against the mock target completes without error
- `train()` (the CLI's own entry point) produces a saved, non-empty model
  artifact on disk
- `build_model()` correctly applies requested PPO hyperparameters
  (`n_steps`, `batch_size`, `n_epochs`, `gamma`)
- a short-trained policy's mean episode reward on the mock target beats a
  uniformly random policy's — the concrete "policy improves with training"
  signal, now grounded in sequencing skill within a bounded, novelty-gated
  reward rather than in the reward-hacking exploit an earlier version of
  this project relied on (see "Correctness fix" below)

Phase 5 (`test_phase5_brainstormer.py`, 15 tests):
- hardware-based model tier resolution (`small`/`medium`/`large` → the
  correct `qwen2.5:*-instruct` model) and rejection of an unknown tier
- grounding correctly keeps a scenario referencing only real operation IDs
  and rejects one referencing a hallucinated `deleteEverything` op
- `LLMClient` talks only to a mocked local `ollama.Client` (never anything
  else) and fails loudly with setup instructions when Ollama is unreachable
- the context builder's prompt includes every endpoint and the coverage summary
- `ScenarioPool` only calls the LLM at the configured episode interval
  (not every episode), correctly drains its queue via `pop_scenario()`,
  and drops hallucinated scenarios via grounding before they're queued
- `ScenarioSeededEnv`: a seeded scenario forces the exact scripted action
  sequence regardless of what the caller passes to `step()`, **and the
  dependency-graph chaining proven in Phase 3 still works within a
  scripted sequence** (create→get returns 200, not 404); control correctly
  reverts to the agent after the scripted prefix ends; `seed_episode_probability=0`
  never seeds; an empty scenario pool degrades gracefully
- `build_env()` returns a plain, unwrapped `ApiFuzzEnv` when
  `--enable-brainstormer` is unset (Phase 4 behavior, byte-for-byte), and
  a correctly-wrapped `ScenarioSeededEnv` when it's set

**A note on Phase 4's 5 tests specifically:** they require `stable-baselines3`,
which pulls in `torch` (plus, on Linux/x86_64, several GB of CUDA
dependency wheels that `pip` fetches regardless of whether you have a
GPU). This is a heavy, disk-hungry install — if `pip install -e ".[dev]"`
runs out of disk space on `stable-baselines3`, that's an environment
constraint, not a project issue: everything through Phase 3 and all of
Phase 5 runs and tests with zero dependency on torch. If you hit this,
free up disk (torch's CUDA wheels alone can need 5–7GB) and retry, or run
`pytest tests/ -v --ignore=tests/test_phase4_train_ppo.py` to verify the
other 44 tests independently in the meantime.

## Try it against your own spec

Drop your OpenAPI spec anywhere and run:

```python
from spec.parser import load_spec, parse_endpoints
from spec.graph import build_dependency_graph
from spec.oracle import check_response

spec = load_spec("path/to/your_api.yaml")   # .yaml, .yml, or .json
endpoints = parse_endpoints(spec)

edges, unresolved = build_dependency_graph(endpoints)
print(f"{len(endpoints)} endpoints parsed")
print(f"{len(edges)} dependency edges auto-derived")
print(f"{len(unresolved)} fields flagged for manual/LLM review")
```

(Run this from inside `src/`, or add `src/` to your `PYTHONPATH` — see how
`tests/test_phase1_spec.py` does it via `sys.path.insert`.)

## Folder structure

```
api-rl-fuzzer/
├── pyproject.toml
├── README.md
├── config/
│   └── target.yaml               # target URL, auth, rate limit, request budget
├── src/
│   ├── __init__.py
│   ├── spec/
│   │   ├── __init__.py
│   │   ├── parser.py              # OpenAPI -> Endpoint/Field objects
│   │   ├── graph.py                # auto-derived dependency graph
│   │   └── oracle.py               # structural response oracle
│   ├── corpus/
│   │   ├── __init__.py
│   │   ├── seed_dict.py            # builds seed corpus from spec + sample inputs
│   │   ├── fallback_generator.py   # type/format-aware defaults for unseeded fields
│   │   └── resolver.py             # combined seed-or-fallback entry point
│   ├── executor/
│   │   ├── __init__.py
│   │   ├── request_runner.py       # budget/rate-limited HTTP execution
│   │   ├── config.py               # loads config/target.yaml
│   │   └── mock_target.py          # in-memory mock Orders API (test/dry-run only)
│   ├── env/
│   │   ├── __init__.py
│   │   ├── coverage.py             # in-memory (op, status) coverage tracker
│   │   ├── reward.py               # coverage novelty + oracle anomaly reward
│   │   ├── action_space.py         # dependency-aware field value resolution
│   │   ├── api_env.py              # ApiFuzzEnv(gym.Env) -- the Phase 3 deliverable
│   │   └── scenario_seeding.py     # ScenarioSeededEnv wrapper -- Phase 5 episode seeding
│   ├── agent/
│   │   ├── __init__.py
│   │   └── train_ppo.py            # PPO training loop (Phase 4) + opt-in brainstormer wiring (Phase 5)
│   └── brainstormer/
│       ├── __init__.py
│       ├── scenarios.py            # structured Scenario/ScenarioBatch schema
│       ├── hardware.py             # hardware-based Ollama model tier selection
│       ├── llm_client.py           # the ONLY local-Ollama LLM call site in the project
│       ├── grounding.py            # rejects scenarios referencing unknown operationIds
│       ├── context.py              # builds the LLM's prompt from spec + corpus + coverage
│       └── pool.py                 # periodic brainstorm cadence + grounded scenario queue
└── tests/
    ├── __init__.py
    ├── test_phase1_spec.py
    ├── test_phase2_corpus.py
    ├── test_phase3_env.py
    ├── test_phase4_train_ppo.py
    ├── test_phase5_brainstormer.py
    └── fixtures/
        ├── sample_api.yaml
        └── sample_inputs.json
```

## Not in scope (future work, not started)

- SQLite-backed coverage/anomaly persistence across runs
- Pointing the executor at a real (non-mocked) target API (supported by
  `train_ppo.py`'s `--target-config` flag, but no real target ships with
  this project — bring your own free/self-hosted one)
- Per-field mutation as its own RL action (currently: one action per
  endpoint, with field values resolved via dependency graph → seed corpus
  → fallback generator)
- Any UI/dashboard for reviewing coverage, anomalies, or brainstormed
  scenarios

This is the complete project through Phase 5. Any further work is a new,
explicitly-scoped increment, not an implicit continuation.
