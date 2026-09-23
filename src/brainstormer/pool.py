"""
Owns the periodic brainstorm cadence and the resulting queue of grounded
scenarios. This is the ONLY place the LLM is invoked -- once every
`brainstorm_interval_episodes` episode boundaries, never per-step, per the
project's core design decision: LLM proposes, PPO disposes, and the LLM
never sits on the training critical path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spec.parser import Endpoint
from corpus.seed_dict import SeedCorpus
from .context import build_brainstorm_prompt
from .grounding import validate_batch
from .scenarios import Scenario


@dataclass
class ScenarioPool:
    llm_client: object  # duck-typed: needs .brainstorm(prompt, max_scenarios) -> ScenarioBatch
    endpoints: list[Endpoint]
    corpus: SeedCorpus
    unresolved_fields: list[dict] = field(default_factory=list)
    api_description: str = ""
    brainstorm_interval_episodes: int = 500
    max_scenarios_per_call: int = 8

    _queue: list[Scenario] = field(default_factory=list, repr=False)
    _episode_count: int = field(default=0, repr=False)
    last_rejected: list[dict] = field(default_factory=list, repr=False)

    def maybe_refresh(self, coverage_summary: dict | None = None) -> int:
        """Call once per episode boundary. Returns the number of newly
        added, grounded scenarios (0 if this call didn't trigger a
        brainstorm cycle, or if the LLM produced nothing groundable)."""
        self._episode_count += 1
        if self._episode_count % self.brainstorm_interval_episodes != 0:
            return 0

        prompt = build_brainstorm_prompt(
            self.endpoints, self.corpus,
            coverage_summary=coverage_summary,
            unresolved_fields=self.unresolved_fields,
            api_description=self.api_description,
        )
        batch = self.llm_client.brainstorm(prompt, self.max_scenarios_per_call)
        report = validate_batch(batch, self.endpoints)

        self._queue.extend(report.valid)
        self.last_rejected = report.rejected
        return len(report.valid)

    def pop_scenario(self) -> Scenario | None:
        if not self._queue:
            return None
        return self._queue.pop(0)

    def __len__(self) -> int:
        return len(self._queue)
