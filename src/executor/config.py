"""
Loads config/target.yaml into the values RequestRunner needs. Kept
separate from request_runner.py so the runner itself has no file-IO
dependency and stays trivially testable with explicit constructor args.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TargetConfig:
    base_url: str
    auth_header: dict | None
    rate_limit_per_sec: float
    max_requests_per_run: int


def load_target_config(path: str | Path) -> TargetConfig:
    raw = yaml.safe_load(Path(path).read_text())
    t = raw["target"]
    auth = t.get("auth_header")
    return TargetConfig(
        base_url=t["base_url"],
        auth_header={"Authorization": auth} if auth else None,
        rate_limit_per_sec=t.get("rate_limit_per_sec", 5),
        max_requests_per_run=t.get("max_requests_per_run", 5000),
    )
