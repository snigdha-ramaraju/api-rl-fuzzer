"""
Picks the right local Ollama model tier for the brainstormer based on
detected hardware. No network call, no paid API, no telemetry -- just
psutil plus an optional nvidia-smi probe.
"""

from __future__ import annotations

import shutil
import subprocess

import psutil

MODEL_TIERS = {
    "small": "qwen2.5:3b-instruct",
    "medium": "qwen2.5:7b-instruct",
    "large": "qwen2.5:14b-instruct",
}


def _gpu_vram_gb() -> float | None:
    """Best-effort NVIDIA VRAM probe. Returns None if no GPU / no nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        )
        mb = int(out.strip().splitlines()[0])
        return mb / 1024
    except Exception:
        return None


def detect_tier() -> str:
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    vram_gb = _gpu_vram_gb()

    if vram_gb is not None and vram_gb >= 16:
        return "large"
    if vram_gb is not None and vram_gb >= 6:
        return "medium"
    if ram_gb >= 32:
        return "large"
    if ram_gb >= 16:
        return "medium"
    return "small"


def resolve_model(tier_setting: str) -> str:
    """tier_setting is 'auto' | 'small' | 'medium' | 'large'."""
    tier = detect_tier() if tier_setting == "auto" else tier_setting
    if tier not in MODEL_TIERS:
        raise ValueError(f"Unknown model tier '{tier}'. Choose from {list(MODEL_TIERS)} or 'auto'.")
    return MODEL_TIERS[tier]
