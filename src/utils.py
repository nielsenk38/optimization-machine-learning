"""Utility functions for reproducible experiments."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy and PyTorch.

    Full determinism can reduce performance and is not guaranteed across all
    hardware/software combinations, but these settings make runs much more
    reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some PyTorch versions/devices do not support this call fully.
            pass


def get_device(device_arg: str = "auto") -> torch.device:
    """Return the requested compute device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as pretty JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cpu_count_for_dataloader() -> int:
    """Choose a conservative DataLoader worker count.

    The default experiment uses 0 workers for reproducibility. This helper is
    here in case you later want to increase it locally.
    """
    return max(0, min(4, (os.cpu_count() or 1) - 1))
