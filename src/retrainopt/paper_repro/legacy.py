"""Legacy paper semantics shared by the internal reproduction harness."""

from __future__ import annotations

import random
from typing import Any, Dict, Sequence


def build_initial_future_losses(
    *,
    num_steps: int,
    behavior: str,
    base_loss: float,
    degradation_rate: float,
    effective_train_time: int = -1,
) -> Dict[int, float]:
    """Build the initial-model loss stream used by the legacy paper runtime."""
    normalized = (behavior or "ZERO_KNOWLEDGE_INF_LOSS").strip().upper()
    losses: Dict[int, float] = {}
    for step in range(max(0, int(num_steps))):
        if normalized == "ZERO_KNOWLEDGE_INF_LOSS":
            losses[step] = float("inf")
            continue
        if normalized == "FIXED_PERFORMANCE":
            losses[step] = float(base_loss)
            continue
        if normalized == "DEGRADING_PERFORMANCE":
            age = max(0, int(step) - int(effective_train_time))
            losses[step] = float(base_loss) + float(degradation_rate) * float(age)
            continue
        raise ValueError(f"Unsupported initial model behavior: {behavior}")
    return losses


def sample_legacy_subset(block: Sequence[Any], *, n_samples: int, seed: int) -> list[Any]:
    """Mirror the legacy ``torch.utils.data.random_split`` subset semantics."""
    data = list(block)
    actual_n = min(max(0, int(n_samples)), len(data))
    if actual_n <= 0:
        return []
    if actual_n >= len(data):
        return data

    try:
        import torch
        from torch.utils.data import random_split
    except ImportError:  # pragma: no cover - optional dependency fallback
        indices = list(range(len(data)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        return [data[index] for index in indices[:actual_n]]

    subset, _ = random_split(
        data,
        [actual_n, len(data) - actual_n],
        generator=torch.Generator().manual_seed(int(seed)),
    )
    return [data[index] for index in subset.indices]
