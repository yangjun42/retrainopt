"""Legacy-aligned data state materialization for paper reproduction timelines."""

from __future__ import annotations

from typing import Any, List, Sequence

from ..adapters import ArrayTimelineAdapter


def build_state_timeline(
    train_blocks: Sequence[Sequence[Any]],
    eval_blocks: Sequence[Sequence[Any]],
    *,
    strategy: str,
    sliding_window_k: int,
) -> ArrayTimelineAdapter:
    return ArrayTimelineAdapter(
        train_blocks=_materialize_blocks(train_blocks, strategy=strategy, sliding_window_k=sliding_window_k),
        eval_blocks=_materialize_blocks(eval_blocks, strategy=strategy, sliding_window_k=sliding_window_k),
    )


def _materialize_blocks(
    blocks: Sequence[Sequence[Any]],
    *,
    strategy: str,
    sliding_window_k: int,
) -> List[List[Any]]:
    normalized = [list(block) for block in blocks]
    materialized: List[List[Any]] = []
    mode = (strategy or "individual").strip().lower()

    for step in range(len(normalized)):
        if mode == "individual":
            start = step
        elif mode == "cumulative":
            start = 0
        elif mode == "sliding_window":
            window = max(1, int(sliding_window_k or 1))
            start = max(0, step - window + 1)
        else:
            raise ValueError(f"Unsupported paper data state strategy: {strategy}")

        block: List[Any] = []
        for source_step in range(start, step + 1):
            block.extend(normalized[source_step])
        materialized.append(block)

    return materialized
