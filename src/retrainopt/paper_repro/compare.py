"""Comparison helpers for offline reproduction summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ComparisonTarget:
    """Expected reference metrics for a run."""

    run_id: str
    total_cost: float
    tolerance: float = 1e-6


def compare_summaries(actual_summaries: Sequence[object], targets: Iterable[ComparisonTarget]) -> dict:
    """Compare actual run summaries against reference targets."""
    actual_map = {summary.run_id: summary for summary in actual_summaries}
    mismatches = []

    for target in targets:
        summary = actual_map.get(target.run_id)
        if summary is None:
            mismatches.append(
                {
                    "run_id": target.run_id,
                    "reason": "missing-summary",
                }
            )
            continue

        delta = abs(float(summary.total_cost) - float(target.total_cost))
        if delta > float(target.tolerance):
            mismatches.append(
                {
                    "run_id": target.run_id,
                    "expected_total_cost": float(target.total_cost),
                    "actual_total_cost": float(summary.total_cost),
                    "delta": delta,
                    "tolerance": float(target.tolerance),
                }
            )

    return {
        "status": "ok" if not mismatches else "drift",
        "mismatches": mismatches,
    }
