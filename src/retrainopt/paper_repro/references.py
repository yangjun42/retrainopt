"""Latest-paper reference targets for clean Phase 1 offline reproduction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def latest_paper_exact_targets() -> Dict[str, float]:
    return {
        "phase1_exact_extreme_sudden_M_seed42": 2.8480,
        "phase1_exact_gradual_drift_M_seed42": 5.0333,
        "phase1_exact_recurring_drift_M_seed42": 20.4002,
        "phase1_exact_airlines_seed42": 41.4583,
        "phase1_exact_electricity_seed42": 37.7083,
        "phase1_exact_covertype_seed42": 31.2553,
    }


def latest_paper_report(artifact_root: Path | str, *, tolerance: float = 1e-3) -> Dict[str, object]:
    root = Path(artifact_root)
    summaries_dir = root / "summaries"
    comparisons_dir = root / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    targets = latest_paper_exact_targets()
    missing = []
    mismatches = []
    matches = []

    for run_id, expected_total in targets.items():
        path = summaries_dir / f"{run_id}.json"
        if not path.exists():
            missing.append(run_id)
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        actual_total = float(payload["total_cost"])
        delta = abs(actual_total - expected_total)
        if delta > tolerance:
            mismatches.append(
                {
                    "run_id": run_id,
                    "expected_total_cost": expected_total,
                    "actual_total_cost": actual_total,
                    "delta": delta,
                    "tolerance": tolerance,
                }
            )
        else:
            matches.append(run_id)

    status = "ok"
    if missing:
        status = "drift" if mismatches else "missing"
    if mismatches:
        status = "drift"

    report = {
        "status": status,
        "matches": matches,
        "missing": missing,
        "mismatches": mismatches,
    }
    output_path = comparisons_dir / "latest_paper_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
