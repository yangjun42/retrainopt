"""Aggregate normalized Phase 1 run summaries into report-friendly artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from .cache import save_json


def aggregate_summaries(artifact_root: Path | str) -> Dict[str, object]:
    root = Path(artifact_root)
    summaries_dir = root / "summaries"
    rows: List[Dict[str, object]] = []

    if summaries_dir.exists():
        for path in sorted(summaries_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(payload)

    by_study: Dict[str, int] = {}
    for row in rows:
        study = str(row.get("study", "unknown"))
        by_study[study] = by_study.get(study, 0) + 1

    return {
        "run_count": len(rows),
        "studies": by_study,
        "rows": rows,
    }


def write_aggregate_reports(artifact_root: Path | str) -> Dict[str, Path]:
    root = Path(artifact_root)
    report = aggregate_summaries(root)
    comparisons_dir = root / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    json_path = comparisons_dir / "aggregate_summary.json"
    save_json(json_path, report)

    csv_path = comparisons_dir / "aggregate_summary.csv"
    rows = report["rows"]
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["run_id"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {"json": json_path, "csv": csv_path}
