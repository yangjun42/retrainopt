"""Disk cache helpers for offline paper reproduction runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from ..types import CandidateEvaluation


@dataclass(frozen=True)
class ArtifactLayout:
    """Standardized artifact tree for Phase 1 offline reproduction."""

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "cache").mkdir(exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        (self.root / "summaries").mkdir(exist_ok=True)
        (self.root / "comparisons").mkdir(exist_ok=True)
        (self.root / "figures").mkdir(exist_ok=True)

    def cache_run_dir(self, cache_key: str) -> Path:
        path = self.root / "cache" / cache_key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_candidates_dir(self, cache_key: str) -> Path:
        path = self.cache_run_dir(cache_key) / "candidates"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def initial_path(self, cache_key: str) -> Path:
        return self.cache_run_dir(cache_key) / "initial.json"

    def candidate_path(self, cache_key: str, step: int, n_samples: int) -> Path:
        return self.cache_candidates_dir(cache_key) / f"step_{step:04d}_n_{n_samples:08d}.json"

    def summary_path(self, run_id: str) -> Path:
        return self.root / "summaries" / f"{run_id}.json"

    def comparison_path(self, name: str) -> Path:
        return self.root / "comparisons" / f"{name}.json"


def save_candidate_evaluation(path: Path, evaluation: CandidateEvaluation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": list(evaluation.model_id),
        "dataset_size": evaluation.dataset_size,
        "training_seconds": evaluation.training_seconds,
        "training_cost": evaluation.training_cost,
        "future_losses": {str(key): value for key, value in evaluation.future_losses.items()},
        "observed": evaluation.observed,
        "metadata": dict(evaluation.metadata),
    }
    _atomic_write_json(path, payload)


def load_candidate_evaluation(path: Path) -> CandidateEvaluation:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CandidateEvaluation(
        model_id=(int(payload["model_id"][0]), int(payload["model_id"][1])),
        dataset_size=int(payload["dataset_size"]),
        training_seconds=float(payload["training_seconds"]),
        training_cost=float(payload["training_cost"]),
        future_losses={int(key): float(value) for key, value in payload["future_losses"].items()},
        observed=bool(payload.get("observed", True)),
        metadata=dict(payload.get("metadata", {})),
    )


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, payload)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_path, path)
