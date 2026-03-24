"""Offline paper reproduction runner with deterministic disk caching."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from ..adapters import BinaryTimelineAdapter, PrototypeModelAdapter, SklearnModelAdapter, TorchModelAdapter
from ..evaluation import EvaluationEngine
from ..planner import RetrainingPlanner
from ..types import CandidateEvaluation, DecisionAction
from .cache import ArtifactLayout, load_candidate_evaluation, save_candidate_evaluation, save_json
from .manifest import DatasetSpec, PaperManifest, RunSpec
from .synthetic import build_synthetic_timeline


@dataclass
class RunSummary:
    """Normalized run summary for Phase 1 artifacts."""

    run_id: str
    study: str
    dataset: str
    method: str
    model: str
    seed: int
    timeline_steps: int
    total_cost: float
    training_cost: float
    loss_cost: float
    estimated_total_cost: float | None
    evaluated_candidates: int
    evaluation_training_seconds: float
    oracle_schedule_training_seconds: float | None
    rog: float | None
    rmt: float | None
    acquisition: str
    bayes_backend: str
    cache_fingerprint: str
    manifest_version: str
    decisions: Sequence[dict] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "study": self.study,
            "dataset": self.dataset,
            "method": self.method,
            "model": self.model,
            "seed": self.seed,
            "timeline_steps": self.timeline_steps,
            "total_cost": self.total_cost,
            "training_cost": self.training_cost,
            "loss_cost": self.loss_cost,
            "estimated_total_cost": self.estimated_total_cost,
            "evaluated_candidates": self.evaluated_candidates,
            "evaluation_training_seconds": self.evaluation_training_seconds,
            "oracle_schedule_training_seconds": self.oracle_schedule_training_seconds,
            "rog": self.rog,
            "rmt": self.rmt,
            "acquisition": self.acquisition,
            "bayes_backend": self.bayes_backend,
            "cache_fingerprint": self.cache_fingerprint,
            "manifest_version": self.manifest_version,
            "decisions": list(self.decisions),
        }


class DiskBackedEvaluationEngine(EvaluationEngine):
    """Evaluation engine that persists candidates to a deterministic cache tree."""

    def __init__(
        self,
        *,
        timeline,
        model_adapter,
        config,
        layout: ArtifactLayout,
        cache_key: str,
    ):
        super().__init__(timeline=timeline, model_adapter=model_adapter, config=config)
        self.layout = layout
        self.cache_key = cache_key

    def evaluate_initial_model(self, initial_model: Any | None = None) -> CandidateEvaluation:
        if initial_model is not None:
            return super().evaluate_initial_model(initial_model=initial_model)

        cache_path = self.layout.initial_path(self.cache_key)
        if cache_path.exists():
            evaluation = load_candidate_evaluation(cache_path)
            self._initial_cache = evaluation
            return evaluation

        evaluation = super().evaluate_initial_model(initial_model=initial_model)
        save_candidate_evaluation(cache_path, evaluation)
        return evaluation

    def evaluate_candidate(self, step: int, n_samples: int) -> CandidateEvaluation:
        actual_n = min(max(0, int(n_samples)), self.timeline.train_size(step))
        cache_path = self.layout.candidate_path(self.cache_key, step, actual_n)
        if cache_path.exists():
            evaluation = load_candidate_evaluation(cache_path)
            self._candidate_cache[evaluation.model_id] = evaluation
            return evaluation

        evaluation = super().evaluate_candidate(step, n_samples)
        save_candidate_evaluation(cache_path, evaluation)
        return evaluation


class OfflinePaperRunner:
    """Execute internal offline paper reproduction runs."""

    def __init__(
        self,
        manifest: PaperManifest,
        *,
        artifact_root: Path,
        model_overrides: Dict[str, Any] | None = None,
    ):
        self.manifest = manifest
        self.layout = ArtifactLayout(Path(artifact_root))
        self.model_overrides = dict(model_overrides or {})

    def run(self, run_id: str) -> RunSummary:
        run = self.manifest.get_run(run_id)
        dataset = self.manifest.get_dataset(run.dataset)
        timeline = self._build_timeline(dataset, run.seed)
        model_adapter = self._build_model_adapter(run, dataset)

        planner = RetrainingPlanner(timeline, model_adapter, config=run.planner_config())
        cache_key = run.cache_key(dataset)
        planner.engine = DiskBackedEvaluationEngine(
            timeline=timeline,
            model_adapter=model_adapter,
            config=planner.config,
            layout=self.layout,
            cache_key=cache_key,
        )
        result = planner.plan(method=run.method)

        evaluation_training_seconds = sum(
            float(candidate.training_seconds)
            for candidate in planner.engine.iter_cached_candidates()
        )
        oracle_schedule_training_seconds = self._oracle_schedule_training_seconds(
            result.decisions,
            planner.engine,
        )

        summary = RunSummary(
            run_id=run.run_id,
            study=run.study,
            dataset=run.dataset,
            method=result.method,
            model=run.model,
            seed=run.seed,
            timeline_steps=timeline.num_steps,
            total_cost=float(result.total_cost),
            training_cost=float(result.training_cost),
            loss_cost=float(result.loss_cost),
            estimated_total_cost=None if result.estimated_total_cost is None else float(result.estimated_total_cost),
            evaluated_candidates=len(list(planner.engine.iter_cached_candidates())),
            evaluation_training_seconds=evaluation_training_seconds,
            oracle_schedule_training_seconds=oracle_schedule_training_seconds,
            rog=None,
            rmt=self._safe_divide(evaluation_training_seconds, oracle_schedule_training_seconds),
            acquisition=run.acquisition,
            bayes_backend=run.bayes_backend,
            cache_fingerprint=cache_key,
            manifest_version=self.manifest.version,
            decisions=result.to_dict()["decisions"],
        )

        reference_run = self.manifest.find_reference_exact_run(run)
        if reference_run is not None and reference_run.run_id != run.run_id:
            reference_path = self.layout.summary_path(reference_run.run_id)
            if reference_path.exists():
                reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
                summary.rog = self._safe_divide(
                    summary.total_cost - float(reference_payload["total_cost"]),
                    float(reference_payload["total_cost"]),
                )
                summary.rmt = self._safe_divide(
                    evaluation_training_seconds,
                    float(reference_payload.get("oracle_schedule_training_seconds") or 0.0),
                )

        save_json(self.layout.summary_path(run.run_id), summary.to_dict())
        return summary

    def _build_timeline(self, dataset: DatasetSpec, seed: int):
        if dataset.source == "synthetic":
            return build_synthetic_timeline(dataset, seed)
        if dataset.source == "binary":
            if not dataset.binary_dir:
                raise ValueError(f"Binary dataset {dataset.name} is missing binary_dir.")
            return BinaryTimelineAdapter(dataset.binary_dir, num_features=dataset.num_features)
        raise ValueError(f"Unsupported dataset source: {dataset.source}")

    def _build_model_adapter(self, run: RunSpec, dataset: DatasetSpec):
        if run.model in self.model_overrides:
            return self.model_overrides[run.model]
        if run.model == "prototype":
            return PrototypeModelAdapter()
        if run.model in {"logistic_regression", "random_forest", "gradient_boosting"}:
            return SklearnModelAdapter(self._build_sklearn_factory(run.model, run.seed))
        if run.model == "paper_mlp":
            hidden_dims = list(dataset.model_defaults.get("hidden_dims", [128, 64]))
            epochs = int(dataset.model_defaults.get("epochs", 5))
            learning_rate = float(dataset.model_defaults.get("learning_rate", 0.01))
            batch_size = int(dataset.model_defaults.get("batch_size", 64))
            return TorchModelAdapter(
                self._build_torch_mlp_factory(dataset.num_features, hidden_dims),
                optimizer_factory=self._build_torch_optimizer_factory(learning_rate),
                epochs=epochs,
                batch_size=batch_size,
            )
        raise ValueError(f"Unsupported internal paper model preset: {run.model}")

    @staticmethod
    def _build_sklearn_factory(model_name: str, random_seed: int):
        def _factory():
            try:
                from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Sklearn-based paper reproduction requires sklearn.") from exc

            if model_name == "logistic_regression":
                return LogisticRegression(max_iter=1000, random_state=random_seed)
            if model_name == "random_forest":
                return RandomForestClassifier(n_estimators=100, random_state=random_seed)
            if model_name == "gradient_boosting":
                return GradientBoostingClassifier(random_state=random_seed)
            raise ValueError(f"Unsupported sklearn paper model: {model_name}")

        return _factory

    @staticmethod
    def _build_torch_mlp_factory(num_features: int, hidden_dims: Sequence[int]):
        def _factory():
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Paper MLP reproduction requires the `torch` extra.") from exc

            layers = []
            in_features = int(num_features)
            for hidden_dim in hidden_dims:
                layers.append(torch.nn.Linear(in_features, int(hidden_dim)))
                layers.append(torch.nn.ReLU())
                in_features = int(hidden_dim)
            layers.append(torch.nn.Linear(in_features, 2))
            return torch.nn.Sequential(*layers)

        return _factory

    @staticmethod
    def _build_torch_optimizer_factory(learning_rate: float):
        def _factory(parameters):
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Paper MLP reproduction requires the `torch` extra.") from exc
            return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)

        return _factory

    @staticmethod
    def _oracle_schedule_training_seconds(decisions, engine: EvaluationEngine) -> float:
        total = 0.0
        for step, decision in enumerate(decisions):
            if decision.action != DecisionAction.RETRAIN:
                continue
            evaluation = engine.evaluate_candidate(step, int(decision.n_samples or 0))
            total += float(evaluation.training_seconds)
        return total

    @staticmethod
    def _safe_divide(numerator: float, denominator: float | None) -> float | None:
        if denominator is None or abs(float(denominator)) <= 1e-12:
            return None
        return float(numerator) / float(denominator)
