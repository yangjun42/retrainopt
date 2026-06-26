"""Backend-selectable surrogate implementations for bayes planning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from ..config import PlannerConfig
from ..types import CandidateEvaluation
from .surrogate import NearestNeighborRegressor


TimeFeatureFn = Callable[[int, int, int], Sequence[float]]
LossFeatureFn = Callable[[int, int, int, int], Sequence[float]]


class BaseFittedSurrogate(ABC):
    """Predictive interface shared by all bayes surrogate backends."""

    @abstractmethod
    def predict_training_cost(self, step: int, n_value: int, dataset_size: int) -> Tuple[float, float]:
        """Predict training cost mean/std for a candidate."""

    @abstractmethod
    def predict_loss(self, step: int, n_value: int, dataset_size: int, eval_step: int) -> Tuple[float, float]:
        """Predict loss mean/std for a candidate and evaluation step."""

    def predict_losses(
        self,
        step: int,
        n_value: int,
        dataset_size: int,
        eval_steps: Iterable[int],
    ) -> List[Tuple[float, float]]:
        """Predict losses for a candidate over many evaluation steps."""
        return [
            self.predict_loss(step, n_value, dataset_size, eval_step)
            for eval_step in eval_steps
        ]


class BaseBayesBackend(ABC):
    """Factory interface for bayes surrogate backends."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        evaluations: Iterable[CandidateEvaluation],
        *,
        config: PlannerConfig,
        time_feature_fn: TimeFeatureFn,
        loss_feature_fn: LossFeatureFn,
    ) -> BaseFittedSurrogate:
        """Fit a surrogate bundle from observed candidate evaluations."""


@dataclass
class _LightweightFittedSurrogate(BaseFittedSurrogate):
    time_regressor: NearestNeighborRegressor
    loss_regressor: NearestNeighborRegressor
    time_feature_fn: TimeFeatureFn
    loss_feature_fn: LossFeatureFn

    def predict_training_cost(self, step: int, n_value: int, dataset_size: int) -> Tuple[float, float]:
        return self.time_regressor.predict(self.time_feature_fn(step, n_value, dataset_size))

    def predict_loss(self, step: int, n_value: int, dataset_size: int, eval_step: int) -> Tuple[float, float]:
        return self.loss_regressor.predict(self.loss_feature_fn(step, n_value, dataset_size, eval_step))

    def predict_losses(
        self,
        step: int,
        n_value: int,
        dataset_size: int,
        eval_steps: Iterable[int],
    ) -> List[Tuple[float, float]]:
        feature_rows = [
            self.loss_feature_fn(step, n_value, dataset_size, eval_step)
            for eval_step in eval_steps
        ]
        return self.loss_regressor.predict_many(feature_rows)


class LightweightBayesBackend(BaseBayesBackend):
    """Default dependency-free surrogate backend."""

    name = "lightweight"

    def fit(
        self,
        evaluations: Iterable[CandidateEvaluation],
        *,
        config: PlannerConfig,
        time_feature_fn: TimeFeatureFn,
        loss_feature_fn: LossFeatureFn,
    ) -> BaseFittedSurrogate:
        time_rows: List[Tuple[float, ...]] = []
        time_targets: List[float] = []
        loss_rows: List[Tuple[float, ...]] = []
        loss_targets: List[float] = []

        for evaluation in evaluations:
            step, n_value = evaluation.model_id
            size = max(1, evaluation.dataset_size)
            time_rows.append(tuple(map(float, time_feature_fn(step, n_value, size))))
            time_targets.append(float(evaluation.training_cost))
            for eval_step, loss_value in evaluation.future_losses.items():
                loss_rows.append(tuple(map(float, loss_feature_fn(step, n_value, size, eval_step))))
                loss_targets.append(float(loss_value))

        if not time_rows or not loss_rows:
            raise RuntimeError("LightweightBayesBackend requires at least one observed evaluation.")

        time_regressor = NearestNeighborRegressor(
            k_neighbors=config.surrogate_k,
            uncertainty_floor=config.surrogate_uncertainty_floor,
        )
        loss_regressor = NearestNeighborRegressor(
            k_neighbors=config.surrogate_k,
            uncertainty_floor=config.surrogate_uncertainty_floor,
        )
        time_regressor.fit(time_rows, time_targets)
        loss_regressor.fit(loss_rows, loss_targets)
        return _LightweightFittedSurrogate(
            time_regressor=time_regressor,
            loss_regressor=loss_regressor,
            time_feature_fn=time_feature_fn,
            loss_feature_fn=loss_feature_fn,
        )


class _BoTorchRegressor:
    """Lazy-import BoTorch regressor wrapper."""

    def __init__(self, rows: Sequence[Sequence[float]], targets: Sequence[float]):
        if len(rows) != len(targets):
            raise ValueError("BoTorch regressor rows/targets length mismatch.")
        if not rows:
            raise ValueError("BoTorch regressor requires at least one training row.")

        self._constant_value = None
        self._constant_std = 0.0
        self._min_vals = None
        self._range_vals = None
        self._target_mean = 0.0
        self._target_std = 1.0
        self._model = None
        self._torch = None
        self._gpytorch_settings = None

        if len(rows) == 1:
            self._constant_value = float(targets[0])
            return

        try:
            import torch
            import gpytorch
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from gpytorch.mlls import ExactMarginalLogLikelihood
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "BoTorch backend requires the optional `botorch` extra "
                "(which also installs torch/gpytorch)."
            ) from exc

        self._torch = torch
        self._gpytorch_settings = gpytorch.settings

        train_x = torch.tensor(rows, dtype=torch.double)
        train_y = torch.tensor(targets, dtype=torch.double).unsqueeze(-1)
        min_vals = train_x.amin(dim=0)
        max_vals = train_x.amax(dim=0)
        range_vals = (max_vals - min_vals).clamp_min(1e-6)
        scaled_x = (train_x - min_vals) / range_vals

        target_mean = train_y.mean()
        target_std = train_y.std()
        if float(target_std.item()) < 1e-6:
            target_std = torch.tensor(1.0, dtype=torch.double)
        normalized_y = (train_y - target_mean) / target_std

        model = SingleTaskGP(scaled_x, normalized_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        model.likelihood.eval()

        self._model = model
        self._min_vals = min_vals
        self._range_vals = range_vals
        self._target_mean = float(target_mean.item())
        self._target_std = float(target_std.item())

    def predict(self, row: Sequence[float]) -> Tuple[float, float]:
        if self._constant_value is not None:
            return self._constant_value, self._constant_std

        torch = self._torch
        assert torch is not None and self._model is not None
        query_x = torch.tensor([list(map(float, row))], dtype=torch.double)
        scaled_x = (query_x - self._min_vals) / self._range_vals

        with torch.no_grad(), self._gpytorch_settings.fast_pred_var():
            posterior = self._model.posterior(scaled_x)
        mean = float(posterior.mean.squeeze().item()) * self._target_std + self._target_mean
        variance = float(posterior.variance.squeeze().item()) * (self._target_std ** 2)
        return mean, max(0.0, variance) ** 0.5


@dataclass
class _BoTorchFittedSurrogate(BaseFittedSurrogate):
    time_regressor: _BoTorchRegressor
    loss_regressor: _BoTorchRegressor
    time_feature_fn: TimeFeatureFn
    loss_feature_fn: LossFeatureFn

    def predict_training_cost(self, step: int, n_value: int, dataset_size: int) -> Tuple[float, float]:
        return self.time_regressor.predict(self.time_feature_fn(step, n_value, dataset_size))

    def predict_loss(self, step: int, n_value: int, dataset_size: int, eval_step: int) -> Tuple[float, float]:
        return self.loss_regressor.predict(self.loss_feature_fn(step, n_value, dataset_size, eval_step))


class BoTorchBayesBackend(BaseBayesBackend):
    """Optional GP-based backend implemented entirely inside RetrainOpt."""

    name = "botorch"

    def fit(
        self,
        evaluations: Iterable[CandidateEvaluation],
        *,
        config: PlannerConfig,
        time_feature_fn: TimeFeatureFn,
        loss_feature_fn: LossFeatureFn,
    ) -> BaseFittedSurrogate:
        time_rows: List[Tuple[float, ...]] = []
        time_targets: List[float] = []
        loss_rows: List[Tuple[float, ...]] = []
        loss_targets: List[float] = []

        for evaluation in evaluations:
            step, n_value = evaluation.model_id
            size = max(1, evaluation.dataset_size)
            time_rows.append(tuple(map(float, time_feature_fn(step, n_value, size))))
            time_targets.append(float(evaluation.training_cost))
            for eval_step, loss_value in evaluation.future_losses.items():
                loss_rows.append(tuple(map(float, loss_feature_fn(step, n_value, size, eval_step))))
                loss_targets.append(float(loss_value))

        if not time_rows or not loss_rows:
            raise RuntimeError("BoTorchBayesBackend requires observed evaluations before fitting.")

        return _BoTorchFittedSurrogate(
            time_regressor=_BoTorchRegressor(time_rows, time_targets),
            loss_regressor=_BoTorchRegressor(loss_rows, loss_targets),
            time_feature_fn=time_feature_fn,
            loss_feature_fn=loss_feature_fn,
        )


def create_bayes_backend(name: str, *, config: PlannerConfig) -> BaseBayesBackend:
    """Instantiate a bayes surrogate backend by name."""
    normalized = (name or "lightweight").strip().lower()
    registry: Dict[str, BaseBayesBackend] = {
        "lightweight": LightweightBayesBackend(),
        "botorch": BoTorchBayesBackend(),
    }
    if normalized not in registry:
        raise ValueError(f"Unknown bayes backend: {name}")
    return registry[normalized]
