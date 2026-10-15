"""Abstract interfaces exposed by RetrainOpt."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from .types import CandidateEvaluation


class TimelineDataset(ABC):
    """Time-sliced train/eval dataset interface."""

    @property
    @abstractmethod
    def num_steps(self) -> int:
        """Return the number of logical timesteps."""

    @abstractmethod
    def get_train_block(self, step: int) -> Any:
        """Return the full train block for ``step``."""

    @abstractmethod
    def get_eval_block(self, step: int) -> Any:
        """Return the full eval block for ``step``."""

    @abstractmethod
    def train_size(self, step: int) -> int:
        """Return the train-set size for ``step``."""

    def sample_train_block(self, step: int, n_samples: int, *, seed: int) -> Any:
        """Return a sampled train block for candidate evaluation."""
        full_block = self.get_train_block(step)
        block_size = self.train_size(step)
        actual_n = min(max(0, int(n_samples)), block_size)
        if actual_n >= block_size:
            return full_block
        if actual_n <= 0:
            return []
        if isinstance(full_block, Sequence):
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(block_size), actual_n))
            return [full_block[idx] for idx in indices]
        raise TypeError(
            "TimelineDataset.sample_train_block requires a Sequence block or "
            "an override on the concrete adapter."
        )


class ModelAdapter(ABC):
    """Model fitting/evaluation abstraction."""

    @abstractmethod
    def fit(self, train_data: Any, *, step: int, n_samples: int, random_seed: int) -> Any:
        """Fit a candidate model and return a model artifact."""

    @abstractmethod
    def evaluate(self, model: Any, eval_data: Any, *, step: int) -> float:
        """Evaluate a model artifact and return a scalar loss."""


class TrainingCostFunction(Protocol):
    """Price a completed fit in objective units.

    Return a finite nonnegative cost. This replaces the default weighted
    fit time; ``time_cost_weight`` is not applied again. ``n_samples`` is the
    actual requested fit size after clipping to available data. ``model`` is
    the artifact returned by ``fit`` and may carry resource measurements.

    Do not mutate the artifact or depend on earlier model choices. A failed
    step may retry the fit and this callback. KEEP actions and initial-model
    setup do not call it; cached candidates reuse their recorded cost.
    """

    def __call__(self, *, step: int, n_samples: int, training_seconds: float, model: Any) -> float: ...


@runtime_checkable
class EvaluationProvider(Protocol):
    """Cost evaluation seam for custom measurement or persistent caching.

    Returned costs use the planner's frozen configuration and immutable
    timeline. Candidate ids and complete finite losses must describe that
    same snapshot. Diagnostic evaluation must not add observed candidates.
    """

    def evaluate_initial_model(self, initial_model: Any | None = None) -> CandidateEvaluation: ...

    def evaluate_candidate(self, step: int, n_samples: int) -> CandidateEvaluation: ...

    def evaluate_candidate_uncached(self, step: int, n_samples: int) -> CandidateEvaluation: ...

    def iter_cached_candidates(self) -> Iterable[CandidateEvaluation]: ...

