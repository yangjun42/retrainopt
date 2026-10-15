"""Candidate evaluation engine."""

from __future__ import annotations

import time
import math
from numbers import Integral, Real
from typing import Any, Dict, Iterable, Optional

from .config import PlannerConfig
from .core.costs import compute_training_cost
from .interfaces import ModelAdapter, TimelineDataset
from .types import CandidateEvaluation, ModelId


def _nonnegative_measurement(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number; received {value!r}.")
    return float(value)


class EvaluationEngine:
    """Train and evaluate candidates with an in-memory result cache."""

    def __init__(self, timeline: TimelineDataset, model_adapter: ModelAdapter, config: PlannerConfig):
        if isinstance(timeline.num_steps, bool) or not isinstance(timeline.num_steps, Integral) or timeline.num_steps <= 0:
            raise ValueError("TimelineDataset must contain at least one timestep.")
        self.timeline = timeline
        self.model_adapter = model_adapter
        self.config = config
        self._candidate_cache: Dict[ModelId, CandidateEvaluation] = {}
        self._initial_cache: Optional[CandidateEvaluation] = None

    def evaluate_initial_model(self, initial_model: Any | None = None) -> CandidateEvaluation:
        """Evaluate the initial deployed model.

        Use a supplied model or the configured constant ``initial_loss``.
        If neither is supplied, train on the first timestep using
        ``initial_model_samples`` or the full first train block.
        """
        if self.config.initial_loss is not None:
            if initial_model is not None:
                raise ValueError("initial_loss and an explicit initial_model are mutually exclusive.")
            evaluation = CandidateEvaluation(
                model_id=(-1, 0), dataset_size=0, training_seconds=0.0, training_cost=0.0,
                future_losses={step: float(self.config.initial_loss) for step in range(self.timeline.num_steps)},
                observed=True, metadata={"source": "configured_initial_loss"},
            )
            # Validate evaluation blocks even when the loss baseline is configured.
            for step in range(self.timeline.num_steps):
                self._eval_block(step)
            self._initial_cache = evaluation
            return evaluation
        if self._initial_cache is not None and initial_model is None:
            return self._initial_cache

        if initial_model is None:
            if self.timeline.num_steps <= 0:
                raise ValueError("TimelineDataset has no steps; cannot synthesize the initial model.")
            initial_size = self.timeline.train_size(0)
            if isinstance(initial_size, bool) or not isinstance(initial_size, Integral) or initial_size < 0:
                raise ValueError("train_size(0) must be a nonnegative integer.")
            if initial_size <= 0:
                raise ValueError(
                    "No initial model was provided and timestep 0 has no training data. "
                    "Set PlannerConfig.initial_loss or pass an initial model artifact."
                )

            requested_n = self.config.initial_model_samples or initial_size
            train_data = self.timeline.sample_train_block(0, requested_n, seed=self.config.random_seed)
            initial_model = self.model_adapter.fit(
                train_data,
                step=0,
                n_samples=min(requested_n, initial_size),
                random_seed=self.config.random_seed,
            )
            source = "trained_on_step0"
        else:
            source = "provided_initial_model"

        losses = {}
        for step in range(self.timeline.num_steps):
            eval_data = self._eval_block(step)
            losses[step] = _nonnegative_measurement(
                self.model_adapter.evaluate(initial_model, eval_data, step=step), f"Initial model loss at step {step}"
            )

        evaluation = CandidateEvaluation(
            model_id=(-1, 0),
            dataset_size=self.timeline.train_size(0) if self.timeline.num_steps > 0 else 0,
            training_seconds=0.0,
            training_cost=0.0,
            future_losses=losses,
            observed=True,
            metadata={"source": source},
        )
        if source == "trained_on_step0":
            self._initial_cache = evaluation
        return evaluation

    def evaluate_candidate(self, step: int, n_samples: int) -> CandidateEvaluation:
        """Evaluate a concrete ``(step, n_samples)`` candidate."""
        return self._evaluate_candidate(step, n_samples, cache=True)

    def evaluate_candidate_uncached(self, step: int, n_samples: int) -> CandidateEvaluation:
        """Evaluate a candidate for diagnostics without adding it to the observed cache."""
        return self._evaluate_candidate(step, n_samples, cache=False)

    def _evaluate_candidate(self, step: int, n_samples: int, *, cache: bool) -> CandidateEvaluation:
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0 or step >= self.timeline.num_steps:
            raise IndexError(f"Candidate timestep out of range: {step}")
        if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
            raise ValueError("Candidate n_samples must be a positive integer.")

        dataset_size = self.timeline.train_size(step)
        if isinstance(dataset_size, bool) or not isinstance(dataset_size, Integral) or dataset_size < 0:
            raise ValueError(f"train_size({step}) must be a nonnegative integer.")
        actual_n = min(max(0, int(n_samples)), dataset_size)
        if actual_n < self.config.min_train_size:
            raise ValueError(f"Candidate at step {step} has fewer than min_train_size samples.")
        model_id = (step, actual_n)
        if model_id in self._candidate_cache:
            return self._candidate_cache[model_id]

        sampled_train = self.timeline.sample_train_block(
            step,
            actual_n,
            seed=self.config.random_seed + step * 1009 + actual_n,
        )

        start_time = time.perf_counter()
        model = self.model_adapter.fit(
            sampled_train,
            step=step,
            n_samples=actual_n,
            random_seed=self.config.random_seed + step * 1009 + actual_n,
        )
        training_seconds = _nonnegative_measurement(time.perf_counter() - start_time, "Measured training time")
        training_cost = _nonnegative_measurement(
            compute_training_cost(training_seconds, self.config.time_cost_weight), "Measured training cost"
        )

        future_losses = {}
        for eval_step in range(step, self.timeline.num_steps):
            eval_block = self._eval_block(eval_step)
            future_losses[eval_step] = _nonnegative_measurement(
                self.model_adapter.evaluate(model, eval_block, step=eval_step), f"Candidate {model_id} loss at step {eval_step}"
            )

        evaluation = CandidateEvaluation(
            model_id=model_id,
            dataset_size=dataset_size,
            training_seconds=training_seconds,
            training_cost=training_cost,
            future_losses=future_losses,
            observed=True,
            metadata={"requested_n": int(n_samples)},
        )
        if cache:
            self._candidate_cache[model_id] = evaluation
        return evaluation

    def iter_cached_candidates(self) -> Iterable[CandidateEvaluation]:
        return self._candidate_cache.values()

    def _eval_block(self, step: int) -> Any:
        block = self.timeline.get_eval_block(step)
        if hasattr(block, "__len__") and len(block) == 0:
            raise ValueError(f"Evaluation block at step {step} must not be empty.")
        return block
