"""Candidate evaluation engine."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional

from .config import PlannerConfig
from .core.costs import compute_training_cost
from .interfaces import ModelAdapter, TimelineDataset
from .types import CandidateEvaluation, ModelId


class EvaluationEngine:
    """Train and evaluate candidate retraining choices without disk side effects."""

    def __init__(self, timeline: TimelineDataset, model_adapter: ModelAdapter, config: PlannerConfig):
        self.timeline = timeline
        self.model_adapter = model_adapter
        self.config = config
        self._candidate_cache: Dict[ModelId, CandidateEvaluation] = {}
        self._initial_cache: Optional[CandidateEvaluation] = None

    def evaluate_initial_model(self, initial_model: Any | None = None) -> CandidateEvaluation:
        """Evaluate the initial deployed model.

        If no model artifact is supplied, RetrainOpt synthesizes the initial
        model by training on the first timestep using either the configured
        ``initial_model_samples`` or the full first train block.
        """
        if self._initial_cache is not None and initial_model is None:
            return self._initial_cache

        if initial_model is None:
            if self.timeline.num_steps <= 0:
                raise ValueError("TimelineDataset has no steps; cannot synthesize the initial model.")
            if self.timeline.train_size(0) <= 0:
                if self.config.initial_loss is None:
                    raise ValueError(
                        "No initial model was provided and timestep 0 has no training data. "
                        "Set PlannerConfig.initial_loss or pass an initial model artifact."
                    )
                evaluation = CandidateEvaluation(
                    model_id=(-1, 0),
                    dataset_size=0,
                    training_seconds=0.0,
                    training_cost=0.0,
                    future_losses={step: float(self.config.initial_loss) for step in range(self.timeline.num_steps)},
                    observed=True,
                    metadata={"source": "configured_initial_loss"},
                )
                self._initial_cache = evaluation
                return evaluation

            requested_n = self.config.initial_model_samples or self.timeline.train_size(0)
            train_data = self.timeline.sample_train_block(0, requested_n, seed=self.config.random_seed)
            initial_model = self.model_adapter.fit(
                train_data,
                step=0,
                n_samples=min(requested_n, self.timeline.train_size(0)),
                random_seed=self.config.random_seed,
            )
            source = "trained_on_step0"
        else:
            source = "provided_initial_model"

        losses = {}
        for step in range(self.timeline.num_steps):
            eval_data = self.timeline.get_eval_block(step)
            losses[step] = float(self.model_adapter.evaluate(initial_model, eval_data, step=step))

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
        if step < 0 or step >= self.timeline.num_steps:
            raise IndexError(f"Candidate timestep out of range: {step}")

        dataset_size = self.timeline.train_size(step)
        actual_n = min(max(0, int(n_samples)), dataset_size)
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
        training_seconds = time.perf_counter() - start_time
        training_cost = compute_training_cost(training_seconds, self.config.time_cost_weight)

        future_losses = {}
        for eval_step in range(step, self.timeline.num_steps):
            eval_block = self.timeline.get_eval_block(eval_step)
            future_losses[eval_step] = float(self.model_adapter.evaluate(model, eval_block, step=eval_step))

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
