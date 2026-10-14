"""Public planner facade."""

from __future__ import annotations

import copy
import math
import random
from numbers import Integral, Real
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .config import PlannerConfig
from .core import AcquisitionManager, create_bayes_backend, evaluate_decision_sequence, solve_exact_schedule
from .core.dp import StageCosts, solve_stage_schedule, validate_model_costs
from .evaluation import EvaluationEngine
from .interfaces import EvaluationProvider, ModelAdapter, TimelineDataset
from .types import ALGORITHM_VERSION, CandidateEvaluation, Decision, DecisionAction, METADATA_SCHEMA_VERSION, ModelId, PlanResult


class RetrainingPlanner:
    """Cost-aware retraining planner for time-sliced datasets."""

    def __init__(
        self,
        timeline: TimelineDataset,
        model_adapter: ModelAdapter,
        config: PlannerConfig | None = None,
        *,
        evaluator: EvaluationProvider | None = None,
    ):
        self.timeline = timeline
        self.model_adapter = model_adapter
        self.config = copy.deepcopy(config) if config is not None else PlannerConfig()
        if (
            isinstance(timeline.num_steps, bool)
            or not isinstance(timeline.num_steps, Integral)
            or timeline.num_steps <= 0
        ):
            raise ValueError("TimelineDataset.num_steps must be a positive integer.")
        if evaluator is not None and not isinstance(evaluator, EvaluationProvider):
            raise TypeError("evaluator must implement the EvaluationProvider protocol.")
        self.engine = evaluator if evaluator is not None else EvaluationEngine(timeline, model_adapter, self.config)
        self.acquisition = AcquisitionManager()
        self.acquisition.parse_name(self.config.acquisition)
        self.bayes_backend = create_bayes_backend(self.config.bayes_backend, config=self.config)

    def plan(self, method: str = "exact", *, initial_model: Any | None = None) -> PlanResult:
        """Plan a retraining schedule.

        Supported methods:

        - ``exact``: exact dynamic programming over all configured candidates.
        - ``bayes``: surrogate-guided candidate evaluation followed by exact DP
          on an estimated full cost matrix.
        """
        if not isinstance(method, str):
            raise ValueError("method must be 'exact' or 'bayes'.")
        normalized = method.strip().lower()
        if normalized == "exact":
            return self._plan_exact(initial_model=initial_model)
        if normalized == "bayes":
            return self._plan_bayes(initial_model=initial_model)
        raise ValueError(f"Unsupported planning method: {method}")

    def _plan_exact(self, *, initial_model: Any | None = None) -> PlanResult:
        state_costs = self._build_exact_state_costs(initial_model=initial_model)
        total_cost, decisions, _, _ = solve_exact_schedule(
            self.timeline.num_steps,
            state_costs,
            loss_cost_weight=self.config.loss_cost_weight,
        )
        total_cost, training_cost, loss_cost = evaluate_decision_sequence(
            decisions,
            state_costs,
            loss_cost_weight=self.config.loss_cost_weight,
        )
        return PlanResult(
            method="exact",
            decisions=decisions,
            total_cost=total_cost,
            training_cost=training_cost,
            loss_cost=loss_cost,
            weighted_loss_cost=self.config.loss_cost_weight * loss_cost,
            estimated_total_cost=total_cost,
            metadata={
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
                "algorithm_version": ALGORITHM_VERSION,
                "initial_model_source": state_costs[(-1, 0)].get("metadata", {}).get("source", "provider_supplied"),
                "candidate_count": len(state_costs) - 1,
                "timeline_steps": self.timeline.num_steps,
            },
        )

    def _plan_bayes(self, *, initial_model: Any | None = None) -> PlanResult:
        initial_eval = self._evaluate_initial_model(initial_model)
        all_candidates = self._all_candidate_ids()
        if not all_candidates:
            return self._plan_keep_only(initial_eval)
        selected = self._seed_bayes_candidates(all_candidates)
        for model_id in selected:
            self._evaluate_candidate(*model_id)

        bayes_selection_strategy = self._normalized_bayes_selection_strategy()
        bayes_iterations_completed = 0
        bayes_estimated_total_history: List[float] = []
        bayes_estimated_decisions_history: List[List[Decision]] = []
        bayes_new_candidates_by_iteration: List[List[ModelId]] = []
        bayes_retrain_steps_by_iteration: List[List[int]] = []
        bayes_stopped_early = False
        for _ in range(max(0, self.config.bayes_iterations)):
            observed = self._observed_candidates()
            if not observed:
                bayes_stopped_early = True
                break

            observed_ids = {evaluation.model_id for evaluation in observed}
            if observed_ids.issuperset(all_candidates):
                bayes_stopped_early = True
                break
            surrogate = self._fit_surrogates(observed)

            if bayes_selection_strategy == "sequence":
                acquisition_state_costs = self._build_acquisition_state_costs(
                    initial_eval,
                    all_candidates,
                    observed,
                    surrogate,
                )
                _, acquisition_decisions, _, _ = solve_stage_schedule(
                    self.timeline.num_steps,
                    acquisition_state_costs,
                )
                next_ids = self._new_retrain_candidates_from_sequence(acquisition_decisions, observed_ids)
                if not next_ids:
                    next_ids = self._top_scored_unobserved_candidates(all_candidates, observed, surrogate, initial_eval=initial_eval)
                else:
                    next_ids = self._supplement_to_minimum_new_candidates(
                        next_ids,
                        all_candidates,
                        observed,
                        surrogate,
                        initial_eval=initial_eval,
                    )
            else:
                next_ids = self._top_scored_unobserved_candidates(all_candidates, observed, surrogate, initial_eval=initial_eval)

            if not next_ids:
                bayes_stopped_early = True
                break

            for model_id in next_ids:
                self._evaluate_candidate(*model_id)

            bayes_iterations_completed += 1
            bayes_new_candidates_by_iteration.append(list(next_ids))
            iteration_observed = self._observed_candidates()
            iteration_surrogate = self._fit_surrogates(iteration_observed) if len(iteration_observed) < len(all_candidates) else None
            iteration_state_costs = self._build_estimated_state_costs(
                initial_eval,
                all_candidates,
                iteration_observed,
                iteration_surrogate,
            )
            iteration_total, iteration_decisions, _, _ = solve_exact_schedule(
                self.timeline.num_steps,
                iteration_state_costs,
                loss_cost_weight=self.config.loss_cost_weight,
            )
            bayes_estimated_total_history.append(float(iteration_total))
            bayes_estimated_decisions_history.append(list(iteration_decisions))
            bayes_retrain_steps_by_iteration.append(
                [
                    step
                    for step, decision in enumerate(iteration_decisions)
                    if decision.action == DecisionAction.RETRAIN
                ]
            )

        observed = self._observed_candidates()
        if not observed:
            raise RuntimeError("Bayes planner did not evaluate any retraining candidates.")

        surrogate = self._fit_surrogates(observed) if len(observed) < len(all_candidates) else None
        estimated_state_costs = self._build_estimated_state_costs(
            initial_eval,
            all_candidates,
            observed,
            surrogate,
        )

        estimated_total, decisions, _, _ = solve_exact_schedule(
            self.timeline.num_steps,
            estimated_state_costs,
            loss_cost_weight=self.config.loss_cost_weight,
        )
        realized_state_costs = {(-1, 0): initial_eval.to_state_cost()}
        for step, decision in enumerate(decisions):
            if decision.action != DecisionAction.RETRAIN:
                continue
            model_id = (step, int(decision.n_samples or 0))
            realized_state_costs[model_id] = self._evaluate_candidate(*model_id).to_state_cost()
        total_cost, training_cost, loss_cost = evaluate_decision_sequence(
            decisions,
            realized_state_costs,
            loss_cost_weight=self.config.loss_cost_weight,
        )
        total_evaluated_candidates = len(self._observed_candidates())
        bayes_true_total_history = (
            self._exact_total_history_for_sequences(
                initial_eval,
                bayes_estimated_decisions_history,
            )
            if self.config.record_bayes_true_history
            else []
        )
        return PlanResult(
            method="bayes",
            decisions=decisions,
            total_cost=total_cost,
            training_cost=training_cost,
            loss_cost=loss_cost,
            weighted_loss_cost=self.config.loss_cost_weight * loss_cost,
            estimated_total_cost=estimated_total,
            metadata={
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
                "algorithm_version": ALGORITHM_VERSION,
                "initial_model_source": initial_eval.metadata.get("source", "provider_supplied"),
                "acquisition_semantics": "complete_stage_independent_posteriors",
                "candidate_count": len(all_candidates),
                "evaluated_candidates": total_evaluated_candidates,
                "surrogate_evaluated_candidates": len(observed),
                "timeline_steps": self.timeline.num_steps,
                "acquisition": self.config.acquisition,
                "bayes_backend": self.config.bayes_backend,
                "bayes_selection_strategy": bayes_selection_strategy,
                "selected_sequence_evaluated_exactly": True,
                "bayes_iterations_completed": bayes_iterations_completed,
                "bayes_stopped_early": bayes_stopped_early,
                "bayes_initial_candidates": list(selected),
                "bayes_new_candidates_by_iteration": bayes_new_candidates_by_iteration,
                "bayes_retrain_steps_by_iteration": bayes_retrain_steps_by_iteration,
                "bayes_estimated_decisions_history": [
                    self._serialize_decisions(iteration_decisions)
                    for iteration_decisions in bayes_estimated_decisions_history
                ],
                "bayes_estimated_total_history": bayes_estimated_total_history,
                "bayes_true_total_history": bayes_true_total_history,
                "record_bayes_true_history": bool(self.config.record_bayes_true_history),
            },
        )

    def _build_exact_state_costs(self, *, initial_model: Any | None = None) -> Dict[ModelId, Dict[str, Any]]:
        state_costs = {(-1, 0): self._evaluate_initial_model(initial_model).to_state_cost()}
        for step, n_value in self._all_candidate_ids():
            state_costs[(step, n_value)] = self._evaluate_candidate(step, n_value).to_state_cost()
        return state_costs

    def _validate_evaluation(self, evaluation: CandidateEvaluation, expected_id: ModelId) -> CandidateEvaluation:
        if getattr(evaluation, "model_id", None) != expected_id:
            raise ValueError(f"EvaluationProvider returned the wrong model_id; expected {expected_id!r}.")
        validate_model_costs(evaluation.model_id, evaluation.to_state_cost(), self.timeline.num_steps)
        seconds = evaluation.training_seconds
        if isinstance(seconds, bool) or not isinstance(seconds, Real) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"EvaluationProvider training_seconds for {expected_id!r} must be finite and nonnegative.")
        size = evaluation.dataset_size
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
            raise ValueError(f"EvaluationProvider dataset_size for {expected_id!r} must be a nonnegative integer.")
        if expected_id != (-1, 0) and size != self.timeline.train_size(expected_id[0]):
            raise ValueError(f"EvaluationProvider dataset_size for {expected_id!r} disagrees with the timeline.")
        if evaluation.observed is not True:
            raise ValueError(f"EvaluationProvider must return observed costs for {expected_id!r}.")
        return evaluation

    def _evaluate_initial_model(self, initial_model: Any | None) -> CandidateEvaluation:
        if initial_model is not None and self.config.initial_loss is not None:
            raise ValueError("initial_loss and an explicit initial_model are mutually exclusive.")
        evaluation = self._validate_evaluation(self.engine.evaluate_initial_model(initial_model=initial_model), (-1, 0))
        if self.config.initial_loss is not None and any(
            loss != self.config.initial_loss for loss in evaluation.future_losses.values()
        ):
            raise ValueError("EvaluationProvider initial losses disagree with configured initial_loss.")
        return evaluation

    def _evaluate_candidate(self, step: int, n_samples: int, *, uncached: bool = False) -> CandidateEvaluation:
        method = self.engine.evaluate_candidate_uncached if uncached else self.engine.evaluate_candidate
        return self._validate_evaluation(method(step, n_samples), (step, n_samples))

    def _observed_candidates(self) -> List[CandidateEvaluation]:
        allowed = set(self._all_candidate_ids())
        seen = set()
        evaluations = []
        for evaluation in self.engine.iter_cached_candidates():
            model_id = evaluation.model_id
            if model_id not in allowed or model_id in seen:
                raise ValueError(f"EvaluationProvider cache contains an unexpected or duplicate candidate: {model_id!r}.")
            evaluations.append(self._validate_evaluation(evaluation, model_id))
            seen.add(model_id)
        return evaluations

    def _plan_keep_only(self, initial_eval: CandidateEvaluation) -> PlanResult:
        decisions = [Decision(DecisionAction.KEEP) for _ in range(self.timeline.num_steps)]
        total, training, loss = evaluate_decision_sequence(
            decisions, {(-1, 0): initial_eval.to_state_cost()}, loss_cost_weight=self.config.loss_cost_weight,
        )
        return PlanResult(
            method="bayes", decisions=decisions, total_cost=total, training_cost=training,
            loss_cost=loss, weighted_loss_cost=self.config.loss_cost_weight * loss, estimated_total_cost=total,
            metadata={
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
                "algorithm_version": ALGORITHM_VERSION,
                "initial_model_source": initial_eval.metadata.get("source", "provider_supplied"),
                "acquisition_semantics": "complete_stage_independent_posteriors",
                "candidate_count": 0, "evaluated_candidates": 0, "surrogate_evaluated_candidates": 0,
                "timeline_steps": self.timeline.num_steps, "acquisition": self.config.acquisition,
                "bayes_backend": self.config.bayes_backend,
                "bayes_selection_strategy": self._normalized_bayes_selection_strategy(),
                "selected_sequence_evaluated_exactly": True,
                "bayes_iterations_completed": 0, "bayes_stopped_early": True,
                "bayes_initial_candidates": [], "bayes_new_candidates_by_iteration": [],
                "bayes_retrain_steps_by_iteration": [], "bayes_estimated_decisions_history": [],
                "bayes_estimated_total_history": [], "bayes_true_total_history": [],
                "record_bayes_true_history": self.config.record_bayes_true_history,
            },
        )

    def _exact_total_history_for_sequences(
        self,
        initial_eval: CandidateEvaluation,
        decision_history: Sequence[Sequence[Decision]],
    ) -> List[float]:
        true_totals: List[float] = []
        for decisions in decision_history:
            realized_state_costs = {(-1, 0): initial_eval.to_state_cost()}
            for step, decision in enumerate(decisions):
                if decision.action != DecisionAction.RETRAIN:
                    continue
                model_id = (step, int(decision.n_samples or 0))
                realized_state_costs[model_id] = self._evaluate_candidate(*model_id, uncached=True).to_state_cost()
            total_cost, _, _ = evaluate_decision_sequence(
                decisions,
                realized_state_costs,
                loss_cost_weight=self.config.loss_cost_weight,
            )
            true_totals.append(float(total_cost))
        return true_totals

    @staticmethod
    def _serialize_decisions(decisions: Sequence[Decision]) -> List[Dict[str, Any]]:
        return [
            {
                "action": decision.action.value,
                "n_samples": decision.n_samples,
                "trained_at": decision.trained_at,
            }
            for decision in decisions
        ]

    def _all_candidate_ids(self) -> List[ModelId]:
        candidates: List[ModelId] = []
        for step in range(self.timeline.num_steps):
            train_size = self.timeline.train_size(step)
            if isinstance(train_size, bool) or not isinstance(train_size, Integral) or train_size < 0:
                raise ValueError(f"TimelineDataset.train_size({step}) must be a nonnegative integer.")
            for n_value in self.config.resolve_n_choices(train_size):
                candidates.append((step, n_value))
        return candidates

    def _seed_bayes_candidates(self, all_candidates: Sequence[ModelId]) -> List[ModelId]:
        if not all_candidates:
            return []

        seed_models: List[ModelId] = []
        seen = set()
        for ratio in self.config.bayes_anchor_steps:
            clamped = min(1.0, max(0.0, float(ratio)))
            step = int(round(clamped * max(0, self.timeline.num_steps - 1)))
            for model_id in [candidate for candidate in all_candidates if candidate[0] == step]:
                if model_id not in seen:
                    seed_models.append(model_id)
                    seen.add(model_id)

        remaining = [candidate for candidate in all_candidates if candidate not in seen]
        rng = random.Random(self.config.random_seed)
        rng.shuffle(remaining)
        target = max(len(seed_models), min(len(all_candidates), self.config.bayes_initial_samples))
        for model_id in remaining:
            if len(seed_models) >= target:
                break
            seed_models.append(model_id)
            seen.add(model_id)
        return seed_models

    def _fit_surrogates(self, evaluations: Iterable[CandidateEvaluation]):
        return self.bayes_backend.fit(
            evaluations,
            config=self.config,
            time_feature_fn=self._time_features,
            loss_feature_fn=self._loss_features,
        )

    def _build_estimated_state_costs(
        self,
        initial_eval: CandidateEvaluation,
        all_candidates: Sequence[ModelId],
        observed: Sequence[CandidateEvaluation],
        surrogate,
    ) -> Dict[ModelId, Dict[str, Any]]:
        estimated_state_costs = {(-1, 0): initial_eval.to_state_cost()}
        observed_map = {evaluation.model_id: evaluation for evaluation in observed}

        for step, n_value in all_candidates:
            dataset_size = self.timeline.train_size(step)
            if (step, n_value) in observed_map:
                estimated_state_costs[(step, n_value)] = observed_map[(step, n_value)].to_state_cost()
                continue

            training_mean, _ = self._predict_training_moments(surrogate, step, n_value, dataset_size)
            future_losses = {}
            for eval_step, loss_mean, _ in self._predict_future_losses(
                surrogate,
                step,
                n_value,
                dataset_size,
            ):
                future_losses[eval_step] = loss_mean
            evaluation = CandidateEvaluation(
                model_id=(step, n_value),
                dataset_size=dataset_size,
                training_seconds=training_mean / self.config.time_cost_weight if self.config.time_cost_weight else 0.0,
                training_cost=training_mean,
                future_losses=future_losses,
                observed=False,
            )
            estimated_state_costs[(step, n_value)] = evaluation.to_state_cost()
        return estimated_state_costs

    def _build_acquisition_state_costs(
        self,
        initial_eval: CandidateEvaluation,
        all_candidates: Sequence[ModelId],
        observed: Sequence[CandidateEvaluation],
        surrogate,
    ) -> StageCosts:
        """Apply acquisition once to each complete weighted stage cost.

        Separate training/loss surrogates use independent posteriors. The
        sequence objective sums stage scores, not a joint schedule posterior.
        """
        weight = self.config.loss_cost_weight
        best = self._best_observed_stage_cost(initial_eval, observed)

        def score(mean: float, std: float) -> float:
            return self.acquisition.score(
                mean, std, self.config.acquisition, beta=self.config.exploration_beta,
                best_cost=best, xi=self.config.improvement_xi,
            )

        keep = {(-1, 0): {
            step: score(weight * loss, 0.0) for step, loss in initial_eval.future_losses.items()
        }}
        retrain = {}
        observed_map = {evaluation.model_id: evaluation for evaluation in observed}
        for model_id in all_candidates:
            step, n_value = model_id
            if model_id in observed_map:
                evaluation = observed_map[model_id]
                retrain[model_id] = score(
                    evaluation.training_cost + weight * evaluation.future_losses[step], 0.0,
                )
                keep[model_id] = {
                    future: score(weight * loss, 0.0)
                    for future, loss in evaluation.future_losses.items() if future > step
                }
                continue
            size = self.timeline.train_size(step)
            training_mean, training_std = self._predict_training_moments(surrogate, step, n_value, size)
            keep[model_id] = {}
            for future, loss_mean, loss_std in self._predict_future_losses(surrogate, step, n_value, size):
                if future == step:
                    retrain[model_id] = score(
                        training_mean + weight * loss_mean,
                        math.hypot(training_std, weight * loss_std),
                    )
                else:
                    keep[model_id][future] = score(weight * loss_mean, weight * loss_std)
        return StageCosts(keep=keep, retrain=retrain)

    @staticmethod
    def _valid_moments(prediction, context: str) -> Tuple[float, float]:
        try:
            mean, std = prediction
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Surrogate {context} must return (mean, std).") from exc
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in (mean, std)):
            raise ValueError(f"Surrogate {context} must return finite real moments.")
        if std < 0:
            raise ValueError(f"Surrogate {context} returned negative standard deviation.")
        return max(0.0, float(mean)), float(std)

    def _predict_training_moments(self, surrogate, step: int, n_value: int, dataset_size: int) -> Tuple[float, float]:
        moments = self._valid_moments(
            surrogate.predict_training_cost(step, n_value, dataset_size), f"training cost for {(step, n_value)!r}",
        )
        return moments if self.config.time_cost_weight else (0.0, 0.0)

    def _predict_future_losses(
        self,
        surrogate,
        step: int,
        n_value: int,
        dataset_size: int,
    ) -> List[Tuple[int, float, float]]:
        eval_steps = list(range(step, self.timeline.num_steps))
        batch_predict = getattr(surrogate, "predict_losses", None)
        if callable(batch_predict):
            predictions = list(batch_predict(step, n_value, dataset_size, eval_steps))
        else:
            predictions = [
                surrogate.predict_loss(step, n_value, dataset_size, eval_step)
                for eval_step in eval_steps
            ]
        if len(predictions) != len(eval_steps):
            raise ValueError(f"Surrogate returned {len(predictions)} losses for {len(eval_steps)} evaluation steps.")
        return [
            (eval_step, *self._valid_moments(prediction, f"loss for {(step, n_value)!r} at {eval_step}"))
            for eval_step, prediction in zip(eval_steps, predictions)
        ]

    def _new_retrain_candidates_from_sequence(
        self,
        decisions: Sequence[Decision],
        observed_ids: set[ModelId],
    ) -> List[ModelId]:
        next_ids: List[ModelId] = []
        seen = set(observed_ids)
        for step, decision in enumerate(decisions):
            if decision.action != DecisionAction.RETRAIN:
                continue
            model_id = (step, int(decision.n_samples or 0))
            if model_id in seen:
                continue
            next_ids.append(model_id)
            seen.add(model_id)
        return next_ids

    def _top_scored_unobserved_candidates(
        self,
        all_candidates: Sequence[ModelId],
        observed: Sequence[CandidateEvaluation],
        surrogate,
        *,
        limit: int | None = None,
        initial_eval: CandidateEvaluation | None = None,
    ) -> List[ModelId]:
        observed_ids = {evaluation.model_id for evaluation in observed}
        best_observed_cost = self._best_observed_stage_cost(initial_eval, observed)
        scored_candidates = []

        for model_id in all_candidates:
            if model_id in observed_ids:
                continue
            step, n_value = model_id
            dataset_size = self.timeline.train_size(step)
            mean_cost, std_cost = self._predict_immediate_cost(
                surrogate,
                step,
                n_value,
                dataset_size,
            )
            score = self.acquisition.score(
                mean_cost,
                std_cost,
                self.config.acquisition,
                beta=self.config.exploration_beta,
                best_cost=best_observed_cost,
                xi=self.config.improvement_xi,
            )
            scored_candidates.append((score, model_id))

        scored_candidates.sort(key=lambda item: item[0])
        target = max(1, int(limit if limit is not None else self.config.bayes_candidates_per_iteration))
        return [
            model_id
            for _, model_id in scored_candidates[:target]
        ]

    def _supplement_to_minimum_new_candidates(
        self,
        selected_ids: Sequence[ModelId],
        all_candidates: Sequence[ModelId],
        observed: Sequence[CandidateEvaluation],
        surrogate,
        *,
        initial_eval: CandidateEvaluation | None = None,
    ) -> List[ModelId]:
        target = max(1, int(self.config.bayes_candidates_per_iteration))
        next_ids = list(selected_ids)
        if len(next_ids) >= target:
            return next_ids

        seen = set(next_ids)
        for model_id in self._top_scored_unobserved_candidates(
            all_candidates,
            observed,
            surrogate,
            limit=len(all_candidates),
            initial_eval=initial_eval,
        ):
            if model_id in seen:
                continue
            next_ids.append(model_id)
            seen.add(model_id)
            if len(next_ids) >= target:
                break
        return next_ids

    def _best_observed_stage_cost(
        self,
        initial_eval: CandidateEvaluation | None,
        evaluations: Iterable[CandidateEvaluation],
    ) -> float | None:
        """Minimum measured *legal* keep/retrain stage cost, in weighted units."""
        weight = self.config.loss_cost_weight
        measured = [] if initial_eval is None else [weight * loss for loss in initial_eval.future_losses.values()]
        for evaluation in evaluations:
            step = evaluation.model_id[0]
            measured.append(evaluation.training_cost + weight * evaluation.future_losses[step])
            measured.extend(weight * loss for future, loss in evaluation.future_losses.items() if future > step)
        return min(measured) if measured else None

    def _predict_immediate_cost(
        self,
        surrogate,
        step: int,
        n_value: int,
        dataset_size: int,
    ) -> Tuple[float, float]:
        training_mean, training_std = self._predict_training_moments(surrogate, step, n_value, dataset_size)
        loss_mean, loss_std = self._valid_moments(
            surrogate.predict_loss(step, n_value, dataset_size, step), f"loss for {(step, n_value)!r} at {step}",
        )
        weight = self.config.loss_cost_weight
        return training_mean + weight * loss_mean, math.hypot(training_std, weight * loss_std)

    def _normalized_bayes_selection_strategy(self) -> str:
        normalized = (self.config.bayes_selection_strategy or "sequence").strip().lower()
        if normalized in {"sequence", "schedule", "dp"}:
            return "sequence"
        if normalized in {"candidate", "topk", "top-k"}:
            return "candidate"
        raise ValueError(
            "PlannerConfig.bayes_selection_strategy must be 'sequence' or 'candidate'."
        )

    @staticmethod
    def _time_features(step: int, n_value: int, dataset_size: int) -> Tuple[float, ...]:
        safe_size = max(1, dataset_size)
        ratio = float(n_value) / float(safe_size)
        return (
            float(step),
            float(n_value),
            float(dataset_size),
            ratio,
        )

    @staticmethod
    def _loss_features(step: int, n_value: int, dataset_size: int, eval_step: int) -> Tuple[float, ...]:
        safe_size = max(1, dataset_size)
        ratio = float(n_value) / float(safe_size)
        age = max(0, eval_step - step)
        return (
            float(step),
            float(n_value),
            float(dataset_size),
            float(eval_step),
            float(age),
            ratio,
        )
