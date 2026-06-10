"""Public planner facade."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .config import PlannerConfig
from .core import AcquisitionManager, create_bayes_backend, evaluate_decision_sequence, solve_exact_schedule
from .evaluation import EvaluationEngine
from .interfaces import ModelAdapter, TimelineDataset
from .types import CandidateEvaluation, Decision, DecisionAction, ModelId, PlanResult


class RetrainingPlanner:
    """Cost-aware retraining planner for time-sliced datasets."""

    def __init__(self, timeline: TimelineDataset, model_adapter: ModelAdapter, config: PlannerConfig | None = None):
        self.timeline = timeline
        self.model_adapter = model_adapter
        self.config = config or PlannerConfig()
        self.engine = EvaluationEngine(timeline, model_adapter, self.config)
        self.acquisition = AcquisitionManager()
        self.bayes_backend = create_bayes_backend(self.config.bayes_backend, config=self.config)

    def plan(self, method: str = "exact", *, initial_model: Any | None = None) -> PlanResult:
        """Plan a retraining schedule.

        Supported methods:
        - ``exact``: exact dynamic programming over all configured candidates.
        - ``bayes``: surrogate-guided candidate evaluation followed by exact DP
          on an estimated full cost matrix.
        """
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
            estimated_total_cost=total_cost,
            metadata={
                "candidate_count": len(state_costs) - 1,
                "timeline_steps": self.timeline.num_steps,
            },
        )

    def _plan_bayes(self, *, initial_model: Any | None = None) -> PlanResult:
        initial_eval = self.engine.evaluate_initial_model(initial_model=initial_model)
        all_candidates = self._all_candidate_ids()
        selected = self._seed_bayes_candidates(all_candidates)
        for model_id in selected:
            self.engine.evaluate_candidate(*model_id)

        bayes_selection_strategy = self._normalized_bayes_selection_strategy()
        bayes_iterations_completed = 0
        bayes_estimated_total_history: List[float] = []
        bayes_estimated_decisions_history: List[List[Decision]] = []
        bayes_new_candidates_by_iteration: List[List[ModelId]] = []
        bayes_retrain_steps_by_iteration: List[List[int]] = []
        bayes_stopped_early = False
        for _ in range(max(0, self.config.bayes_iterations)):
            observed = list(self.engine.iter_cached_candidates())
            if not observed:
                bayes_stopped_early = True
                break

            surrogate = self._fit_surrogates(observed)
            observed_ids = {evaluation.model_id for evaluation in observed}
            if observed_ids.issuperset(all_candidates):
                bayes_stopped_early = True
                break

            if bayes_selection_strategy == "sequence":
                acquisition_state_costs = self._build_acquisition_state_costs(
                    initial_eval,
                    all_candidates,
                    observed,
                    surrogate,
                )
                _, acquisition_decisions, _, _ = solve_exact_schedule(
                    self.timeline.num_steps,
                    acquisition_state_costs,
                    loss_cost_weight=self.config.loss_cost_weight,
                )
                next_ids = self._new_retrain_candidates_from_sequence(acquisition_decisions, observed_ids)
                if not next_ids:
                    next_ids = self._top_scored_unobserved_candidates(all_candidates, observed, surrogate)
                else:
                    next_ids = self._supplement_to_minimum_new_candidates(
                        next_ids,
                        all_candidates,
                        observed,
                        surrogate,
                    )
            else:
                next_ids = self._top_scored_unobserved_candidates(all_candidates, observed, surrogate)

            if not next_ids:
                bayes_stopped_early = True
                break

            for model_id in next_ids:
                self.engine.evaluate_candidate(*model_id)

            bayes_iterations_completed += 1
            bayes_new_candidates_by_iteration.append(list(next_ids))
            iteration_observed = list(self.engine.iter_cached_candidates())
            iteration_surrogate = self._fit_surrogates(iteration_observed)
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

        observed = list(self.engine.iter_cached_candidates())
        if not observed:
            raise RuntimeError("Bayes planner did not evaluate any retraining candidates.")

        surrogate = self._fit_surrogates(observed)
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
            realized_state_costs[model_id] = self.engine.evaluate_candidate(*model_id).to_state_cost()
        total_cost, training_cost, loss_cost = evaluate_decision_sequence(
            decisions,
            realized_state_costs,
            loss_cost_weight=self.config.loss_cost_weight,
        )
        total_evaluated_candidates = len(list(self.engine.iter_cached_candidates()))
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
            estimated_total_cost=estimated_total,
            metadata={
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
                "bayes_new_candidates_by_iteration": bayes_new_candidates_by_iteration,
                "bayes_retrain_steps_by_iteration": bayes_retrain_steps_by_iteration,
                "bayes_estimated_total_history": bayes_estimated_total_history,
                "bayes_true_total_history": bayes_true_total_history,
                "record_bayes_true_history": bool(self.config.record_bayes_true_history),
            },
        )

    def _build_exact_state_costs(self, *, initial_model: Any | None = None) -> Dict[ModelId, Dict[str, Any]]:
        state_costs = {(-1, 0): self.engine.evaluate_initial_model(initial_model=initial_model).to_state_cost()}
        for step, n_value in self._all_candidate_ids():
            state_costs[(step, n_value)] = self.engine.evaluate_candidate(step, n_value).to_state_cost()
        return state_costs

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
                realized_state_costs[model_id] = self.engine.evaluate_candidate_uncached(*model_id).to_state_cost()
            total_cost, _, _ = evaluate_decision_sequence(
                decisions,
                realized_state_costs,
                loss_cost_weight=self.config.loss_cost_weight,
            )
            true_totals.append(float(total_cost))
        return true_totals

    def _all_candidate_ids(self) -> List[ModelId]:
        candidates: List[ModelId] = []
        for step in range(self.timeline.num_steps):
            train_size = self.timeline.train_size(step)
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

            training_mean, _ = surrogate.predict_training_cost(step, n_value, dataset_size)
            training_mean = max(0.0, float(training_mean))
            future_losses = {}
            for eval_step in range(step, self.timeline.num_steps):
                loss_mean, _ = surrogate.predict_loss(step, n_value, dataset_size, eval_step)
                future_losses[eval_step] = max(0.0, float(loss_mean))
            evaluation = CandidateEvaluation(
                model_id=(step, n_value),
                dataset_size=dataset_size,
                training_seconds=training_mean / max(self.config.time_cost_weight, 1e-12),
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
    ) -> Dict[ModelId, Dict[str, Any]]:
        acquisition_state_costs = {(-1, 0): initial_eval.to_state_cost()}
        observed_map = {evaluation.model_id: evaluation for evaluation in observed}
        best_training_cost, best_loss_cost = self._best_observed_component_costs(observed)

        for step, n_value in all_candidates:
            dataset_size = self.timeline.train_size(step)
            if (step, n_value) in observed_map:
                acquisition_state_costs[(step, n_value)] = observed_map[(step, n_value)].to_state_cost()
                continue

            training_mean, training_std = surrogate.predict_training_cost(step, n_value, dataset_size)
            training_score = self.acquisition.score(
                training_mean,
                training_std,
                self.config.acquisition,
                beta=self.config.exploration_beta,
                best_cost=best_training_cost,
                xi=self.config.improvement_xi,
            )
            future_losses = {}
            for eval_step in range(step, self.timeline.num_steps):
                loss_mean, loss_std = surrogate.predict_loss(step, n_value, dataset_size, eval_step)
                future_losses[eval_step] = self.acquisition.score(
                    loss_mean,
                    loss_std,
                    self.config.acquisition,
                    beta=self.config.exploration_beta,
                    best_cost=best_loss_cost,
                    xi=self.config.improvement_xi,
                )
            evaluation = CandidateEvaluation(
                model_id=(step, n_value),
                dataset_size=dataset_size,
                training_seconds=0.0,
                training_cost=float(training_score),
                future_losses=future_losses,
                observed=False,
            )
            acquisition_state_costs[(step, n_value)] = evaluation.to_state_cost()
        return acquisition_state_costs

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
    ) -> List[ModelId]:
        observed_ids = {evaluation.model_id for evaluation in observed}
        best_observed_cost = self._best_observed_immediate_cost(observed)
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
        ):
            if model_id in seen:
                continue
            next_ids.append(model_id)
            seen.add(model_id)
            if len(next_ids) >= target:
                break
        return next_ids

    def _best_observed_component_costs(
        self,
        evaluations: Iterable[CandidateEvaluation],
    ) -> Tuple[float | None, float | None]:
        best_training = float("inf")
        best_loss = float("inf")
        for evaluation in evaluations:
            best_training = min(best_training, float(evaluation.training_cost))
            for loss_value in evaluation.future_losses.values():
                best_loss = min(best_loss, float(loss_value))
        return (
            best_training if best_training < float("inf") else None,
            best_loss if best_loss < float("inf") else None,
        )

    def _predict_immediate_cost(
        self,
        surrogate,
        step: int,
        n_value: int,
        dataset_size: int,
    ) -> Tuple[float, float]:
        time_mean, time_std = surrogate.predict_training_cost(step, n_value, dataset_size)
        loss_mean, loss_std = surrogate.predict_loss(step, n_value, dataset_size, step)
        mean_cost = max(0.0, float(time_mean)) + self.config.loss_cost_weight * max(0.0, float(loss_mean))
        std_cost = abs(float(time_std)) + self.config.loss_cost_weight * abs(float(loss_std))
        return mean_cost, std_cost

    def _best_observed_immediate_cost(self, evaluations: Iterable[CandidateEvaluation]) -> float:
        best = float("inf")
        for evaluation in evaluations:
            step = evaluation.model_id[0]
            immediate_loss = float(evaluation.future_losses.get(step, float("inf")))
            total = evaluation.training_cost + self.config.loss_cost_weight * immediate_loss
            best = min(best, total)
        return best

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
