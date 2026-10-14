"""Verify exact DP against exhaustive schedules through public evaluator injection.

All costs are fixed. This example is independent of model fitting and timing.
"""

from __future__ import annotations

from itertools import product

from retrainopt import CandidateEvaluation, PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter


class TableEvaluator:
    """Minimal EvaluationProvider backed by a measured cost table."""

    def __init__(self) -> None:
        self.initial = CandidateEvaluation(
            model_id=(-1, 0), dataset_size=0,
            training_seconds=0.0, training_cost=0.0,
            future_losses={0: 0.8, 1: 0.9, 2: 1.0},
        )
        self.table = {}
        for step in range(3):
            for n_samples in (2, 4):
                cost = 0.1 * n_samples
                self.table[(step, n_samples)] = CandidateEvaluation(
                    model_id=(step, n_samples), dataset_size=4,
                    training_seconds=cost, training_cost=cost,
                    future_losses={
                        t: 0.4 / n_samples + 0.2 * (t - step)
                        for t in range(step, 3)
                    },
                )
        self.observed = {}

    def evaluate_initial_model(self, initial_model=None):
        if initial_model is not None:
            raise ValueError("This fixed-table example does not accept a model artifact.")
        return self.initial

    def evaluate_candidate(self, step, n_samples):
        record = self.table[(step, n_samples)]
        self.observed[record.model_id] = record
        return record

    def evaluate_candidate_uncached(self, step, n_samples):
        return self.table[(step, n_samples)]

    def iter_cached_candidates(self):
        return tuple(self.observed.values())


def brute_force_cost(evaluator, loss_weight):
    """Enumerate all 3**3 schedules without calling a planner or DP helper."""
    optimum = float("inf")
    for actions in product((None, 2, 4), repeat=3):
        current = evaluator.initial
        total = 0.0
        for step, n_samples in enumerate(actions):
            if n_samples is not None:
                current = evaluator.table[(step, n_samples)]
                total += current.training_cost
            total += loss_weight * current.future_losses[step]
        optimum = min(optimum, total)
    return optimum


def main() -> None:
    block = [((0.0,), 0), ((0.2,), 0), ((0.8,), 1), ((1.0,), 1)]
    timeline = ArrayTimelineAdapter([block] * 3, [block] * 3)
    evaluator = TableEvaluator()
    config = PlannerConfig(n_choices=(2, 4), loss_cost_weight=2.0)
    planner = RetrainingPlanner(
        timeline, PrototypeModelAdapter(), config, evaluator=evaluator,
    )
    result = planner.plan("exact")
    optimum = brute_force_cost(evaluator, config.loss_cost_weight)
    if abs(result.total_cost - optimum) > 1e-12:
        raise AssertionError(f"exact={result.total_cost}; exhaustive={optimum}")
    print(f"Exact cost: {result.total_cost:.6f}")
    print(f"Brute-force optimum: {optimum:.6f}")
    print("VERIFIED: exact planner matches brute-force optimum")


if __name__ == "__main__":
    main()
