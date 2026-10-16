"""Measured final plans must retain the best available candidate combinations."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from retrainopt import CandidateEvaluation, PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter
from retrainopt.types import Decision, DecisionAction


class TableProvider:
    def __init__(self, table):
        self.table = table
        self.cache = {}
        self.fit_calls = []
        self.diagnostic_calls = []

    def evaluation(self, model_id):
        cost, losses = self.table[model_id]
        return CandidateEvaluation(model_id, 1, 0.0, cost, dict(losses))

    def evaluate_initial_model(self, initial_model=None):
        return self.evaluation((-1, 0))

    def evaluate_candidate(self, step, n_samples):
        model_id = (step, n_samples)
        if model_id not in self.cache:
            self.fit_calls.append(model_id)
            self.cache[model_id] = self.evaluation(model_id)
        return self.cache[model_id]

    def evaluate_candidate_uncached(self, step, n_samples):
        self.diagnostic_calls.append((step, n_samples))
        # A deliberately cheaper diagnostic measurement must not change selection.
        losses = self.table[step, n_samples][1]
        return CandidateEvaluation((step, n_samples), 1, 0.0, 0.0, {t: 0.0 for t in losses})

    def iter_cached_candidates(self):
        return self.cache.values()


class FixedPredictions:
    def __init__(self, table):
        self.table = table

    def predict_training_cost(self, step, n_value, dataset_size):
        return self.table[step, n_value][0], 0.0

    def predict_loss(self, step, n_value, dataset_size, eval_step):
        return self.table[step, n_value][1][eval_step], 0.0


def make_planner(table, **options):
    steps = len(table[-1, 0][1])
    timeline = ArrayTimelineAdapter(
        [[((0.0,), 0)] for _ in range(steps)],
        [[((0.0,), 0)] for _ in range(steps)],
    )
    config = PlannerConfig(
        n_choices=(1,), random_seed=1, bayes_anchor_steps=(),
        bayes_initial_samples=1, **{"bayes_iterations": 0, **options},
    )
    provider = TableProvider(table)
    return RetrainingPlanner(timeline, object(), config, evaluator=provider), provider


class BayesFinalizationContracts(unittest.TestCase):
    def test_final_measurement_replaces_optimistic_plan_with_known_better_schedule(self):
        table = {
            (-1, 0): (0.0, {0: 0.5, 1: 0.5}),
            (0, 1): (0.1, {0: 1.0, 1: 1.0}),
            (1, 1): (0.1, {1: 0.0}),
        }
        planner, provider = make_planner(table)

        result = planner.plan("bayes")

        # Before final measurement the proposal was RETRAIN(0), KEEP: estimated
        # 0.1, but actually 2.1. Reuse the already observed step-1 candidate.
        self.assertEqual(result.decisions, [Decision(DecisionAction.KEEP), Decision(DecisionAction.RETRAIN, 1, 1)])
        self.assertAlmostEqual(result.total_cost, 0.6)
        self.assertAlmostEqual(result.estimated_total_cost, 0.6)
        self.assertAlmostEqual(result.training_cost, 0.1)
        self.assertAlmostEqual(result.loss_cost, 0.5)
        self.assertEqual(provider.fit_calls, [(1, 1), (0, 1)])
        self.assertEqual(result.metadata["finalization_strategy"], "observed_candidates_dp")
        self.assertEqual(result.metadata["finalization_new_candidates"], [(0, 1)])
        self.assertEqual(result.metadata["surrogate_evaluated_candidates"], 1)
        self.assertEqual(result.metadata["evaluated_candidates"], 2)
        self.assertEqual(result.metadata["bayes_iterations_completed"], 0)

    def test_final_measurements_are_bounded_and_can_be_recombined_without_refitting(self):
        table = {(-1, 0): (0.0, {t: 10.0 for t in range(4)})}
        predictions = {}
        for step in range(4):
            table[step, 1] = (0.25, {t: 0.0 if step == 3 else 9.0 for t in range(step, 4)})
            predictions[step, 1] = (0.25, {t: 0.0 if t == step else 10.0 for t in range(step, 4)})
        planner, provider = make_planner(table)
        # Seed only step 3; the estimated proposal retrains at all four steps.
        with patch.object(planner, "_seed_bayes_candidates", return_value=[(3, 1)]), patch.object(
            planner, "_fit_surrogates", return_value=FixedPredictions(predictions),
        ) as fit_surrogates:
            result = planner.plan("bayes")

        self.assertEqual(result.decisions, [
            Decision(DecisionAction.RETRAIN, 1, 0), Decision(DecisionAction.KEEP),
            Decision(DecisionAction.KEEP), Decision(DecisionAction.RETRAIN, 1, 3),
        ])
        self.assertAlmostEqual(result.total_cost, 27.5)
        self.assertAlmostEqual(result.estimated_total_cost, 20.5)
        self.assertEqual(provider.fit_calls, [(3, 1), (0, 1), (1, 1), (2, 1)])
        self.assertEqual(result.metadata["finalization_new_candidates"], [(0, 1), (1, 1), (2, 1)])
        self.assertEqual(result.metadata["evaluated_candidates"], 4)
        fit_surrogates.assert_called_once()

    def test_new_candidate_can_win_and_preserve_its_pre_measurement_estimate(self):
        table = {
            (-1, 0): (0.0, {0: 0.5, 1: 0.5}),
            (0, 1): (0.2, {0: 0.1, 1: 0.1}),
            (1, 1): (10.0, {1: 0.0}),
        }
        planner, provider = make_planner(table, loss_cost_weight=2.0)
        predictions = {(0, 1): (0.1, {0: 0.0, 1: 0.0})}
        with patch.object(planner, "_fit_surrogates", return_value=FixedPredictions(predictions)):
            result = planner.plan("bayes")
        self.assertEqual(result.decisions, [Decision(DecisionAction.RETRAIN, 1, 0), Decision(DecisionAction.KEEP)])
        self.assertAlmostEqual(result.total_cost, 0.6)
        self.assertAlmostEqual(result.training_cost, 0.2)
        self.assertAlmostEqual(result.loss_cost, 0.2)
        self.assertAlmostEqual(result.weighted_loss_cost, 0.4)
        self.assertAlmostEqual(result.estimated_total_cost, 0.1)
        self.assertEqual(provider.fit_calls, [(1, 1), (0, 1)])

    def test_diagnostics_cannot_change_selection_or_observed_candidate_counts(self):
        table = {
            (-1, 0): (0.0, {0: 0.5, 1: 0.5}),
            (0, 1): (0.1, {0: 1.0, 1: 1.0}),
            (1, 1): (0.1, {1: 0.0}),
        }
        planner, provider = make_planner(table, bayes_iterations=1)
        plain = planner.plan("bayes")
        diagnostic_planner, diagnostic_provider = make_planner(
            table, bayes_iterations=1, record_bayes_true_history=True,
        )
        diagnostic = diagnostic_planner.plan("bayes")
        self.assertEqual(diagnostic.decisions, plain.decisions)
        self.assertEqual(diagnostic.total_cost, plain.total_cost)
        self.assertEqual(diagnostic.estimated_total_cost, plain.estimated_total_cost)
        self.assertEqual(diagnostic.metadata["evaluated_candidates"], plain.metadata["evaluated_candidates"])
        self.assertEqual(diagnostic.metadata["finalization_new_candidates"], [])
        self.assertEqual(diagnostic_provider.fit_calls, provider.fit_calls)
        self.assertEqual(diagnostic_provider.diagnostic_calls, [(1, 1)])
        self.assertEqual(diagnostic.metadata["bayes_true_total_history"], [0.5])
        self.assertAlmostEqual(diagnostic.total_cost, 0.6)

    def test_keep_only_finalization_needs_no_candidate_measurements(self):
        planner, provider = make_planner({(-1, 0): (0.0, {0: 0.5})}, min_train_size=2)
        result = planner.plan("bayes")
        self.assertEqual(result.decisions, [Decision(DecisionAction.KEEP)])
        self.assertEqual(result.total_cost, 0.5)
        self.assertEqual(result.metadata["finalization_strategy"], "observed_candidates_dp")
        self.assertEqual(result.metadata["finalization_new_candidates"], [])
        self.assertEqual(result.metadata["evaluated_candidates"], 0)
        self.assertEqual(provider.fit_calls, [])

    def test_failed_final_measurement_propagates_without_retry(self):
        table = {
            (-1, 0): (0.0, {0: 0.5, 1: 0.5}),
            (0, 1): (0.1, {0: 1.0, 1: 1.0}),
            (1, 1): (0.1, {1: 0.0}),
        }
        planner, provider = make_planner(table)
        original = provider.evaluate_candidate
        error = RuntimeError("measurement failed")
        def evaluate(step, n_samples):
            if step == 0:
                raise error
            return original(step, n_samples)
        provider.evaluate_candidate = Mock(side_effect=evaluate)
        with self.assertRaises(RuntimeError) as caught:
            planner.plan("bayes")
        self.assertIs(caught.exception, error)
        self.assertEqual(provider.evaluate_candidate.call_count, 2)
        self.assertEqual(set(provider.cache), {(1, 1)})


if __name__ == "__main__":
    unittest.main()
