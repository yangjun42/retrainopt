"""Independent paper-contract and public boundary checks for the algorithm API."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math
import random
import unittest

from retrainopt import CandidateEvaluation, EvaluationProvider, PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter
from retrainopt.core.acquisition import AcquisitionManager
from retrainopt.core.dp import StageCosts, evaluate_decision_sequence, solve_exact_schedule, solve_stage_schedule
from retrainopt.interfaces import ModelAdapter
from retrainopt.types import Decision, DecisionAction


class NoTraining(ModelAdapter):
    def fit(self, *args, **kwargs):
        raise AssertionError("Injected evaluator must own training")

    def evaluate(self, *args, **kwargs):
        raise AssertionError("Injected evaluator must own evaluation")


class TableProvider:
    def __init__(self, table, size=10):
        self.table = table
        self.size = size
        self.observed = {}

    def _get(self, model_id):
        record = self.table[model_id]
        return CandidateEvaluation(
            model_id=model_id, dataset_size=self.size,
            training_seconds=record["training_cost"], training_cost=record["training_cost"],
            future_losses=dict(record["future_losses"]),
        )

    def evaluate_initial_model(self, initial_model=None):
        return self._get((-1, 0))

    def evaluate_candidate(self, step, n_samples):
        model_id = (step, n_samples)
        if model_id not in self.observed:
            self.observed[model_id] = self._get(model_id)
        return self.observed[model_id]

    def evaluate_candidate_uncached(self, step, n_samples):
        return self._get((step, n_samples))

    def iter_cached_candidates(self):
        return self.observed.values()


class Moments:
    def __init__(self, training, losses):
        self.training = training
        self.losses = losses

    def predict_training_cost(self, step, n_value, dataset_size):
        return self.training[step, n_value]

    def predict_loss(self, step, n_value, dataset_size, eval_step):
        return self.losses[step, n_value, eval_step]


def timeline(steps=2, size=10):
    return ArrayTimelineAdapter(
        [[((0.0,), 0)] * size for _ in range(steps)],
        [[((0.0,), 0)] for _ in range(steps)],
    )


def evaluation(model_id, training, losses):
    return CandidateEvaluation(model_id, 10, training, training, losses)


class AcquisitionContracts(unittest.TestCase):
    def test_alias_uses_explicit_config_and_unknown_name_is_rejected(self):
        acquisition = AcquisitionManager()
        self.assertEqual(acquisition.score(10, 2, "lcb", beta=3), 4)
        self.assertEqual(acquisition.score(10, 2, "ucb", beta=3), 16)
        self.assertEqual(acquisition.score(10, 2, "lcb_beta_2", beta=3), 6)
        self.assertAlmostEqual(
            acquisition.score(1, 1, "ei", xi=2, best_cost=2),
            acquisition.score(1, 1, "ei_xi_2", xi=0, best_cost=2),
        )
        with self.assertRaisesRegex(ValueError, "Unknown acquisition"):
            acquisition.parse_name("lcb_typo")
        with self.assertRaises(ValueError):
            acquisition.score(10, 2, "lcb", beta=True)
        with self.assertRaisesRegex(ValueError, "overflowed"):
            acquisition.score(10, 1e300, "variance")

    def test_observed_costs_follow_all_six_acquisitions(self):
        initial = evaluation((-1, 0), 0, {0: 5, 1: 4})
        observed = evaluation((0, 1), 3, {0: 2, 1: 1})
        for acquisition in ("expected", "variance", "lcb", "ucb", "ei", "pi"):
            with self.subTest(acquisition=acquisition):
                planner = RetrainingPlanner(
                    timeline(), NoTraining(), PlannerConfig(acquisition=acquisition, loss_cost_weight=2),
                )
                stages = planner._build_acquisition_state_costs(initial, [(0, 1)], [observed], None)
                zero = acquisition in ("variance", "ei", "pi")
                self.assertEqual(stages.retrain[(0, 1)], 0 if zero else 7)
                self.assertEqual(stages.keep[(0, 1)][1], 0 if zero else 2)
                self.assertEqual(stages.keep[(-1, 0)][0], 0 if zero else 10)
                self.assertNotIn(0, stages.keep[(0, 1)])

    def test_variance_weight_squared_changes_candidate_order(self):
        planner = RetrainingPlanner(
            timeline(1), NoTraining(), PlannerConfig(acquisition="variance", loss_cost_weight=2),
        )
        surrogate = Moments(
            {(0, 1): (3, 2), (0, 2): (3, math.sqrt(7))},
            {(0, 1, 0): (2, 1), (0, 2, 0): (2, 0)},
        )
        stages = planner._build_acquisition_state_costs(
            evaluation((-1, 0), 0, {0: 10}), [(0, 1), (0, 2)], [], surrogate,
        )
        self.assertAlmostEqual(stages.retrain[(0, 1)], -8)
        self.assertAlmostEqual(stages.retrain[(0, 2)], -7)
        self.assertEqual(solve_stage_schedule(1, stages)[1][0].n_samples, 1)

    def test_lcb_and_fallback_combine_variances_not_standard_deviations(self):
        planner = RetrainingPlanner(
            timeline(1), NoTraining(), PlannerConfig(acquisition="lcb", loss_cost_weight=2, exploration_beta=3),
        )
        surrogate = Moments({(0, 1): (10, 3)}, {(0, 1, 0): (4, 2)})
        self.assertEqual(planner._predict_immediate_cost(surrogate, 0, 1, 10), (18, 5))
        stages = planner._build_acquisition_state_costs(evaluation((-1, 0), 0, {0: 10}), [(0, 1)], [], surrogate)
        self.assertEqual(stages.retrain[(0, 1)], 3)

    def test_zero_objective_weights_remove_their_uncertainty(self):
        surrogate = Moments({(0, 1): (10, 3)}, {(0, 1, 0): (4, 2)})
        for time_weight, loss_weight, expected in ((1, 0, (10, 3)), (0, 2, (8, 4)), (0, 0, (0, 0))):
            with self.subTest(time_weight=time_weight, loss_weight=loss_weight):
                planner = RetrainingPlanner(
                    timeline(1), NoTraining(), PlannerConfig(time_cost_weight=time_weight, loss_cost_weight=loss_weight),
                )
                self.assertEqual(planner._predict_immediate_cost(surrogate, 0, 1, 10), expected)

    def test_ei_pi_use_one_legal_complete_incumbent_and_paper_zero_std(self):
        initial = evaluation((-1, 0), 0, {0: 5, 1: 4})
        first = evaluation((0, 1), 10, {0: 0.1, 1: 3})
        last = evaluation((1, 1), 12, {1: 0.001})
        for name in ("ei", "pi"):
            planner = RetrainingPlanner(
                timeline(), NoTraining(), PlannerConfig(acquisition=name, loss_cost_weight=2, improvement_xi=0.5),
            )
            self.assertEqual(planner._best_observed_stage_cost(initial, [first, last]), 6)
            surrogate = Moments({(0, 2): (3, 1)}, {(0, 2, 0): (2, 2), (0, 2, 1): (2, 2)})
            stages = planner._build_acquisition_state_costs(initial, [(0, 2)], [first, last], surrogate)
            expected = AcquisitionManager().score(7, math.sqrt(17), name, best_cost=6, xi=0.5)
            self.assertAlmostEqual(stages.retrain[(0, 2)], expected)
            self.assertEqual(AcquisitionManager().score(1, 0, name, best_cost=2), 0)

    def test_surrogate_output_boundary_rejects_nan_and_wrong_batch_length(self):
        planner = RetrainingPlanner(timeline(1), NoTraining())
        with self.assertRaisesRegex(ValueError, "finite"):
            planner._predict_immediate_cost(Moments({(0, 1): (float("nan"), 1)}, {(0, 1, 0): (1, 1)}), 0, 1, 10)
        class ShortBatch:
            def predict_losses(self, *args):
                return []
        with self.assertRaisesRegex(ValueError, "evaluation steps"):
            planner._predict_future_losses(ShortBatch(), 0, 1, 10)


class DynamicProgrammingContracts(unittest.TestCase):
    def test_random_objective_matrices_match_independent_enumeration(self):
        rng = random.Random(619)
        for _ in range(100):
            steps = rng.randrange(1, 5)
            weight = rng.uniform(0, 3)
            table = {(-1, 0): {"training_cost": 0, "future_losses": {t: rng.uniform(0, 4) for t in range(steps)}}}
            for t in range(steps):
                for n in range(1, rng.randrange(1, 4)):
                    table[t, n] = {"training_cost": rng.uniform(0, 3), "future_losses": {s: rng.uniform(0, 4) for s in range(t, steps)}}
            def enumerate_cost(t, model):
                if t == steps:
                    return 0
                alternatives = [weight * table[model]["future_losses"][t] + enumerate_cost(t + 1, model)]
                for candidate, info in table.items():
                    if candidate[0] == t:
                        alternatives.append(info["training_cost"] + weight * info["future_losses"][t] + enumerate_cost(t + 1, candidate))
                return min(alternatives)
            cost, decisions, _, _ = solve_exact_schedule(steps, table, loss_cost_weight=weight)
            self.assertAlmostEqual(cost, enumerate_cost(0, (-1, 0)), places=12)
            self.assertAlmostEqual(evaluate_decision_sequence(decisions, table, loss_cost_weight=weight)[0], cost, places=12)

    def test_ties_keep_and_then_choose_smallest_retrain(self):
        table = {
            (-1, 0): {"training_cost": 0, "future_losses": {0: 1}},
            (0, 2): {"training_cost": 1, "future_losses": {0: 0}},
            (0, 1): {"training_cost": 1, "future_losses": {0: 0}},
        }
        self.assertEqual(solve_exact_schedule(1, table)[1][0].action, DecisionAction.KEEP)
        table[-1, 0]["future_losses"][0] = 2
        self.assertEqual(solve_exact_schedule(1, table)[1][0].n_samples, 1)

    def test_public_solver_rejects_invalid_cost_records(self):
        for loss in (float("nan"), float("inf"), -1, True, "1"):
            with self.subTest(loss=loss), self.assertRaises(ValueError):
                solve_exact_schedule(1, {(-1, 0): {"training_cost": 0, "future_losses": {0: loss}}})
        for table in (
            {(-1, 0): {"future_losses": {}}},
            {(-1, 0): {"future_losses": {0: 1}}, (0, 0): {"training_cost": 1, "future_losses": {0: 1}}},
            {(-1, 0): {"future_losses": {0: 1}}, (0, 1): {"future_losses": {0: 1}}},
        ):
            with self.subTest(table=table), self.assertRaises(ValueError):
                solve_exact_schedule(1, table)
        with self.assertRaises(ValueError):
            solve_exact_schedule(1, {(-1, 0): {"future_losses": {0: 1}}}, loss_cost_weight=-1)

    def test_signed_stage_costs_can_explore_without_becoming_observations(self):
        stages = StageCosts({(-1, 0): {0: 0, 1: 0}, (0, 1): {1: -4}, (1, 1): {}}, {(0, 1): -1, (1, 1): -2})
        cost, decisions, _, _ = solve_stage_schedule(2, stages)
        self.assertEqual(cost, -5)
        self.assertEqual([d.action for d in decisions], [DecisionAction.RETRAIN, DecisionAction.KEEP])

    def test_random_signed_stage_matrices_match_independent_enumeration(self):
        rng = random.Random(620)
        for _ in range(80):
            steps = rng.randrange(1, 5)
            keep = {(-1, 0): {t: rng.randint(-5, 5) for t in range(steps)}}
            retrain = {}
            for t in range(steps):
                for n in range(1, rng.randrange(1, 4)):
                    retrain[t, n] = rng.randint(-5, 5)
                    keep[t, n] = {s: rng.randint(-5, 5) for s in range(t + 1, steps)}
            def enumerate_cost(t, model):
                if t == steps:
                    return 0
                alternatives = [keep[model][t] + enumerate_cost(t + 1, model)]
                alternatives.extend(cost + enumerate_cost(t + 1, candidate) for candidate, cost in retrain.items() if candidate[0] == t)
                return min(alternatives)
            actual = solve_stage_schedule(steps, StageCosts(keep, retrain))[0]
            self.assertEqual(actual, enumerate_cost(0, (-1, 0)))

    def test_decision_evaluation_rejects_absent_candidate(self):
        with self.assertRaisesRegex(ValueError, "No candidate"):
            evaluate_decision_sequence([Decision(DecisionAction.RETRAIN, 1)], {(-1, 0): {"future_losses": {0: 1}}})

    def test_decision_enforces_action_and_payload_before_serialization(self):
        for kwargs in (
            {"action": "keep"},
            {"action": DecisionAction.RETRAIN, "n_samples": 1, "trained_at": True},
            {"action": DecisionAction.RETRAIN, "n_samples": 0},
            {"action": DecisionAction.KEEP, "n_samples": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Decision(**kwargs)


class PlannerBoundaryContracts(unittest.TestCase):
    def test_injected_provider_and_weighted_output_schema(self):
        table = {
            (-1, 0): {"training_cost": 0, "future_losses": {0: 5, 1: 5}},
            (0, 1): {"training_cost": 1, "future_losses": {0: 1, 1: 1}},
            (1, 1): {"training_cost": 8, "future_losses": {1: 1}},
        }
        provider = TableProvider(table)
        self.assertIsInstance(provider, EvaluationProvider)
        result = RetrainingPlanner(timeline(), NoTraining(), PlannerConfig(n_choices=(1,), loss_cost_weight=2), evaluator=provider).plan()
        self.assertEqual(result.total_cost, 5)
        self.assertEqual(result.loss_cost, 2)
        self.assertEqual(result.weighted_loss_cost, 4)
        self.assertEqual(result.total_cost, result.training_cost + result.weighted_loss_cost)
        self.assertEqual(result.to_dict()["schema_version"], 2)
        self.assertEqual(result.metadata["metadata_schema_version"], 1)
        self.assertEqual(result.metadata["algorithm_version"], "0.3")
        self.assertEqual(result.metadata["initial_model_source"], "provider_supplied")

    def test_zero_candidates_are_keep_only_for_both_methods(self):
        for method in ("exact", "bayes"):
            planner = RetrainingPlanner(
                timeline(size=1), PrototypeModelAdapter(), PlannerConfig(n_choices=(0.01,), initial_loss=0.5, loss_cost_weight=2),
            )
            result = planner.plan(method)
            self.assertEqual(result.total_cost, 2)
            self.assertTrue(all(decision.action == DecisionAction.KEEP for decision in result.decisions))
            self.assertEqual(result.metadata["candidate_count"], 0)
            self.assertEqual(result.metadata["initial_model_source"], "configured_initial_loss")

    def test_fully_observed_bayes_needs_no_surrogate_fit(self):
        table = {
            (-1, 0): {"training_cost": 0, "future_losses": {0: 5}},
            (0, 1): {"training_cost": 1, "future_losses": {0: 1}},
        }
        planner = RetrainingPlanner(timeline(1), NoTraining(), PlannerConfig(n_choices=(1,)), evaluator=TableProvider(table))
        def unexpected_fit(*args):
            raise AssertionError("All candidate costs are already observed")
        planner._fit_surrogates = unexpected_fit
        result = planner.plan("bayes")
        self.assertEqual(result.total_cost, 2)
        self.assertEqual(result.metadata["bayes_iterations_completed"], 0)
        self.assertEqual(result.metadata["initial_model_source"], "provider_supplied")

    def test_provider_cannot_alias_integer_model_ids_with_floats_or_booleans(self):
        table = {
            (-1, 0): {"training_cost": 0, "future_losses": {0: 1}},
            (0, 1): {"training_cost": 0, "future_losses": {0: 0}},
        }
        for requested, aliases in (
            ((-1, 0), ((-1.0, 0), (-1, False))),
            ((0, 1), ((0.0, 1), (False, 1), (0, 1.0), (0, True))),
        ):
            for alias in aliases:
                class AliasedProvider(TableProvider):
                    def _get(self, model_id):
                        result = super()._get(model_id)
                        return replace(result, model_id=alias) if model_id == requested else result
                for method in ("exact", "bayes"):
                    with self.subTest(requested=requested, alias=alias, method=method):
                        planner = RetrainingPlanner(
                            timeline(1), NoTraining(), PlannerConfig(n_choices=(1,)), evaluator=AliasedProvider(table),
                        )
                        with self.assertRaisesRegex(ValueError, "Invalid model id"):
                            planner.plan(method)

    def test_provider_observed_flag_must_be_true_boolean(self):
        table = {
            (-1, 0): {"training_cost": 0, "future_losses": {0: 1}},
            (0, 1): {"training_cost": 0, "future_losses": {0: 0}},
        }
        for requested in table:
            for invalid in (False, "false", "true", 0, 1, None):
                class UnobservedProvider(TableProvider):
                    def _get(self, model_id):
                        result = super()._get(model_id)
                        return replace(result, observed=invalid) if model_id == requested else result
                for method in ("exact", "bayes"):
                    with self.subTest(requested=requested, observed=invalid, method=method):
                        planner = RetrainingPlanner(
                            timeline(1), NoTraining(), PlannerConfig(n_choices=(1,)), evaluator=UnobservedProvider(table),
                        )
                        with self.assertRaisesRegex(ValueError, "observed costs"):
                            planner.plan(method)

    def test_provider_nan_or_wrong_candidate_is_rejected(self):
        invalid = TableProvider({(-1, 0): {"training_cost": 0, "future_losses": {0: float("nan")}}})
        with self.assertRaisesRegex(ValueError, "finite"):
            RetrainingPlanner(timeline(1), NoTraining(), evaluator=invalid).plan()
        with self.assertRaisesRegex(TypeError, "EvaluationProvider"):
            RetrainingPlanner(timeline(1), NoTraining(), evaluator=object())
        class WrongCandidate(TableProvider):
            def evaluate_candidate(self, step, n_samples):
                return evaluation((step, n_samples + 1), 1, {0: 1})
        wrong = WrongCandidate({(-1, 0): {"training_cost": 0, "future_losses": {0: 1}}})
        with self.assertRaisesRegex(ValueError, "wrong model_id"):
            RetrainingPlanner(timeline(1), NoTraining(), PlannerConfig(n_choices=(1,)), evaluator=wrong).plan()

    def test_injected_initial_baseline_must_honor_config(self):
        provider = TableProvider({(-1, 0): {"training_cost": 0, "future_losses": {0: 9}}})
        planner = RetrainingPlanner(timeline(1), NoTraining(), PlannerConfig(initial_loss=0.5), evaluator=provider)
        with self.assertRaisesRegex(ValueError, "disagree"):
            planner.plan()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            planner.plan(initial_model=object())

    def test_config_snapshot_isolated_from_mutation(self):
        options = {"nested": [1]}
        config = PlannerConfig(n_choices=[1], bayes_backend_options=options)
        planner = RetrainingPlanner(timeline(1), NoTraining(), config)
        config.bayes_backend_options["nested"].append(2)
        options["nested"].append(3)
        self.assertEqual(planner.config.bayes_backend_options["nested"], [1])
        self.assertEqual(planner.config.n_choices, (1,))
        with self.assertRaises(FrozenInstanceError):
            config.loss_cost_weight = 5


if __name__ == "__main__":
    unittest.main()
