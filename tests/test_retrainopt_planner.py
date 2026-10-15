"""Unit tests for the RetrainOpt public planner API."""

from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrainopt import PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter
from retrainopt.core.costs import compute_step_cost
from retrainopt.core.dp import evaluate_decision_sequence, solve_exact_schedule
from retrainopt.core.surrogate import NearestNeighborRegressor
import retrainopt.core.surrogate as surrogate_module
from retrainopt.interfaces import ModelAdapter
from retrainopt.types import CandidateEvaluation, Decision, DecisionAction


class PrototypeClassifierAdapter(ModelAdapter):
    """A tiny no-dependency classifier used for public API tests."""

    def fit(self, train_data, *, step, n_samples, random_seed):
        if not train_data:
            return {"prototypes": {}, "majority": 0}

        grouped = {}
        counts = {}
        for features, label in train_data:
            grouped.setdefault(int(label), []).append(float(features[0]))
            counts[int(label)] = counts.get(int(label), 0) + 1

        prototypes = {
            label: sum(values) / len(values)
            for label, values in grouped.items()
        }
        majority = max(counts.items(), key=lambda item: item[1])[0]
        return {"prototypes": prototypes, "majority": majority}

    def evaluate(self, model, eval_data, *, step):
        if not eval_data:
            return 0.0

        mistakes = 0
        prototypes = model["prototypes"]
        majority = model["majority"]

        for features, label in eval_data:
            feature_value = float(features[0])
            if prototypes:
                predicted = min(
                    prototypes.items(),
                    key=lambda item: abs(item[1] - feature_value),
                )[0]
            else:
                predicted = majority
            if int(predicted) != int(label):
                mistakes += 1
        return mistakes / len(eval_data)


class _TinyTimeline:
    def __init__(self, num_steps=4):
        self.num_steps = num_steps

    def train_size(self, step):
        return 10

    def sample_train_block(self, step, n_samples, seed):
        return [((float(step),), 0)] * int(n_samples)

    def get_eval_block(self, step):
        return [((float(step),), 0)]


class _NoopAdapter(ModelAdapter):
    def fit(self, train_data, *, step, n_samples, random_seed):
        return {"step": step, "n_samples": n_samples}

    def evaluate(self, model, eval_data, *, step):
        return 0.0


class _StaticSurrogate:
    def __init__(self, state_costs):
        self.state_costs = state_costs

    def predict_training_cost(self, step, n_value, dataset_size):
        return float(self.state_costs[(step, n_value)]["training_cost"]), 0.0

    def predict_loss(self, step, n_value, dataset_size, eval_step):
        return float(self.state_costs[(step, n_value)]["future_losses"].get(eval_step, 999.0)), 0.0


class _BatchLossOnlySurrogate:
    def __init__(self):
        self.loss_batch_calls = []

    def predict_training_cost(self, step, n_value, dataset_size):
        return 1.0, 0.0

    def predict_loss(self, step, n_value, dataset_size, eval_step):
        raise AssertionError("planner should use batched loss prediction when available")

    def predict_losses(self, step, n_value, dataset_size, eval_steps):
        steps = list(eval_steps)
        self.loss_batch_calls.append((step, n_value, dataset_size, steps))
        return [(0.1 + 0.01 * eval_step, 0.001) for eval_step in steps]


class _StaticEvaluationEngine:
    def __init__(self, state_costs):
        self.state_costs = state_costs
        self._candidate_cache = {}
        self.evaluate_candidate_calls = []

    def evaluate_initial_model(self, initial_model=None):
        return self._evaluation((-1, 0), observed=True)

    def evaluate_candidate(self, step, n_samples):
        model_id = (int(step), int(n_samples))
        if model_id not in self._candidate_cache:
            self.evaluate_candidate_calls.append(model_id)
            self._candidate_cache[model_id] = self._evaluation(model_id, observed=True)
        return self._candidate_cache[model_id]

    def evaluate_candidate_uncached(self, step, n_samples):
        return self._evaluation((int(step), int(n_samples)), observed=True)

    def iter_cached_candidates(self):
        return self._candidate_cache.values()

    def _evaluation(self, model_id, *, observed):
        info = self.state_costs[model_id]
        return CandidateEvaluation(
            model_id=model_id,
            dataset_size=10,
            training_seconds=float(info.get("training_cost", 0.0)),
            training_cost=float(info.get("training_cost", 0.0)),
            future_losses=dict(info["future_losses"]),
            observed=observed,
        )


def brute_force_cost(state_costs, num_steps, loss_weight, initial_model_id=(-1, 0)):
    best_cost = float("inf")

    choices_by_step = {}
    for model_id in state_costs:
        if model_id == initial_model_id:
            continue
        choices_by_step.setdefault(model_id[0], []).append(model_id[1])

    def recurse(step, current_model_id, running_cost):
        nonlocal best_cost
        if step == num_steps:
            best_cost = min(best_cost, running_cost)
            return

        keep_loss = state_costs[current_model_id]["future_losses"].get(step, float("inf"))
        recurse(
            step + 1,
            current_model_id,
            running_cost + compute_step_cost(0.0, keep_loss, loss_weight),
        )

        for n_value in choices_by_step.get(step, []):
            model_id = (step, n_value)
            info = state_costs[model_id]
            retrain_cost = info["training_cost"]
            loss_value = info["future_losses"].get(step, float("inf"))
            recurse(
                step + 1,
                model_id,
                running_cost + compute_step_cost(retrain_cost, loss_value, loss_weight),
            )

    recurse(0, initial_model_id, 0.0)
    return best_cost


class RetrainOptPlannerTests(unittest.TestCase):
    def build_timeline(self):
        train_blocks = [
            [((0.0,), 0), ((0.2,), 0), ((0.8,), 1), ((1.0,), 1)],
            [((0.1,), 0), ((0.3,), 0), ((0.7,), 1), ((0.9,), 1)],
            [((0.0,), 1), ((0.2,), 1), ((0.8,), 0), ((1.0,), 0)],
        ]
        eval_blocks = [
            [((0.05,), 0), ((0.95,), 1)],
            [((0.15,), 0), ((0.85,), 1)],
            [((0.05,), 1), ((0.95,), 0)],
        ]
        return ArrayTimelineAdapter(train_blocks=train_blocks, eval_blocks=eval_blocks)

    def test_exact_planner_matches_bruteforce(self):
        timeline = self.build_timeline()
        config = PlannerConfig(
            n_choices=(0.5, 1.0),
            min_train_size=1,
            time_cost_weight=0.0,
            loss_cost_weight=1.0,
            random_seed=7,
        )
        planner = RetrainingPlanner(timeline, PrototypeClassifierAdapter(), config=config)
        result = planner.plan(method="exact")
        state_costs = planner._build_exact_state_costs()
        brute_force = brute_force_cost(
            state_costs=state_costs,
            num_steps=timeline.num_steps,
            loss_weight=config.loss_cost_weight,
        )

        self.assertAlmostEqual(result.total_cost, brute_force, places=9)
        self.assertEqual(len(result.decisions), timeline.num_steps)

    def test_exact_solver_standalone_dp_matches_bruteforce(self):
        state_costs = {
            (-1, 0): {
                "training_cost": 0.0,
                "future_losses": {0: 0.8, 1: 0.8, 2: 0.8},
            },
            (0, 2): {
                "training_cost": 0.1,
                "future_losses": {0: 0.2, 1: 0.3, 2: 0.4},
            },
            (1, 2): {
                "training_cost": 0.1,
                "future_losses": {1: 0.1, 2: 0.2},
            },
            (2, 2): {
                "training_cost": 0.1,
                "future_losses": {2: 0.1},
            },
        }
        optimal_cost, decisions, _, _ = solve_exact_schedule(
            3,
            state_costs,
            loss_cost_weight=1.0,
        )
        brute_force = brute_force_cost(state_costs=state_costs, num_steps=3, loss_weight=1.0)
        self.assertAlmostEqual(optimal_cost, brute_force, places=9)
        self.assertEqual(len(decisions), 3)

    def test_bayes_planner_returns_structured_sequence(self):
        timeline = self.build_timeline()
        config = PlannerConfig(
            n_choices=(0.5, 1.0),
            min_train_size=1,
            time_cost_weight=0.0,
            loss_cost_weight=1.0,
            bayes_iterations=2,
            bayes_initial_samples=2,
            bayes_candidates_per_iteration=1,
            bayes_anchor_steps=(0.0,),
            acquisition="expected",
            record_bayes_true_history=True,
        )
        planner = RetrainingPlanner(timeline, PrototypeClassifierAdapter(), config=config)
        result = planner.plan(method="bayes")

        self.assertEqual(result.method, "bayes")
        self.assertEqual(len(result.decisions), timeline.num_steps)
        self.assertIn("evaluated_candidates", result.metadata)
        self.assertGreaterEqual(result.metadata["evaluated_candidates"], 1)
        self.assertIn("surrogate_evaluated_candidates", result.metadata)
        self.assertGreaterEqual(
            result.metadata["evaluated_candidates"],
            result.metadata["surrogate_evaluated_candidates"],
        )
        self.assertEqual(result.metadata["bayes_backend"], "lightweight")
        self.assertTrue(result.metadata["selected_sequence_evaluated_exactly"])
        self.assertEqual(result.metadata["bayes_iterations_completed"], 2)
        self.assertFalse(result.metadata["bayes_stopped_early"])
        self.assertEqual(len(result.metadata["bayes_initial_candidates"]), 2)
        self.assertTrue(
            all(
                isinstance(candidate, tuple) and len(candidate) == 2
                for candidate in result.metadata["bayes_initial_candidates"]
            )
        )
        self.assertEqual(len(result.metadata["bayes_estimated_total_history"]), 2)
        self.assertTrue(
            all(
                isinstance(value, float)
                for value in result.metadata["bayes_estimated_total_history"]
            )
        )
        self.assertEqual(len(result.metadata["bayes_estimated_decisions_history"]), 2)
        for decision_history in result.metadata["bayes_estimated_decisions_history"]:
            self.assertEqual(len(decision_history), timeline.num_steps)
            self.assertTrue(
                all(
                    set(decision) == {"action", "n_samples", "trained_at"}
                    for decision in decision_history
                )
            )
        self.assertEqual(len(result.metadata["bayes_true_total_history"]), 2)
        self.assertTrue(
            all(
                isinstance(value, float)
                for value in result.metadata["bayes_true_total_history"]
            )
        )
        self.assertGreaterEqual(
            min(result.metadata["bayes_true_total_history"]),
            0.0,
        )
        serialized = result.to_dict()
        self.assertEqual(serialized["method"], "bayes")
        self.assertEqual(len(serialized["decisions"]), timeline.num_steps)

    def test_bayes_total_cost_is_exactly_recomputed_for_selected_sequence(self):
        timeline = self.build_timeline()
        config = PlannerConfig(
            n_choices=(0.5, 1.0),
            min_train_size=1,
            time_cost_weight=0.0,
            loss_cost_weight=1.0,
            bayes_iterations=2,
            bayes_initial_samples=2,
            bayes_candidates_per_iteration=1,
        )
        planner = RetrainingPlanner(timeline, PrototypeClassifierAdapter(), config=config)
        result = planner.plan(method="bayes")

        realized_state_costs = {(-1, 0): planner.engine.evaluate_initial_model().to_state_cost()}
        for step, decision in enumerate(result.decisions):
            if decision.action != DecisionAction.RETRAIN:
                continue
            realized_state_costs[(step, int(decision.n_samples or 0))] = (
                planner.engine.evaluate_candidate(step, int(decision.n_samples or 0)).to_state_cost()
            )

        expected_total, expected_training, expected_loss = evaluate_decision_sequence(
            result.decisions,
            realized_state_costs,
            loss_cost_weight=config.loss_cost_weight,
        )
        self.assertAlmostEqual(result.total_cost, expected_total, places=9)
        self.assertAlmostEqual(result.training_cost, expected_training, places=9)
        self.assertAlmostEqual(result.loss_cost, expected_loss, places=9)

    def test_bayes_true_history_does_not_pollute_observed_candidate_cache(self):
        timeline = self.build_timeline()
        config = PlannerConfig(
            n_choices=(0.5, 1.0),
            min_train_size=1,
            time_cost_weight=0.0,
            loss_cost_weight=1.0,
        )
        planner = RetrainingPlanner(timeline, PrototypeClassifierAdapter(), config=config)
        initial_eval = planner.engine.evaluate_initial_model()
        decision_history = [
            [
                Decision(DecisionAction.KEEP),
                Decision(DecisionAction.RETRAIN, n_samples=2, trained_at=1),
                Decision(DecisionAction.KEEP),
            ]
        ]

        self.assertEqual(len(list(planner.engine.iter_cached_candidates())), 0)
        true_totals = planner._exact_total_history_for_sequences(initial_eval, decision_history)

        self.assertEqual(len(true_totals), 1)
        self.assertEqual(len(list(planner.engine.iter_cached_candidates())), 0)

    def test_bayes_sequence_acquisition_evaluates_retrain_decisions_from_acquisition_schedule(self):
        state_costs = {
            (-1, 0): {
                "training_cost": 0.0,
                "future_losses": {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0},
            },
            (0, 1): {
                "training_cost": 500.0,
                "future_losses": {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0},
            },
            (1, 1): {
                "training_cost": 1.0,
                "future_losses": {1: 1.0, 2: 1.0, 3: 100.0},
            },
            (2, 1): {
                "training_cost": 500.0,
                "future_losses": {2: 100.0, 3: 100.0},
            },
            (3, 1): {
                "training_cost": 1.0,
                "future_losses": {3: 1.0},
            },
        }
        planner = RetrainingPlanner(
            _TinyTimeline(),
            _NoopAdapter(),
            config=PlannerConfig(
                n_choices=(1,),
                loss_cost_weight=1.0,
                bayes_iterations=1,
                bayes_initial_samples=1,
                bayes_anchor_steps=(0.0,),
                bayes_candidates_per_iteration=1,
                acquisition="expected",
            ),
        )
        planner.engine = _StaticEvaluationEngine(state_costs)
        planner._fit_surrogates = lambda evaluations: _StaticSurrogate(state_costs)

        result = planner.plan(method="bayes")

        self.assertEqual(result.metadata["bayes_selection_strategy"], "sequence")
        self.assertEqual(
            result.metadata["bayes_new_candidates_by_iteration"],
            [[(1, 1), (3, 1)]],
        )
        self.assertEqual(result.metadata["bayes_retrain_steps_by_iteration"], [[1, 3]])

    def test_estimated_state_costs_use_batched_loss_prediction_when_available(self):
        planner = RetrainingPlanner(
            _TinyTimeline(num_steps=3),
            _NoopAdapter(),
            config=PlannerConfig(n_choices=(1,), min_train_size=1),
        )
        all_candidates = planner._all_candidate_ids()
        initial_eval = CandidateEvaluation(
            model_id=(-1, 0),
            dataset_size=0,
            training_seconds=0.0,
            training_cost=0.0,
            future_losses={0: 0.2, 1: 0.2, 2: 0.2},
            observed=True,
        )
        surrogate = _BatchLossOnlySurrogate()

        state_costs = planner._build_estimated_state_costs(
            initial_eval,
            all_candidates,
            [],
            surrogate,
        )

        self.assertEqual(len(surrogate.loss_batch_calls), len(all_candidates))
        self.assertEqual(
            state_costs[(0, 1)]["future_losses"],
            {0: 0.1, 1: 0.11, 2: 0.12000000000000001},
        )

    def test_bayes_sequence_acquisition_supplements_to_minimum_new_candidates(self):
        state_costs = {
            (-1, 0): {
                "training_cost": 0.0,
                "future_losses": {0: 100.0, 1: 100.0, 2: 100.0},
            },
            (0, 1): {
                "training_cost": 500.0,
                "future_losses": {0: 100.0, 1: 100.0, 2: 100.0},
            },
            (1, 1): {
                "training_cost": 1.0,
                "future_losses": {1: 1.0, 2: 1.0},
            },
            (2, 1): {
                "training_cost": 2.0,
                "future_losses": {2: 2.0},
            },
        }
        planner = RetrainingPlanner(
            _TinyTimeline(num_steps=3),
            _NoopAdapter(),
            config=PlannerConfig(
                n_choices=(1,),
                loss_cost_weight=1.0,
                bayes_iterations=1,
                bayes_initial_samples=1,
                bayes_anchor_steps=(0.0,),
                bayes_candidates_per_iteration=2,
                acquisition="expected",
            ),
        )
        planner.engine = _StaticEvaluationEngine(state_costs)
        planner._fit_surrogates = lambda evaluations: _StaticSurrogate(state_costs)

        result = planner.plan(method="bayes")

        self.assertEqual(
            result.metadata["bayes_new_candidates_by_iteration"],
            [[(1, 1), (2, 1)]],
        )

    def test_bayes_seed_candidates_cover_all_n_choices_at_anchor_steps(self):
        planner = RetrainingPlanner(
            _TinyTimeline(num_steps=5),
            _NoopAdapter(),
            config=PlannerConfig(
                n_choices=(1, 2),
                bayes_initial_samples=1,
                bayes_anchor_steps=(0.0, 0.5, 1.0),
            ),
        )

        selected = set(planner._seed_bayes_candidates(planner._all_candidate_ids()))

        self.assertTrue({(0, 1), (0, 2), (2, 1), (2, 2), (4, 1), (4, 2)}.issubset(selected))

    def test_empty_and_out_of_range_sampling_is_handled(self):
        timeline = ArrayTimelineAdapter(
            train_blocks=[
                [((0.0,), 0), ((1.0,), 1)],
                [],
            ],
            eval_blocks=[
                [((0.0,), 0), ((1.0,), 1)],
                [((0.0,), 0), ((1.0,), 1)],
            ],
        )
        sample = timeline.sample_train_block(0, 10, seed=1)
        self.assertEqual(len(sample), 2)

        planner = RetrainingPlanner(
            timeline,
            PrototypeClassifierAdapter(),
            config=PlannerConfig(
                n_choices=(1.0,),
                min_train_size=1,
                time_cost_weight=0.0,
                loss_cost_weight=1.0,
            ),
        )
        result = planner.plan(method="exact")
        self.assertEqual(len(result.decisions), timeline.num_steps)
        self.assertEqual(result.decisions[1].action, DecisionAction.KEEP)

    def test_initial_loss_bootstraps_empty_initial_timestep(self):
        timeline = ArrayTimelineAdapter(
            train_blocks=[
                [],
                [((0.0,), 0), ((1.0,), 1)],
            ],
            eval_blocks=[
                [((0.1,), 0), ((0.9,), 1)],
                [((0.0,), 0), ((1.0,), 1)],
            ],
        )
        planner = RetrainingPlanner(
            timeline,
            PrototypeClassifierAdapter(),
            config=PlannerConfig(
                n_choices=(1.0,),
                min_train_size=1,
                time_cost_weight=0.0,
                loss_cost_weight=1.0,
                initial_loss=0.5,
            ),
        )
        result = planner.plan(method="exact")
        self.assertEqual(len(result.decisions), 2)
        self.assertGreaterEqual(result.total_cost, 0.0)

    def test_builtin_prototype_adapter_supports_multiclass(self):
        timeline = ArrayTimelineAdapter(
            train_blocks=[
                [((0.0,), 0), ((1.0,), 1), ((2.0,), 2)],
                [((0.1,), 0), ((1.1,), 1), ((2.1,), 2)],
            ],
            eval_blocks=[
                [((0.05,), 0), ((1.05,), 1), ((2.05,), 2)],
                [((0.15,), 0), ((1.15,), 1), ((2.15,), 2)],
            ],
        )
        planner = RetrainingPlanner(
            timeline,
            PrototypeModelAdapter(),
            config=PlannerConfig(n_choices=(1.0,), time_cost_weight=0.0, loss_cost_weight=1.0),
        )
        result = planner.plan(method="exact")
        self.assertEqual(len(result.decisions), 2)
        self.assertLessEqual(result.loss_cost, 1.0)

    def test_resolve_n_choices_supports_mixed_fraction_and_absolute_inputs(self):
        config = PlannerConfig(n_choices=(0.25, 0.5, 3, 99), min_train_size=2)
        self.assertEqual(config.resolve_n_choices(8), [2, 3, 4, 8])

    def test_unknown_bayes_backend_is_rejected(self):
        timeline = self.build_timeline()
        with self.assertRaisesRegex(ValueError, "Unknown bayes backend"):
            RetrainingPlanner(
                timeline,
                PrototypeClassifierAdapter(),
                config=PlannerConfig(bayes_backend="not-a-backend"),
            )

    @unittest.skipIf(surrogate_module._np is None, "NumPy is not installed.")
    def test_nearest_neighbor_numpy_path_matches_dependency_free_path(self):
        rows = [
            (0.0, 1.0, 5.0),
            (1.0, 2.0, 4.0),
            (2.0, 3.0, 3.0),
            (3.0, 4.0, 2.0),
            (4.0, 5.0, 1.0),
        ]
        targets = [5.0, 4.0, 3.0, 2.0, 1.0]
        query = (1.5, 2.5, 3.5)

        fast = NearestNeighborRegressor(k_neighbors=3, uncertainty_floor=0.05)
        fast.fit(rows, targets)
        fast_mean, fast_std = fast.predict(query)

        original_numpy = surrogate_module._np
        try:
            surrogate_module._np = None
            fallback = NearestNeighborRegressor(k_neighbors=3, uncertainty_floor=0.05)
            fallback.fit(rows, targets)
            fallback_mean, fallback_std = fallback.predict(query)
        finally:
            surrogate_module._np = original_numpy

        self.assertAlmostEqual(fast_mean, fallback_mean, places=12)
        self.assertAlmostEqual(fast_std, fallback_std, places=12)

    @unittest.skipIf(surrogate_module._np is None, "NumPy is not installed.")
    def test_nearest_neighbor_predict_many_matches_single_predictions(self):
        rows = [
            (0.0, 1.0, 5.0),
            (1.0, 2.0, 4.0),
            (2.0, 3.0, 3.0),
            (3.0, 4.0, 2.0),
            (4.0, 5.0, 1.0),
        ]
        targets = [5.0, 4.0, 3.0, 2.0, 1.0]
        queries = [
            (1.5, 2.5, 3.5),
            (3.0, 4.0, 2.0),
            (4.2, 5.2, 0.8),
        ]
        regressor = NearestNeighborRegressor(k_neighbors=3, uncertainty_floor=0.05)
        regressor.fit(rows, targets)

        batch_predictions = regressor.predict_many(queries)
        single_predictions = [regressor.predict(query) for query in queries]

        self.assertEqual(len(batch_predictions), len(single_predictions))
        for (batch_mean, batch_std), (single_mean, single_std) in zip(batch_predictions, single_predictions):
            self.assertAlmostEqual(batch_mean, single_mean, delta=1e-10)
            self.assertAlmostEqual(batch_std, single_std, delta=1e-10)

    @unittest.skipIf(
        surrogate_module._np is None or importlib.util.find_spec("sklearn") is None,
        "NumPy and scikit-learn are required for indexed nearest-neighbor tests.",
    )
    def test_nearest_neighbor_builds_optional_sklearn_index_without_changing_predictions(self):
        rows = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            (2.0, 3.0, 4.0),
            (3.0, 4.0, 5.0),
            (4.0, 5.0, 6.0),
        ]
        targets = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        queries = [
            (0.0, 0.0, 0.0),
            (1.5, 2.5, 3.5),
            (4.1, 5.1, 6.1),
        ]
        regressor = NearestNeighborRegressor(k_neighbors=3, uncertainty_floor=0.05)
        regressor.fit(rows, targets)

        self.assertIsNotNone(getattr(regressor, "_neighbor_index", None))
        batch_predictions = regressor.predict_many(queries)
        single_predictions = [regressor.predict(query) for query in queries]

        for (batch_mean, batch_std), (single_mean, single_std) in zip(batch_predictions, single_predictions):
            self.assertAlmostEqual(batch_mean, single_mean, delta=1e-10)
            self.assertAlmostEqual(batch_std, single_std, delta=1e-10)

    @unittest.skipIf(importlib.util.find_spec("botorch") is not None, "BoTorch extra is installed.")
    def test_botorch_backend_requires_optional_dependency(self):
        timeline = self.build_timeline()
        planner = RetrainingPlanner(
            timeline,
            PrototypeClassifierAdapter(),
            config=PlannerConfig(
                n_choices=(0.5, 1.0),
                min_train_size=1,
                time_cost_weight=0.0,
                loss_cost_weight=1.0,
                bayes_iterations=1,
                bayes_initial_samples=1,
                bayes_backend="botorch",
                bayes_anchor_steps=(0.0,),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "BoTorch backend requires"):
            planner.plan(method="bayes")


class BinaryTimelineCompatibilityTests(unittest.TestCase):
    def test_binary_timeline_adapter_round_trip(self):
        from retrainopt.adapters import BinaryTimelineAdapter

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_dir = root / "train"
            test_dir = root / "test"
            train_dir.mkdir()
            test_dir.mkdir()
            (root / "metadata.json").write_text('{"num_features": 1}')

            self._write_block(train_dir / "0.bin", [((0.0,), 0), ((1.0,), 1)])
            self._write_block(test_dir / "0.bin", [((0.1,), 0), ((0.9,), 1)])

            timeline = BinaryTimelineAdapter(root)
            planner = RetrainingPlanner(
                timeline,
                PrototypeClassifierAdapter(),
                config=PlannerConfig(n_choices=(1.0,), time_cost_weight=0.0),
            )
            result = planner.plan(method="exact")

            self.assertEqual(len(result.decisions), 1)

    @staticmethod
    def _write_block(path: Path, samples):
        with path.open("wb") as handle:
            for features, label in samples:
                handle.write(struct.pack("<2f", float(features[0]), float(label)))


if __name__ == "__main__":
    unittest.main()
