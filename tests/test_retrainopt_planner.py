"""Unit tests for the new RetrainOpt public planner API."""

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
from retrainopt.interfaces import ModelAdapter
from retrainopt.types import DecisionAction


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
            acquisition="expected",
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

    def test_empty_and_out_of_range_sampling_is_handled(self):
        timeline = ArrayTimelineAdapter(
            train_blocks=[
                [((0.0,), 0), ((1.0,), 1)],
                [],
            ],
            eval_blocks=[
                [((0.0,), 0), ((1.0,), 1)],
                [],
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
        config = PlannerConfig(n_choices=(0.0, 0.25, 0.5, 3, 99), min_train_size=2)
        self.assertEqual(config.resolve_n_choices(8), [2, 3, 4, 8])

    def test_unknown_bayes_backend_is_rejected(self):
        timeline = self.build_timeline()
        with self.assertRaisesRegex(ValueError, "Unknown bayes backend"):
            RetrainingPlanner(
                timeline,
                PrototypeClassifierAdapter(),
                config=PlannerConfig(bayes_backend="not-a-backend"),
            )

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
