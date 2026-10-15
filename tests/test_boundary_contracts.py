"""Boundary, reproducibility, and adapter contracts."""

from dataclasses import FrozenInstanceError
import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from retrainopt import PlannerConfig
from retrainopt.adapters import ArrayTimelineAdapter, BinaryTimelineAdapter, PrototypeModelAdapter
from retrainopt.cli import build_parser
from retrainopt.evaluation import EvaluationEngine
from retrainopt.core import surrogate as surrogate_module
from retrainopt.core.surrogate import NearestNeighborRegressor


class ConstantModel:
    def __init__(self, loss=0.25):
        self.loss = loss
        self.fit_calls = 0

    def fit(self, train_data, **kwargs):
        self.fit_calls += 1
        return object()

    def evaluate(self, model, eval_data, **kwargs):
        return self.loss


def timeline():
    return ArrayTimelineAdapter(
        [[((0.0,), 0), ((1.0,), 1)]], [[((0.1,), 0), ((0.9,), 1)]],
    )


class ConfigContracts(unittest.TestCase):
    def test_counts_and_fractions_are_unambiguous_and_below_minimum_stays_empty(self):
        self.assertEqual(PlannerConfig(n_choices=(1, 1.0)).resolve_n_choices(10), [1, 10])
        self.assertEqual(PlannerConfig(n_choices=(0.01,)).resolve_n_choices(10), [])
        self.assertEqual(PlannerConfig(n_choices=(1,), min_train_size=2).resolve_n_choices(10), [])
        self.assertEqual(PlannerConfig(n_choices=(1.0,), min_train_size=2).resolve_n_choices(1), [])

    def test_invalid_choices_are_rejected_without_fallback(self):
        for choices in ((), (True,), (0,), (-1,), (0.0,), (2.0,), (math.nan,), (math.inf,), ("0.5",)):
            with self.subTest(choices=choices), self.assertRaises(ValueError):
                PlannerConfig(n_choices=choices)

    def test_invalid_weights_and_budgets_are_rejected(self):
        for field in ("time_cost_weight", "loss_cost_weight", "initial_loss", "exploration_beta", "improvement_xi", "surrogate_uncertainty_floor"):
            for value in (-1, math.nan, math.inf, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    PlannerConfig(**{field: value})
        for field in ("min_train_size", "bayes_initial_samples", "bayes_candidates_per_iteration", "surrogate_k"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                PlannerConfig(**{field: 0})

    def test_config_snapshots_sequences_and_is_frozen(self):
        choices = [0.5]
        options = {"nested": [1]}
        config = PlannerConfig(n_choices=choices, bayes_backend_options=options)
        choices.append(1.0)
        options["nested"].append(2)
        self.assertEqual(config.n_choices, (0.5,))
        self.assertEqual(config.bayes_backend_options, {"nested": [1]})
        with self.assertRaises(FrozenInstanceError):
            config.time_cost_weight = 5

    def test_cli_preserves_count_and_fraction_types(self):
        args = build_parser().parse_args(["plan", "--dataset-dir", ".", "--n-choice", "1", "--n-choice", "1.0"])
        self.assertIs(type(args.n_choices[0]), int)
        self.assertIs(type(args.n_choices[1]), float)


class EvaluationContracts(unittest.TestCase):
    def test_invalid_initial_train_size_fails_before_training(self):
        for invalid in (True, 1.5, -1):
            data = timeline()
            data.train_size = lambda step: invalid
            adapter = ConstantModel()
            with self.subTest(size=invalid), self.assertRaisesRegex(ValueError, "train_size"):
                EvaluationEngine(data, adapter, PlannerConfig()).evaluate_initial_model()
            self.assertEqual(adapter.fit_calls, 0)

    def test_configured_initial_loss_prevents_implicit_training(self):
        adapter = ConstantModel()
        engine = EvaluationEngine(timeline(), adapter, PlannerConfig(initial_loss=0.8))
        result = engine.evaluate_initial_model()
        self.assertEqual(result.future_losses, {0: 0.8})
        self.assertEqual(adapter.fit_calls, 0)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            engine.evaluate_initial_model(object())

    def test_measured_losses_must_be_finite_and_nonnegative(self):
        for loss in (-1, math.inf, math.nan, True):
            with self.subTest(loss=loss):
                engine = EvaluationEngine(timeline(), ConstantModel(loss), PlannerConfig())
                with self.assertRaisesRegex(ValueError, "loss"):
                    engine.evaluate_initial_model()
                with self.assertRaisesRegex(ValueError, "loss"):
                    engine.evaluate_candidate(0, 1)
                self.assertEqual(list(engine.iter_cached_candidates()), [])

    def test_measured_time_is_validated(self):
        for end in (-1.0, math.inf, math.nan):
            engine = EvaluationEngine(timeline(), ConstantModel(), PlannerConfig())
            with self.subTest(end=end), patch("retrainopt.evaluation.time.perf_counter", side_effect=[0.0, end]):
                with self.assertRaisesRegex(ValueError, "training time"):
                    engine.evaluate_candidate(0, 1)

    def test_empty_timelines_and_evaluation_blocks_fail(self):
        for train, evaluation in (([], []), ([[]], [[]])):
            with self.assertRaises(ValueError):
                ArrayTimelineAdapter(train, evaluation)
        with self.assertRaises(ValueError):
            PrototypeModelAdapter().evaluate({}, [], step=0)

    def test_no_candidate_below_minimum_is_evaluated(self):
        engine = EvaluationEngine(timeline(), ConstantModel(), PlannerConfig(min_train_size=2))
        with self.assertRaisesRegex(ValueError, "min_train_size"):
            engine.evaluate_candidate(0, 1)


class BinaryContracts(unittest.TestCase):
    def write_pair(self, root, *, train_name="0", eval_name="0", record=(0.0, 0.0)):
        for directory, name in (("train", train_name), ("test", eval_name)):
            path = root / directory
            path.mkdir(exist_ok=True)
            (path / f"{name}.bin").write_bytes(struct.pack("<2f", *record))

    def test_identifiers_must_match_even_when_file_counts_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_pair(root, train_name="0", eval_name="1")
            with self.assertRaisesRegex(ValueError, "identifiers"):
                BinaryTimelineAdapter(root, num_features=1)

    def test_dimensions_features_and_labels_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_pair(root)
            for dimension in (0, -1, True, 1.5):
                with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                    BinaryTimelineAdapter(root, num_features=dimension)
            for record in ((math.nan, 0), (1, 0.5), (1, math.inf), (1, -1)):
                self.write_pair(root, record=record)
                with self.subTest(record=record), self.assertRaises(ValueError):
                    BinaryTimelineAdapter(root, num_features=1)
            self.write_pair(root)
            (root / "train" / "0.bin").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "layout"):
                BinaryTimelineAdapter(root, num_features=1)

    def test_ambiguous_numeric_identifiers_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_pair(root)
            self.write_pair(root, train_name="00", eval_name="00")
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                BinaryTimelineAdapter(root, num_features=1)


class SurrogateContracts(unittest.TestCase):
    def test_invalid_fit_shapes_and_values_fail_at_boundary(self):
        for rows, values in (([(1,)], []), ([(1,), (1, 2)], [1, 2]), ([(math.inf,)], [1]), ([(1,)], [math.nan])):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                NearestNeighborRegressor().fit(rows, values)

    @unittest.skipIf(surrogate_module._np is None, "NumPy is optional")
    def test_tied_neighbors_match_across_python_numpy_and_optional_index(self):
        rows = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-1.0, 0.0)]
        targets = [10.0, 20.0, 30.0, 40.0, 50.0]
        queries = [(0.0, 0.0), (-1.0, 0.0), (0.1, 0.2)]
        for k in (1, 2, 3):
            with patch.object(surrogate_module, "_np", None):
                pure = NearestNeighborRegressor(k_neighbors=k)
                pure.fit(rows, targets)
                expected = pure.predict_many(queries)
            accelerated = NearestNeighborRegressor(k_neighbors=k)
            accelerated.fit(rows, targets)
            indexed = accelerated.predict_many(queries)
            accelerated._neighbor_index = None
            numpy_results = accelerated.predict_many(queries)
            for result_set in (indexed, numpy_results, [accelerated.predict(row) for row in queries]):
                for actual, reference in zip(result_set, expected):
                    for value, target in zip(actual, reference):
                        self.assertAlmostEqual(value, target, places=10)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "sklearn is optional")
class SklearnContracts(unittest.TestCase):
    def test_estimator_clone_failures_propagate_without_copy_fallback(self):
        from retrainopt.adapters import SklearnModelAdapter
        original = RuntimeError("invalid estimator configuration")
        with patch("sklearn.base.clone", side_effect=original):
            with self.assertRaises(RuntimeError) as raised:
                SklearnModelAdapter(object()).fit([], step=0, n_samples=1, random_seed=7)
        self.assertIs(raised.exception, original)

    def test_seed_is_applied_only_when_estimator_has_no_explicit_seed(self):
        from sklearn.ensemble import RandomForestClassifier
        from retrainopt.adapters import SklearnModelAdapter
        data = [([0.0], 0), ([0.1], 0), ([0.9], 1), ([1.0], 1)]
        for explicit, expected in ((None, 7), (11, 11)):
            adapter = SklearnModelAdapter(RandomForestClassifier(n_estimators=2, random_state=explicit))
            model = adapter.fit(data, step=0, n_samples=4, random_seed=7)
            self.assertEqual(model.random_state, expected)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is optional")
class TorchContracts(unittest.TestCase):
    def test_training_is_seeded_and_preserves_callers_rng(self):
        import torch
        from retrainopt.adapters import TorchModelAdapter
        data = torch.utils.data.TensorDataset(torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1]))
        adapter = TorchModelAdapter(lambda: torch.nn.Linear(1, 2), epochs=1, batch_size=1)
        torch.manual_seed(91)
        before = torch.random.get_rng_state().clone()
        first = adapter.fit(data, step=0, n_samples=2, random_seed=7)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = adapter.fit(data, step=0, n_samples=2, random_seed=7)
        for left, right in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(left, right))
        adapter.evaluate(first, data, step=0)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))


@unittest.skipUnless(importlib.util.find_spec("botorch"), "BoTorch is optional")
class BoTorchContracts(unittest.TestCase):
    def test_gp_fitting_uses_local_rng(self):
        import torch
        from retrainopt.core.bayes_backends import _BoTorchRegressor
        sampled = []
        def fake_fit(mll):
            sampled.append(torch.rand(3))
            return mll
        torch.manual_seed(91)
        before = torch.random.get_rng_state().clone()
        with patch("botorch.fit.fit_gpytorch_mll", side_effect=fake_fit):
            _BoTorchRegressor([(0.0,), (1.0,)], [0.0, 1.0], random_seed=7)
            _BoTorchRegressor([(0.0,), (1.0,)], [0.0, 1.0], random_seed=7)
        self.assertTrue(torch.equal(sampled[0], sampled[1]))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
