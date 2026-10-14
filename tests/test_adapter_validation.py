"""Regression checks for classification labels and Prototype feature inputs."""

from importlib.util import find_spec
import math
import unittest
from unittest.mock import Mock

from retrainopt.adapters import PrototypeModelAdapter, SklearnModelAdapter


INVALID_LABELS = (0.2, -0.5, math.nan, math.inf, -math.inf, True, False, "1", None)


def fit(adapter, data):
    return adapter.fit(data, step=0, n_samples=len(data), random_seed=42)


class PrototypeValidationTests(unittest.TestCase):
    def test_fractional_or_other_invalid_labels_are_not_truncated_in_fit_or_evaluation(self):
        adapter = PrototypeModelAdapter()
        model = fit(adapter, [(0.0, 0), (1.0, 1)])
        for label in INVALID_LABELS:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "labels"):
                    fit(adapter, [(0.0, label)])
                with self.assertRaisesRegex(ValueError, "labels"):
                    adapter.evaluate(model, [(0.0, label)], step=0)

    def test_integer_valued_float_and_negative_classes_keep_their_identity(self):
        adapter = PrototypeModelAdapter()
        model = fit(adapter, [(0.0, -1.0), (1.0, 2)])
        self.assertEqual(set(model["prototypes"]), {-1, 2})
        self.assertEqual(adapter.evaluate(model, [(0.0, -1), (1.0, 2.0)], step=0), 0.0)

    @unittest.skipUnless(find_spec("numpy") is not None, "NumPy is optional")
    def test_numpy_integer_and_integer_valued_real_labels_are_supported(self):
        import numpy as np
        adapter = PrototypeModelAdapter()
        model = fit(adapter, [(np.float64(0.0), np.int64(-1)), ([np.float32(1.0)], np.uint32(2))])
        self.assertEqual(set(model["prototypes"]), {-1, 2})
        self.assertEqual(adapter.evaluate(model, [(0.0, np.float64(-1.0)), (1.0, np.int32(2))], step=0), 0.0)
        for label in (np.bool_(True), np.float64(0.2)):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "labels"):
                fit(adapter, [(0.0, label)])

    @unittest.skipUnless(find_spec("numpy") is not None, "NumPy is optional")
    def test_numpy_rows_remain_valid_labeled_pairs(self):
        import numpy as np
        data = list(np.array([[0.0, 0.0], [1.0, 1.0]]))
        adapter = PrototypeModelAdapter()
        model = fit(adapter, data)
        self.assertEqual(model["prototypes"], {0: (0.0,), 1: (1.0,)})
        self.assertEqual(adapter.evaluate(model, data, step=0), 0.0)

    def test_nonfinite_empty_nested_or_nonnumeric_features_raise_for_fit_and_evaluation(self):
        adapter = PrototypeModelAdapter()
        model = fit(adapter, [(0.0, 0), (1.0, 1)])
        invalid = (math.nan, math.inf, -math.inf, [], (), [math.nan], [math.inf], [[1.0]], "1", None, 1j)
        for features in invalid:
            with self.subTest(features=features):
                with self.assertRaisesRegex(ValueError, "features"):
                    fit(adapter, [(features, 0)])
                with self.assertRaisesRegex(ValueError, "features"):
                    adapter.evaluate(model, [(features, 0)], step=0)

    def test_training_dimension_is_checked_across_and_within_classes(self):
        for labels in ((0, 0), (0, 1)):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "dimension"):
                fit(PrototypeModelAdapter(), [([0.0], labels[0]), ([1.0, 2.0], labels[1])])

    def test_evaluation_dimension_must_match_training_in_both_directions(self):
        adapter = PrototypeModelAdapter()
        model = fit(adapter, [((0.0, 0.0), 0), ((1.0, 1.0), 1)])
        for features in ([0.0], [0.0, 0.0, 0.0]):
            with self.subTest(features=features), self.assertRaisesRegex(ValueError, "dimension"):
                adapter.evaluate(model, [(features, 0)], step=0)
        self.assertEqual(adapter.evaluate(model, [([0.0, 0.0], 0), ([1.0, 1.0], 1)], step=0), 0.0)

    def test_scalar_and_flat_vectors_keep_existing_classification_behavior(self):
        adapter = PrototypeModelAdapter()
        model = fit(adapter, [(False, 0), ([True], 1)])
        self.assertEqual(model["prototypes"], {0: (0.0,), 1: (1.0,)})
        self.assertEqual(adapter.evaluate(model, [((0.0,), 0), (1.0, 1)], step=0), 0.0)
        self.assertEqual(adapter.evaluate(model, [(0.0, 1), ([1.0], 1)], step=0), 0.5)

    def test_empty_training_baseline_is_preserved_but_eval_features_are_validated(self):
        adapter = PrototypeModelAdapter()
        model = fit(adapter, [])
        self.assertEqual(model, {"prototypes": {}, "majority": 0})
        self.assertEqual(adapter.evaluate(model, [(0.0, 0), (1.0, 1)], step=0), 0.5)
        with self.assertRaisesRegex(ValueError, "empty"):
            adapter.evaluate(model, [], step=0)
        with self.assertRaisesRegex(ValueError, "dimension"):
            adapter.evaluate(model, [([0.0], 0), ([1.0, 2.0], 1)], step=0)

    def test_malformed_labeled_rows_fail_consistently(self):
        adapter = PrototypeModelAdapter()
        model = fit(adapter, [(0.0, 0)])
        for row in ((0.0,), (0.0, 0, 1), None, "01", b"01", bytearray(b"01"), {0, 1}, frozenset({0, 1}), {0: 0, 1: 1}):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "labeled items"):
                    fit(adapter, [row])
                with self.assertRaisesRegex(ValueError, "labeled items"):
                    adapter.evaluate(model, [row], step=0)

    def test_overflow_does_not_create_nonfinite_prototypes_or_distances(self):
        adapter = PrototypeModelAdapter()
        with self.assertRaisesRegex(ValueError, "mean overflowed"):
            fit(adapter, [(1e308, 0), (1e308, 0)])
        model = fit(adapter, [(0.0, 0), (1.0, 1)])
        with self.assertRaisesRegex(ValueError, "distance overflowed"):
            adapter.evaluate(model, [(1e308, 0)], step=0)


@unittest.skipUnless(find_spec("sklearn") is not None, "sklearn is optional")
class SklearnLabelValidationTests(unittest.TestCase):
    def test_invalid_labels_fail_before_model_training_or_prediction(self):
        for label in INVALID_LABELS:
            with self.subTest(label=label):
                factory = Mock()
                adapter = SklearnModelAdapter(factory)
                with self.assertRaisesRegex(ValueError, "labels"):
                    fit(adapter, [([0.0], label)])
                factory.assert_not_called()
                model = Mock()
                with self.assertRaisesRegex(ValueError, "labels"):
                    adapter.evaluate(model, [([0.0], label)], step=0)
                model.predict.assert_not_called()
                model.predict_proba.assert_not_called()

    def test_integer_valued_labels_reach_estimator_without_changing_classes(self):
        from sklearn.tree import DecisionTreeClassifier
        adapter = SklearnModelAdapter(DecisionTreeClassifier())
        model = fit(adapter, [([0.0], -1.0), ([1.0], 2)])
        self.assertEqual(model.classes_.tolist(), [-1, 2])
        self.assertLess(adapter.evaluate(model, [([0.0], -1), ([1.0], 2.0)], step=0), 1e-12)

    @unittest.skipUnless(find_spec("numpy") is not None, "NumPy is optional")
    def test_numpy_integer_labels_keep_their_values(self):
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier
        adapter = SklearnModelAdapter(DecisionTreeClassifier())
        model = fit(adapter, [([0.0], np.int64(1)), ([1.0], np.int32(3))])
        self.assertEqual(model.classes_.tolist(), [1, 3])
        self.assertLess(adapter.evaluate(model, [([0.0], np.float64(1.0)), ([1.0], np.int64(3))], step=0), 1e-12)


if __name__ == "__main__":
    unittest.main()
