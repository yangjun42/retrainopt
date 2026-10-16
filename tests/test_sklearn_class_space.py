"""Class-space regressions for small candidates and later evaluation labels."""

from importlib.util import find_spec
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from retrainopt import PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter, SklearnModelAdapter


@unittest.skipUnless(find_spec("sklearn") is not None, "sklearn is optional")
class SklearnClassSpaceTests(unittest.TestCase):
    def test_declared_classes_are_validated_and_snapshotted(self):
        for classes in ([], [0], [0, 0], [0, 0.0], [0, True], [0, 0.5], [0, math.inf], [0, "1"]):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                SklearnModelAdapter(Mock(), classes=classes)
        classes = [2.0, -1]
        adapter = SklearnModelAdapter(Mock(), classes=classes)
        classes.append(3)
        self.assertEqual(adapter.classes, (-1, 2))

    def test_single_class_forest_scores_declared_but_unseen_class(self):
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        adapter = SklearnModelAdapter(RandomForestClassifier(n_estimators=2), classes=[0, 1])
        model = adapter.fit([([0.0], 1)], step=0, n_samples=1, random_seed=7)
        self.assertEqual(model.classes_.tolist(), [1])
        loss = adapter.evaluate(model, [([0.0], 1), ([0.0], 0)], step=1)
        self.assertAlmostEqual(loss, -math.log(np.finfo(float).eps) / 2, places=12)

    def test_probability_columns_follow_labels_and_missing_classes_get_zero(self):
        import numpy as np

        model = SimpleNamespace(
            classes_=[9, -3],
            predict_proba=Mock(return_value=np.array([[0.75, 0.25]] * 3)),
        )
        adapter = SklearnModelAdapter(Mock(), classes=[9, 5, -3])
        loss = adapter.evaluate(model, [([0.0], -3), ([0.0], 5), ([0.0], 9)], step=2)
        expected = (-math.log(0.25) - math.log(np.finfo(float).eps) - math.log(0.75)) / 3
        self.assertAlmostEqual(loss, expected, places=12)

    def test_undeclared_evaluation_classes_fail_before_version_dependent_log_loss(self):
        for declared in (None, [0, 1]):
            with self.subTest(classes=declared):
                model = SimpleNamespace(classes_=[0, 1], predict_proba=Mock())
                adapter = SklearnModelAdapter(Mock(), classes=declared)
                with patch("sklearn.metrics.log_loss") as metric:
                    with self.assertRaisesRegex(ValueError, "Evaluation labels.*outside the class space"):
                        adapter.evaluate(model, [([0.0], 0), ([0.0], 1), ([0.0], 2)], step=2)
                    metric.assert_not_called()
                model.predict_proba.assert_not_called()

    def test_single_class_model_without_declaration_has_actionable_error(self):
        model = SimpleNamespace(classes_=[1], predict_proba=Mock())
        with self.assertRaisesRegex(ValueError, "single-class model.*explicit classes"):
            SklearnModelAdapter(Mock()).evaluate(model, [([0.0], 1)], step=1)
        model.predict_proba.assert_not_called()

    def test_single_class_eval_block_is_valid_for_two_class_model(self):
        model = SimpleNamespace(classes_=[0, 1], predict_proba=Mock(return_value=[[0.75, 0.25]]))
        loss = SklearnModelAdapter(Mock()).evaluate(model, [([0.0], 0)], step=1)
        self.assertAlmostEqual(loss, -math.log(0.75))

    def test_training_and_external_model_cannot_expand_declared_classes(self):
        factory = Mock()
        adapter = SklearnModelAdapter(factory, classes=[0, 1])
        with self.assertRaisesRegex(ValueError, "Training labels.*outside the class space"):
            adapter.fit([([0.0], 2)], step=1, n_samples=1, random_seed=7)
        factory.assert_not_called()
        model = SimpleNamespace(classes_=[0, 2], predict_proba=Mock())
        with self.assertRaisesRegex(ValueError, "Model labels.*outside the class space"):
            adapter.evaluate(model, [([0.0], 0)], step=1)
        model.predict_proba.assert_not_called()

    def test_probability_shape_must_match_model_class_columns(self):
        model = SimpleNamespace(classes_=[0, 1], predict_proba=Mock(return_value=[[0.2, 0.3, 0.5]]))
        with self.assertRaisesRegex(ValueError, "one column per model class"):
            SklearnModelAdapter(Mock(), classes=[0, 1, 2]).evaluate(model, [([0.0], 0)], step=1)

    def test_classifier_without_probabilities_retains_error_rate(self):
        model = SimpleNamespace(predict=Mock(return_value=[0, 0]))
        adapter = SklearnModelAdapter(Mock(), classes=[0, 1])
        self.assertEqual(adapter.evaluate(model, [([0.0], 0), ([1.0], 1)], step=1), 0.5)
        with self.assertRaisesRegex(ValueError, "outside the class space"):
            adapter.evaluate(model, [([0.0], 2)], step=1)

    def test_logistic_single_class_fit_keeps_cause_and_candidate_context(self):
        from sklearn.linear_model import LogisticRegression

        adapter = SklearnModelAdapter(LogisticRegression(), classes=[0, 1])
        with self.assertRaisesRegex(ValueError, "step=3, n_samples=1.*training candidates or timeline sampling") as raised:
            adapter.fit([([0.0], 1)], step=3, n_samples=1, random_seed=7)
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertIn("at least 2 classes", str(raised.exception.__cause__))

    def test_other_estimator_failures_propagate_unchanged(self):
        for error, labels in ((RuntimeError("training failed"), [0]), (ValueError("bad features"), [0, 1])):
            with self.subTest(error=error, labels=labels):
                estimator = Mock()
                estimator.get_params.return_value = {}
                estimator.fit.side_effect = error
                adapter = SklearnModelAdapter(lambda: estimator, classes=[0, 1])
                with self.assertRaises(type(error)) as raised:
                    adapter.fit([([0.0], label) for label in labels], step=1, n_samples=len(labels), random_seed=7)
                self.assertIs(raised.exception, error)

    def test_exact_and_bayes_plan_with_default_small_candidates_and_new_declared_class(self):
        from sklearn.ensemble import RandomForestClassifier

        first = [([-2.0], 0), ([-1.0], 0), ([1.0], 1), ([2.0], 1)]
        later = first[:2] + [([1.0], 2), ([2.0], 2)]
        timeline = ArrayTimelineAdapter([first, later], [first, later])
        for method in ("exact", "bayes"):
            with self.subTest(method=method):
                planner = RetrainingPlanner(
                    timeline,
                    SklearnModelAdapter(RandomForestClassifier(n_estimators=2), classes=[0, 1, 2]),
                    PlannerConfig(time_cost_weight=0.0),
                )
                result = planner.plan(method)
                self.assertTrue(math.isfinite(result.total_cost))
                self.assertEqual(len(result.decisions), 2)
                candidates = {item.model_id: item for item in planner.engine.iter_cached_candidates()}
                self.assertIn((0, 1), candidates)
                self.assertIn((1, 1), candidates)
                self.assertTrue(all(math.isfinite(loss) for item in candidates.values() for loss in item.future_losses.values()))


if __name__ == "__main__":
    unittest.main()
