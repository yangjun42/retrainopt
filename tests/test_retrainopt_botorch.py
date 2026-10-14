"""Integration tests for the optional BoTorch backend."""

from __future__ import annotations

import math
import sys
import time
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from botorch.exceptions.warnings import InputDataWarning

    BOTORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    InputDataWarning = Warning
    BOTORCH_AVAILABLE = False

from retrainopt import PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter
from retrainopt.core.bayes_backends import _BoTorchRegressor
from retrainopt.interfaces import ModelAdapter


class TimedPrototypeClassifierAdapter(ModelAdapter):
    """Small dependency-free classifier with measurable training-time variation."""

    def fit(self, train_data, *, step, n_samples, random_seed):
        time.sleep(0.002 * max(1, int(n_samples)))
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


@unittest.skipUnless(BOTORCH_AVAILABLE, "BoTorch extra is not installed.")
class RetrainOptBoTorchTests(unittest.TestCase):
    def test_botorch_backend_runs_end_to_end(self):
        from botorch.fit import fit_gpytorch_mll

        timeline = ArrayTimelineAdapter(
            train_blocks=[
                [((0.0,), 0), ((0.2,), 0), ((0.8,), 1), ((1.0,), 1)],
                [((0.1,), 0), ((0.3,), 0), ((0.7,), 1), ((0.9,), 1)],
                [((0.0,), 1), ((0.2,), 1), ((0.8,), 0), ((1.0,), 0)],
            ],
            eval_blocks=[
                [((0.05,), 0), ((0.95,), 1)],
                [((0.15,), 0), ((0.85,), 1)],
                [((0.05,), 1), ((0.95,), 0)],
            ],
        )
        planner = RetrainingPlanner(
            timeline,
            TimedPrototypeClassifierAdapter(),
            config=PlannerConfig(
                n_choices=(0.5, 1.0),
                min_train_size=1,
                time_cost_weight=50.0,
                loss_cost_weight=1.0,
                random_seed=7,
                bayes_iterations=1,
                bayes_initial_samples=2,
                # Default anchors would measure all six candidates before search.
                bayes_anchor_steps=(),
                bayes_candidates_per_iteration=1,
                bayes_backend="botorch",
            ),
        )

        predictions = []
        original_predict = _BoTorchRegressor.predict

        def record_prediction(regressor, row):
            moments = original_predict(regressor, row)
            predictions.append((len(row), moments))
            return moments

        # These spies delegate to the actual optimizer and posterior prediction.
        with warnings.catch_warnings(), patch(
            "botorch.fit.fit_gpytorch_mll", wraps=fit_gpytorch_mll
        ) as fit_gp, patch.object(
            _BoTorchRegressor, "predict", autospec=True, side_effect=record_prediction
        ) as predict:
            warnings.simplefilter("ignore", category=InputDataWarning)
            result = planner.plan(method="bayes")

        self.assertGreaterEqual(fit_gp.call_count, 2)  # Training-cost and loss GPs.
        self.assertGreater(predict.call_count, 0)
        self.assertEqual(len(result.metadata["bayes_initial_candidates"]), 2)
        self.assertEqual(result.metadata["candidate_count"], 6)
        self.assertEqual(result.metadata["bayes_iterations_completed"], 1)
        self.assertLess(
            result.metadata["surrogate_evaluated_candidates"],
            result.metadata["candidate_count"],
        )
        # Both surrogates must predict unobserved candidates with real uncertainty.
        self.assertEqual({dimension for dimension, _ in predictions}, {4, 6})
        for dimension in (4, 6):
            moments = [value for width, value in predictions if width == dimension]
            self.assertTrue(all(
                math.isfinite(mean) and math.isfinite(std) and std >= 0.0
                for mean, std in moments
            ))
            self.assertTrue(any(std > 0.0 for _, std in moments))

        self.assertEqual(result.method, "bayes")
        self.assertEqual(result.metadata["bayes_backend"], "botorch")
        self.assertTrue(result.metadata["selected_sequence_evaluated_exactly"])
        self.assertEqual(len(result.decisions), timeline.num_steps)
        self.assertGreater(result.metadata["evaluated_candidates"], 2)
        self.assertTrue(math.isfinite(result.total_cost))
        self.assertTrue(math.isfinite(result.estimated_total_cost))
        self.assertAlmostEqual(result.total_cost, result.training_cost + result.weighted_loss_cost)


class BoTorchFallbackTests(unittest.TestCase):
    def test_one_row_returns_constant_mean_and_zero_uncertainty(self):
        # The existing one-row fallback does not need the optional dependencies.
        regressor = _BoTorchRegressor([(1.0, 2.0)], [3.25])

        self.assertEqual(regressor.predict((1.0, 2.0)), (3.25, 0.0))
        self.assertEqual(regressor.predict((100.0, -20.0)), (3.25, 0.0))


if __name__ == "__main__":
    unittest.main()
