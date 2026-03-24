"""Offline invariant tests ported from the legacy experiment repository."""

from __future__ import annotations

import math
import unittest

from retrainopt.core.acquisition import AcquisitionManager
from retrainopt.core.costs import compute_step_cost, compute_training_cost


class RetrainOptOfflineInvariantTests(unittest.TestCase):
    def test_cost_composition_matches_weighted_sum(self):
        training_cost = compute_training_cost(0.5, time_cost_weight=2.0)
        self.assertAlmostEqual(training_cost, 1.0, places=12)

        step_cost = compute_step_cost(training_cost, 0.4, loss_cost_weight=3.0)
        self.assertAlmostEqual(step_cost, 2.2, places=12)

    def test_acquisition_formulas_match_expected_values(self):
        manager = AcquisitionManager()
        mean = 2.2
        std = 0.25
        best_cost = 2.0
        beta = 1.5
        xi = 0.1

        self.assertAlmostEqual(manager.score(mean, std, "expected"), mean, places=12)
        self.assertAlmostEqual(manager.score(mean, std, "variance"), -(std ** 2), places=12)
        self.assertAlmostEqual(manager.score(mean, std, "lcb_beta_1.5"), mean - beta * std, places=12)
        self.assertAlmostEqual(manager.score(mean, std, "ucb_beta_1.5"), mean + beta * std, places=12)

        z_value = (best_cost - mean - xi) / std
        normal_cdf = 0.5 * (1.0 + math.erf(z_value / math.sqrt(2.0)))
        normal_pdf = math.exp(-0.5 * z_value * z_value) / math.sqrt(2.0 * math.pi)
        expected_improvement = (best_cost - mean - xi) * normal_cdf + std * normal_pdf
        probability_improvement = normal_cdf

        self.assertAlmostEqual(
            manager.score(mean, std, "ei_xi_0.1", best_cost=best_cost),
            -max(expected_improvement, 0.0),
            places=12,
        )
        self.assertAlmostEqual(
            manager.score(mean, std, "pi_xi_0.1", best_cost=best_cost),
            -probability_improvement,
            places=12,
        )

    def test_argmin_sign_convention_matches_paper_search(self):
        manager = AcquisitionManager()

        lower_variance_score = manager.score(1.0, 0.1, "variance")
        higher_variance_score = manager.score(1.0, 0.4, "variance")
        self.assertLess(higher_variance_score, lower_variance_score)

        worse_mean_better_ei = manager.score(1.8, 0.2, "ei_xi_0.01", best_cost=2.0)
        better_mean_worse_ei = manager.score(2.4, 0.2, "ei_xi_0.01", best_cost=2.0)
        self.assertLess(worse_mean_better_ei, better_mean_worse_ei)


if __name__ == "__main__":
    unittest.main()
