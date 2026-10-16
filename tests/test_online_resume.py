"""Deployment resumption preserves the policy's signed historical state."""

from dataclasses import replace
from itertools import cycle
import unittest
from unittest.mock import patch

from retrainopt import (
    BaseOnlinePolicy,
    Decision,
    DecisionAction,
    OnlineController,
    PlannerConfig,
)
from retrainopt.adapters import ArrayTimelineAdapter


class AccumulatedStalenessPolicy:
    def decide(self, state, *, train_size):
        if state.accumulated_staleness >= 0.25 and state.time_since_last_retrain >= 3:
            return Decision(DecisionAction.RETRAIN, 2, state.step)
        return Decision(DecisionAction.KEEP)


class LossTableAdapter:
    losses = {
        "initial": {0: 0.5, 1: 0.75, 2: 0.125, 3: 0.75, 4: 0.875},
        (4, 2): {4: 0.25, 5: 0.125, 6: 0.5, 7: 0.5},
        (7, 2): {7: 0.125},
    }

    def fit(self, train_data, *, step, n_samples, random_seed):
        return (step, n_samples)

    def evaluate(self, model, eval_data, *, step):
        return self.losses[model][step]


def deployment_state(controller):
    return (
        controller.current_model,
        controller.reference_loss,
        controller.last_retrain_step,
        controller.next_step,
        controller.accumulated_staleness,
    )


class OnlineResumeContracts(unittest.TestCase):
    def test_resume_synthetic_baseline_before_and_after_first_retrain(self):
        class ConstantLossAdapter:
            def fit(self, train_data, *, step, n_samples, random_seed):
                return (step, n_samples)

            def evaluate(self, model, eval_data, *, step):
                return 0.125

        block = [((0.0,), 0), ((1.0,), 1)]
        timeline = ArrayTimelineAdapter([block] * 4, [block] * 4)
        config = PlannerConfig(initial_loss=0.5, time_cost_weight=3.0, loss_cost_weight=2.0)
        policy = BaseOnlinePolicy(period=2, n_samples=2)
        continuous = OnlineController(
            timeline, ConstantLossAdapter(), policy, config,
            initial_trained_at=-1, initial_accumulated_staleness=-0.125,
        )
        results, checkpoints = [], []
        with patch("retrainopt.online.time.perf_counter", side_effect=cycle((20.0, 20.5))):
            for _ in range(4):
                results.append(continuous.step())
                checkpoints.append(deployment_state(continuous))
        self.assertEqual(checkpoints[0], (None, 0.5, -1, 1, -0.125))
        self.assertEqual(checkpoints[1], ((1, 2), 0.125, 1, 2, 0.0))
        self.assertEqual([result.total_cost for result in results], [1.0, 1.75, 0.25, 1.75])

        for start_step in (1, 2):
            with self.subTest(start_step=start_step):
                model, reference, trained_at, next_step, accumulated = checkpoints[start_step - 1]
                resume_config = replace(config, initial_loss=None) if model is not None else config
                resumed = OnlineController(
                    timeline, ConstantLossAdapter(), policy, resume_config,
                    initial_model=model,
                    initial_reference_loss=reference if model is not None else None,
                    start_step=next_step, initial_trained_at=trained_at,
                    initial_accumulated_staleness=accumulated,
                )
                self.assertEqual(deployment_state(resumed), checkpoints[start_step - 1])
                with patch("retrainopt.online.time.perf_counter", side_effect=cycle((20.0, 20.5))):
                    for step in range(start_step, 4):
                        self.assertEqual(resumed.step(), results[step])
                        self.assertEqual(deployment_state(resumed), checkpoints[step])
        self.assertEqual(config.initial_loss, 0.5)

    def test_nonzero_signed_carry_matches_continuous_actions_features_costs_and_state(self):
        block = [((0.0,), 0), ((1.0,), 1)]
        timeline = ArrayTimelineAdapter(train_blocks=[block] * 8, eval_blocks=[block] * 8)
        config = PlannerConfig(time_cost_weight=3.0, loss_cost_weight=2.0)
        continuous = OnlineController(
            timeline, LossTableAdapter(), AccumulatedStalenessPolicy(), config,
            initial_model="initial", initial_reference_loss=0.5,
        )
        # Binary fractions keep the hand-calculated expectations exact. The
        # negative contribution at step 2 must survive the first checkpoint:
        # resetting it would incorrectly trigger RETRAIN at step 3.
        expected_features = (
            (0.0, 0.0, 0),
            (0.25, 0.25, 1),
            (-0.375, -0.125, 2),
            (0.25, 0.125, 3),
            (0.375, 0.5, 4),
            (-0.125, -0.125, 1),
            (0.25, 0.125, 2),
            (0.25, 0.375, 3),
        )
        expected_losses = (0.5, 0.75, 0.125, 0.75, 0.25, 0.125, 0.5, 0.125)
        expected_costs = (1.0, 1.5, 0.25, 1.5, 2.0, 0.25, 1.0, 1.75)
        results, checkpoints = [], []
        with patch("retrainopt.online.time.perf_counter", side_effect=cycle((20.0, 20.5))) as timer:
            for step in range(8):
                result = continuous.step()
                retrained = step in (4, 7)
                expected_decision = (
                    Decision(DecisionAction.RETRAIN, 2, step)
                    if retrained else Decision(DecisionAction.KEEP)
                )
                self.assertEqual(tuple(result.state.features()), expected_features[step])
                self.assertEqual(result.decision, expected_decision)
                self.assertEqual(result.loss, expected_losses[step])
                self.assertEqual(result.training_seconds, 0.5 if retrained else 0.0)
                self.assertEqual(result.training_cost, 1.5 if retrained else 0.0)
                self.assertEqual(result.weighted_loss_cost, 2.0 * expected_losses[step])
                self.assertEqual(result.total_cost, expected_costs[step])
                results.append(result)
                checkpoints.append(deployment_state(continuous))
            self.assertEqual(timer.call_count, 4)
        self.assertEqual(sum(result.total_cost for result in results), 9.25)
        self.assertEqual(checkpoints[2], ("initial", 0.5, 0, 3, -0.125))
        self.assertEqual(checkpoints[6], ((4, 2), 0.25, 4, 7, 0.125))
        self.assertEqual(checkpoints[7], ((7, 2), 0.125, 7, 8, 0.0))

        # Restore both negative and positive cumulative history. The second
        # checkpoint also carries a different model, reference loss and clock.
        for start_step in (3, 7):
            with self.subTest(start_step=start_step):
                model, reference, trained_at, next_step, accumulated = checkpoints[start_step - 1]
                resumed = OnlineController(
                    timeline, LossTableAdapter(), AccumulatedStalenessPolicy(), config,
                    initial_model=model, initial_reference_loss=reference,
                    start_step=next_step, initial_trained_at=trained_at,
                    initial_accumulated_staleness=accumulated,
                )
                self.assertEqual(deployment_state(resumed), checkpoints[start_step - 1])
                resumed_cost = 0.0
                with patch("retrainopt.online.time.perf_counter", side_effect=cycle((20.0, 20.5))) as timer:
                    for step in range(start_step, 8):
                        result = resumed.step()
                        self.assertEqual(result, results[step])
                        self.assertEqual(deployment_state(resumed), checkpoints[step])
                        self.assertEqual(
                            resumed.next_step - resumed.last_retrain_step,
                            checkpoints[step][3] - checkpoints[step][2],
                        )
                        resumed_cost += result.total_cost
                    self.assertEqual(timer.call_count, 2 * sum(step >= start_step for step in (4, 7)))
                self.assertEqual(resumed_cost, sum(expected_costs[start_step:]))


if __name__ == "__main__":
    unittest.main()
