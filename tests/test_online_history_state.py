"""Historical windows preserve the deployed model's pre-window state."""

from itertools import cycle
import math
import unittest
from unittest.mock import Mock, patch

from retrainopt import (
    Decision,
    DecisionAction,
    OnlineController,
    OnlineState,
    PlannerConfig,
    collect_online_training_data,
)


KEEP = Decision(DecisionAction.KEEP)


class WindowTimeline:
    """Data retain their identity when a window is indexed from zero."""

    def __init__(self, data_steps, *, visible_through):
        self.data_steps = tuple(data_steps)
        self.visible_through = visible_through
        self.reads = []

    @property
    def num_steps(self):
        return len(self.data_steps)

    def _read(self, kind, step):
        if not 0 <= step <= self.visible_through:
            raise AssertionError(f"Read outside available history: {kind}({step})")
        self.reads.append((kind, step))
        return self.data_steps[step]

    def train_size(self, step):
        self._read("size", step)
        return 2

    def get_train_block(self, step):
        return [self._read("train", step)] * 2

    def sample_train_block(self, step, n_samples, *, seed):
        return [self._read("sample", step)] * n_samples

    def get_eval_block(self, step):
        return [self._read("eval", step)]


class WindowModel:
    losses = {
        "initial": {0: 0.5, 1: 0.75, 2: 0.125, 3: 0.75, 4: 0.875, 5: 0.9},
        ("trained", 3): {3: 1.0, 4: 1.0, 5: 1.0},
        ("trained", 4): {4: 0.25, 5: 0.125},
        ("trained", 5): {5: 0.6},
    }

    def __init__(self):
        self.fit_calls = []
        self.eval_calls = []

    def fit(self, train_data, *, step, n_samples, random_seed):
        self.fit_calls.append((step, n_samples))
        return ("trained", train_data[0])

    def evaluate(self, model, eval_data, *, step):
        self.eval_calls.append((model, step))
        return self.losses[model][eval_data[0]]


class ReplayPolicy:
    def __init__(self, decisions, *, offset=0):
        self.decisions = tuple(decisions)
        self.offset = offset

    def decide(self, state, *, train_size):
        index = state.step - self.offset
        decision = self.decisions[index] if index >= 0 else KEEP
        if decision.action == DecisionAction.RETRAIN:
            return Decision(DecisionAction.RETRAIN, decision.n_samples, state.step)
        return KEEP


def config():
    return PlannerConfig(
        n_choices=(2,), time_cost_weight=2.0, loss_cost_weight=2.0,
        bayes_initial_samples=3, bayes_iterations=0, bayes_anchor_steps=(),
    )


class OnlineHistoryStateContracts(unittest.TestCase):
    def collect(self, *, method="exact", **kwargs):
        timeline = WindowTimeline(range(3, 8), visible_through=2)
        model = WindowModel()
        with patch("retrainopt.evaluation.time.perf_counter", side_effect=cycle((10.0, 10.5))):
            data = collect_online_training_data(
                timeline, model, config(), history_steps=3, method=method, **kwargs,
            )
        return data, timeline, model

    def test_signed_history_matches_controller_replay_for_exact_and_bayes(self):
        for method in ("exact", "bayes"):
            with self.subTest(method=method):
                history, timeline, model = self.collect(
                    method=method, initial_model="initial", initial_reference_loss=0.5,
                    initial_model_age=3, initial_accumulated_staleness=-0.125,
                )
                self.assertEqual(history.decisions, (KEEP, Decision(DecisionAction.RETRAIN, 2, 1), KEEP))
                self.assertEqual(
                    [state.features() for state in history.states],
                    [[0.25, 0.125, 3.0], [0.375, 0.5, 4.0], [-0.125, -0.125, 1.0]],
                )
                self.assertEqual(history.plan.total_cost, 3.25)
                self.assertEqual(history.plan.training_cost, 1.0)
                self.assertEqual(history.plan.loss_cost, 1.125)
                self.assertEqual(sorted(model.fit_calls), [(0, 2), (1, 2), (2, 2)])
                self.assertEqual(len(model.eval_calls), 9)
                self.assertTrue(all(step < 3 for _, step in timeline.reads))

                replay_timeline = WindowTimeline(range(3, 8), visible_through=0)
                replay = OnlineController(
                    replay_timeline, WindowModel(), ReplayPolicy(history.decisions), config(),
                    initial_model="initial", initial_reference_loss=0.5,
                    initial_trained_at=-3, initial_accumulated_staleness=-0.125,
                )
                results = []
                with patch("retrainopt.online.time.perf_counter", side_effect=cycle((10.0, 10.5))):
                    for step, expected in enumerate(history.states):
                        replay_timeline.visible_through = step
                        result = replay.step()
                        self.assertEqual(result.state, expected)
                        self.assertEqual(result.decision, history.decisions[step])
                        results.append(result)
                self.assertEqual(sum(row.training_cost for row in results), history.plan.training_cost)
                self.assertEqual(sum(row.loss for row in results), history.plan.loss_cost)
                self.assertEqual(sum(row.total_cost for row in results), history.plan.total_cost)
                self.assertEqual(replay.last_retrain_step, 1)
                self.assertEqual(replay.reference_loss, 0.25)
                self.assertEqual(replay.accumulated_staleness, -0.125)

    def test_reindexed_window_matches_continuous_run_for_same_actions(self):
        history, _, _ = self.collect(
            initial_model="initial", initial_reference_loss=0.5,
            initial_model_age=3, initial_accumulated_staleness=-0.125,
        )
        continuous = OnlineController(
            WindowTimeline(range(6), visible_through=5), WindowModel(),
            ReplayPolicy(history.decisions, offset=3), config(),
            initial_model="initial", initial_reference_loss=0.5,
        )
        for _ in range(3):
            continuous.step()
        self.assertEqual(continuous.accumulated_staleness, -0.125)
        self.assertEqual(continuous.next_step - continuous.last_retrain_step, 3)
        window = OnlineController(
            WindowTimeline(range(3, 6), visible_through=2), WindowModel(),
            ReplayPolicy(history.decisions), config(), initial_model=continuous.current_model,
            initial_reference_loss=continuous.reference_loss,
            initial_trained_at=continuous.last_retrain_step - continuous.next_step,
            initial_accumulated_staleness=continuous.accumulated_staleness,
        )
        with patch("retrainopt.online.time.perf_counter", side_effect=cycle((10.0, 10.5))):
            for _ in range(3):
                full_result, window_result = continuous.step(), window.step()
                self.assertEqual(full_result.state.features(), window_result.state.features())
                self.assertEqual(full_result.state.step, window_result.state.step + 3)
                self.assertEqual(full_result.decision.action, window_result.decision.action)
                self.assertEqual(full_result.decision.n_samples, window_result.decision.n_samples)
                for field in ("loss", "training_seconds", "training_cost", "weighted_loss_cost", "total_cost"):
                    self.assertEqual(getattr(full_result, field), getattr(window_result, field))
                self.assertEqual(continuous.current_model, window.current_model)
                self.assertEqual(continuous.reference_loss, window.reference_loss)
                self.assertEqual(continuous.accumulated_staleness, window.accumulated_staleness)
                self.assertEqual(continuous.last_retrain_step, window.last_retrain_step + 3)

    def test_zero_defaults_preserve_existing_teacher_behavior(self):
        default, _, _ = self.collect(initial_model="initial", initial_reference_loss=0.5)
        explicit, _, _ = self.collect(
            initial_model="initial", initial_reference_loss=0.5,
            initial_model_age=0, initial_accumulated_staleness=0.0,
        )
        self.assertEqual(default.states, explicit.states)
        self.assertEqual(default.decisions, explicit.decisions)
        self.assertEqual(default.plan.total_cost, explicit.plan.total_cost)
        self.assertEqual(default.states[0].features(), [0.25, 0.25, 0.0])
        bootstrap, _, model = self.collect()
        self.assertEqual(bootstrap.states[0].features(), [0.0, 0.0, 0.0])
        self.assertEqual(model.fit_calls.count((0, 2)), 2)

    def test_zero_time_weight_removes_estimated_fit_cost_in_historical_bayes(self):
        class NoisyCostSurrogate:
            def predict_training_cost(self, step, n_samples, dataset_size):
                return (99.0, 3.0)

            def predict_loss(self, step, n_samples, dataset_size, eval_step):
                return (0.0, 0.0)

        # Only the middle candidate is measured initially. The other fits
        # have known zero cost, regardless of the surrogate's cost moments.
        backend = Mock()
        backend.fit.return_value = NoisyCostSurrogate()
        model = WindowModel()
        with patch("retrainopt.planner.create_bayes_backend", return_value=backend):
            history = collect_online_training_data(
                WindowTimeline(range(3, 8), visible_through=2), model,
                PlannerConfig(
                    n_choices=(2,), time_cost_weight=0, bayes_initial_samples=1,
                    bayes_iterations=0, bayes_anchor_steps=(0.5,),
                ),
                history_steps=3, method="bayes", initial_model="initial",
                initial_reference_loss=0.5,
            )
        self.assertEqual(history.plan.metadata["bayes_initial_candidates"], [(1, 2)])
        # The zero-cost optimistic proposal causes step 0 to be measured even
        # though the surrogate predicted a fit cost of 99. Its measured loss is
        # poor, so final optimization retains the initial model until step 1.
        self.assertEqual(history.plan.metadata["finalization_new_candidates"], [(0, 2)])
        self.assertEqual(history.plan.estimated_total_cost, 1.125)
        self.assertEqual(history.plan.total_cost, 1.125)
        self.assertEqual(history.decisions, (KEEP, Decision(DecisionAction.RETRAIN, 2, 1), KEEP))
        self.assertEqual(model.fit_calls, [(1, 2), (0, 2)])
        self.assertEqual(history.plan.training_cost, 0.0)

    def test_synthetic_baseline_accepts_explicit_age_and_signed_carry(self):
        for accumulated in (-0.25, 0.25):
            with self.subTest(accumulated=accumulated):
                model = WindowModel()
                history = collect_online_training_data(
                    WindowTimeline(range(3, 5), visible_through=1), model,
                    PlannerConfig(initial_loss=0.5, n_choices=(2,), min_train_size=3),
                    history_steps=2, initial_model_age=5,
                    initial_accumulated_staleness=accumulated,
                )
                self.assertEqual(history.decisions, (KEEP, KEEP))
                self.assertEqual(
                    [state.features() for state in history.states],
                    [[0.0, accumulated, 5.0], [0.0, accumulated, 6.0]],
                )
                self.assertEqual(model.fit_calls, [])
                self.assertEqual(model.eval_calls, [])

    def test_invalid_history_state_fails_before_training_or_evaluation(self):
        invalid = [
            ("initial_model_age", value) for value in (-1, True, 0.5, "3", math.nan, math.inf)
        ] + [
            ("initial_accumulated_staleness", value)
            for value in (True, "0.0", math.nan, math.inf, -math.inf)
        ]
        for parameter, value in invalid:
            with self.subTest(parameter=parameter, value=value):
                model = WindowModel()
                with self.assertRaises(ValueError):
                    collect_online_training_data(
                        WindowTimeline(range(3, 6), visible_through=2), model, config(),
                        history_steps=3, initial_model="initial", initial_reference_loss=0.5,
                        **{parameter: value},
                    )
                self.assertEqual(model.fit_calls, [])
                self.assertEqual(model.eval_calls, [])

    def test_implicit_bootstrap_rejects_nonzero_history_before_fitting(self):
        for kwargs in (
            {"initial_model_age": 1},
            {"initial_accumulated_staleness": 0.25},
            {"initial_accumulated_staleness": -0.25},
        ):
            with self.subTest(kwargs=kwargs):
                model = WindowModel()
                with self.assertRaises(ValueError):
                    collect_online_training_data(
                        WindowTimeline(range(3, 6), visible_through=2), model, config(),
                        history_steps=3, **kwargs,
                    )
                self.assertEqual(model.fit_calls, [])
                self.assertEqual(model.eval_calls, [])

    def test_window_clock_accepts_old_models_but_rejects_invalid_timestamps(self):
        self.assertEqual(OnlineState(0, 0.25, -0.125, 7).features(), [0.25, -0.125, 7.0])
        for trained_at in (True, -1.5, "-1", math.nan, math.inf, 1):
            with self.subTest(trained_at=trained_at), self.assertRaises(ValueError):
                OnlineController(
                    WindowTimeline(range(3, 6), visible_through=0), WindowModel(),
                    ReplayPolicy((KEEP,)), initial_model="initial", initial_reference_loss=0.5,
                    initial_trained_at=trained_at,
                )


if __name__ == "__main__":
    unittest.main()
