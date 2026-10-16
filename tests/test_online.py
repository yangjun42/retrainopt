"""Independent Section 6 examples and causal streaming contract checks."""

from __future__ import annotations

from importlib.util import find_spec
import math
import unittest
from unittest.mock import Mock, patch

from retrainopt import (
    BaseOnlinePolicy,
    Decision,
    DecisionAction,
    LearnedOnlinePolicy,
    OnlineController,
    OnlinePolicy,
    OnlineState,
    PlannerConfig,
    PlanResult,
    RetrainingPlanner,
    collect_online_training_data,
)


KEEP = Decision(DecisionAction.KEEP)


def retrain(n=2):
    return Decision(DecisionAction.RETRAIN, n_samples=n)


def plan_with(decisions):
    return PlanResult("exact", list(decisions), 0.0, 0.0, 0.0)


def features(state):
    return (
        state.current_staleness,
        state.accumulated_staleness,
        state.time_since_last_retrain,
    )


class GuardedTimeline:
    """Expose the currently observed prefix, and fail on any future read."""

    def __init__(self, steps=5, *, visible_through=0, train_size=4):
        self.num_steps = steps
        self.visible_through = visible_through
        self.size = train_size
        self.reads = []

    def _read(self, kind, step):
        if not 0 <= step <= self.visible_through:
            raise AssertionError(f"Future {kind} read at {step}")
        self.reads.append((kind, step))

    def train_size(self, step):
        self._read("size", step)
        return self.size

    def get_train_block(self, step):
        self._read("train", step)
        return [step] * self.size

    def get_eval_block(self, step):
        self._read("eval", step)
        return [step]

    def sample_train_block(self, step, n_samples, *, seed):
        self._read("sample", step)
        return [step] * n_samples


class TableModel:
    def __init__(self, losses):
        self.losses = losses
        self.fit_calls = []
        self.eval_calls = []
        self.fail_fit = False
        self.fail_new_evaluation = False

    def fit(self, train_data, *, step, n_samples, random_seed):
        self.fit_calls.append((step, n_samples))
        if self.fail_fit:
            raise RuntimeError("candidate fit failed")
        return (step, n_samples)

    def evaluate(self, model, eval_data, *, step):
        self.eval_calls.append((model, step))
        if self.fail_new_evaluation and model != "initial":
            raise RuntimeError("candidate evaluation failed")
        return self.losses[model][step]


class RecordingPolicy:
    def __init__(self, decisions=None):
        self.decisions = decisions or {}
        self.states = []

    def decide(self, state, *, train_size):
        self.states.append(state)
        return self.decisions.get(state.step, KEEP)


class OnlinePaperContracts(unittest.TestCase):
    def assertFeatures(self, state, expected):
        for actual, value in zip(features(state), expected):
            self.assertAlmostEqual(actual, value, places=12)

    def test_base_uses_horizon_over_retrain_count_and_mean_volume(self):
        # Unequal gaps intentionally distinguish T/R from mean observed gaps.
        decisions = [KEEP] * 10
        for step, n in ((1, 2), (2, 4), (8, 6)):
            decisions[step] = retrain(n)
        policy = BaseOnlinePolicy.from_plan(plan_with(decisions))
        self.assertAlmostEqual(policy.period, 10 / 3)
        self.assertEqual(policy.n_samples, 4)
        self.assertEqual(policy.decide(OnlineState(9, 0, 0, 3), train_size=10), KEEP)
        action = policy.decide(OnlineState(10, 0, 0, 4), train_size=3)
        self.assertEqual(action.action, DecisionAction.RETRAIN)
        self.assertEqual(action.n_samples, 3)

    def test_base_rounds_mean_half_up_and_empty_training_keeps(self):
        policy = BaseOnlinePolicy.from_plan(plan_with([retrain(2), retrain(3)]))
        state = OnlineState(2, 0, 0, 1)
        self.assertEqual(policy.decide(state, train_size=10).n_samples, 3)
        self.assertEqual(policy.decide(state, train_size=0), KEEP)

    def test_base_no_retrains_is_never(self):
        policy = BaseOnlinePolicy.from_plan(plan_with([KEEP] * 4))
        self.assertIsNone(policy.period)
        self.assertIsNone(policy.n_samples)
        self.assertEqual(policy.decide(OnlineState(100, 100, 1000, 100), train_size=4), KEEP)

    def test_signed_staleness_accumulates_without_clipping(self):
        timeline = GuardedTimeline(4, visible_through=1)
        model = TableModel({"initial": {1: 0.5, 2: 0.1, 3: 0.4}})
        policy = RecordingPolicy()
        self.assertIsInstance(policy, OnlinePolicy)
        controller = OnlineController(
            timeline, model, policy, PlannerConfig(time_cost_weight=0),
            initial_model="initial", initial_reference_loss=0.2, start_step=1,
        )
        for step, expected in ((1, (0.3, 0.3, 1)), (2, (-0.1, 0.2, 2)), (3, (0.2, 0.4, 3))):
            timeline.visible_through = step
            result = controller.step()
            self.assertFeatures(result.state, expected)
            self.assertEqual(result.training_seconds, 0)
            self.assertEqual(result.training_cost, 0)
        self.assertEqual(model.fit_calls, [])
        self.assertEqual(controller.next_step, 4)

    def test_retrain_records_pre_action_state_then_resets_to_new_loss(self):
        timeline = GuardedTimeline(3, visible_through=1)
        model = TableModel({"initial": {1: 0.5}, (1, 2): {1: 0.05, 2: 0.2}})
        policy = RecordingPolicy({1: retrain(2)})
        controller = OnlineController(
            timeline, model, policy,
            PlannerConfig(time_cost_weight=3, loss_cost_weight=2),
            initial_model="initial", initial_reference_loss=0.2, start_step=1,
        )
        result = controller.step()
        self.assertFeatures(result.state, (0.3, 0.3, 1))
        self.assertEqual(result.loss, 0.05)
        self.assertEqual(result.weighted_loss_cost, 0.1)
        self.assertGreaterEqual(result.training_seconds, 0)
        self.assertAlmostEqual(result.training_cost, 3 * result.training_seconds)
        self.assertAlmostEqual(result.total_cost, result.training_cost + 0.1)
        self.assertEqual(controller.current_model, (1, 2))
        timeline.visible_through = 2
        next_result = controller.step()
        self.assertFeatures(next_result.state, (0.15, 0.15, 1))
        self.assertEqual(next_result.loss, 0.2)
        self.assertEqual(next_result.total_cost, 0.4)

    def test_failed_fit_or_evaluation_does_not_commit_model_clock_or_staleness(self):
        for failure in ("fail_fit", "fail_new_evaluation"):
            with self.subTest(failure=failure):
                timeline = GuardedTimeline(3, visible_through=1)
                model = TableModel({"initial": {1: 0.5}, (1, 2): {1: 0.05, 2: 0.2}})
                policy = RecordingPolicy({1: retrain(2)})
                controller = OnlineController(
                    timeline, model, policy, PlannerConfig(time_cost_weight=0),
                    initial_model="initial", initial_reference_loss=0.2, start_step=1,
                )
                setattr(model, failure, True)
                with self.assertRaisesRegex(RuntimeError, "candidate"):
                    controller.step()
                self.assertEqual(controller.next_step, 1)
                self.assertEqual(controller.current_model, "initial")
                setattr(model, failure, False)
                recovered = controller.step()
                self.assertFeatures(recovered.state, (0.3, 0.3, 1))
                self.assertEqual(policy.states[0], policy.states[1])
                timeline.visible_through = 2
                self.assertFeatures(controller.step().state, (0.15, 0.15, 1))

    def test_controller_never_reads_future_blocks(self):
        timeline = GuardedTimeline(100, visible_through=0)
        model = TableModel({"initial": {0: 0.2, 1: 0.3}})
        controller = OnlineController(
            timeline, model, RecordingPolicy(),
            initial_model="initial", initial_reference_loss=0.2,
        )
        controller.step()
        self.assertTrue(timeline.reads)
        self.assertTrue(all(step == 0 for _, step in timeline.reads))
        timeline.visible_through = 1
        controller.step()
        self.assertTrue(all(step <= 1 for _, step in timeline.reads))
        self.assertEqual(model.eval_calls, [("initial", 0), ("initial", 1)])

    def test_unavailable_timestep_can_be_retried_when_stream_grows(self):
        timeline = GuardedTimeline(1, visible_through=0)
        model = TableModel({"initial": {0: 0.2, 1: 0.3}})
        controller = OnlineController(
            timeline, model, RecordingPolicy(),
            initial_model="initial", initial_reference_loss=0.2,
        )
        controller.step()
        with self.assertRaises(IndexError):
            controller.step()
        self.assertEqual(controller.next_step, 1)
        self.assertEqual(controller.current_model, "initial")
        timeline.num_steps = 2
        timeline.visible_through = 1
        resumed = controller.step()
        self.assertFeatures(resumed.state, (0.1, 0.1, 1))
        self.assertEqual(resumed.loss, 0.3)

    def test_invalid_measurement_or_action_does_not_partially_commit(self):
        for invalid in ("current_loss", "retrained_loss", "action"):
            with self.subTest(invalid=invalid):
                timeline = GuardedTimeline(2, visible_through=1)
                model = TableModel({"initial": {1: 0.5}, (1, 2): {1: 0.05}})
                policy = RecordingPolicy({1: retrain(2)})
                controller = OnlineController(
                    timeline, model, policy,
                    initial_model="initial", initial_reference_loss=0.2, start_step=1,
                )
                if invalid == "current_loss":
                    model.losses["initial"][1] = math.nan
                elif invalid == "retrained_loss":
                    model.losses[1, 2][1] = -0.1
                else:
                    policy.decisions[1] = "retrain"
                with self.assertRaises((ValueError, TypeError)):
                    controller.step()
                self.assertEqual(controller.next_step, 1)
                self.assertEqual(controller.current_model, "initial")
                self.assertEqual(controller.reference_loss, 0.2)
                self.assertEqual(controller.last_retrain_step, 0)
                self.assertEqual(controller.accumulated_staleness, 0)
                model.losses["initial"][1] = 0.5
                model.losses[1, 2][1] = 0.05
                policy.decisions[1] = retrain(2)
                self.assertFeatures(controller.step().state, (0.3, 0.3, 1))

    def test_teacher_features_are_pre_action_prefix_only_and_reuse_evaluations(self):
        timeline = GuardedTimeline(5, visible_through=2)
        model = TableModel({
            "initial": {0: 0.2, 1: 0.8, 2: 0.9},
            (0, 2): {0: 0.2, 1: 0.8, 2: 0.9},
            (1, 2): {1: 0.1, 2: 0.15},
            (2, 2): {2: 0.5},
        })
        data = collect_online_training_data(
            timeline, model, PlannerConfig(n_choices=(2,), time_cost_weight=0),
            history_steps=3, initial_model="initial", initial_reference_loss=0.2,
        )
        self.assertEqual([d.action for d in data.decisions], [DecisionAction.KEEP, DecisionAction.RETRAIN, DecisionAction.KEEP])
        self.assertEqual(data.decisions[1].n_samples, 2)
        self.assertAlmostEqual(data.plan.total_cost, 0.45)
        for state, expected in zip(data.states, ((0, 0, 0), (0.6, 0.6, 1), (0.05, 0.05, 1))):
            self.assertFeatures(state, expected)
        self.assertEqual(sorted(model.fit_calls), [(0, 2), (1, 2), (2, 2)])
        self.assertEqual(len(model.eval_calls), 9)
        self.assertTrue(all(step <= 2 for _, step in timeline.reads))

    def test_teacher_reoptimizes_prefix_instead_of_truncating_full_horizon_oracle(self):
        # One step: KEEP=.5 < RETRAIN=.6. Two steps: retrain now costs .6,
        # while keeping throughout costs1.5 and deferring retraining costs1.1.
        losses = {"initial": {0: 0.5, 1: 1.0}, (0, 2): {0: 0.0, 1: 0.0}, (1, 2): {1: 0.0}}
        config = PlannerConfig(n_choices=(2,), time_cost_weight=1, loss_cost_weight=1)
        prefix_timeline = GuardedTimeline(2, visible_through=0)
        with patch("retrainopt.evaluation.time.perf_counter", side_effect=[0.0, 0.6]):
            prefix = collect_online_training_data(
                prefix_timeline, TableModel(losses), config, history_steps=1,
                initial_model="initial", initial_reference_loss=0.5,
            )
        self.assertEqual(prefix.decisions, (KEEP,))
        self.assertAlmostEqual(prefix.plan.total_cost, 0.5)
        with patch("retrainopt.evaluation.time.perf_counter", side_effect=[0.0, 0.6, 1.0, 1.6]):
            full = RetrainingPlanner(
                GuardedTimeline(2, visible_through=1), TableModel(losses), config,
            ).plan("exact", initial_model="initial")
        self.assertEqual(full.decisions[0].action, DecisionAction.RETRAIN)
        self.assertAlmostEqual(full.total_cost, 0.6)

    def test_bayes_teacher_uses_measured_selected_candidates_outside_initial_design(self):
        model = TableModel({
            "initial": {0: 0.2, 1: 0.8, 2: 0.9},
            (0, 2): {0: 0.05, 1: 0.3, 2: 0.4},
            (1, 2): {1: 0.1, 2: 0.15},
            (2, 2): {2: 0.5},
        })
        config = PlannerConfig(
            n_choices=(2,), time_cost_weight=0, bayes_initial_samples=1,
            bayes_iterations=0, bayes_anchor_steps=(),
        )
        data = collect_online_training_data(
            GuardedTimeline(5, visible_through=2), model, config,
            history_steps=3, method="bayes", initial_model="initial",
            initial_reference_loss=0.2,
        )
        initial_candidates = {tuple(item) for item in data.plan.metadata["bayes_initial_candidates"]}
        self.assertEqual(initial_candidates, {(1, 2)})
        self.assertEqual(data.decisions[0].action, DecisionAction.RETRAIN)
        self.assertEqual(data.decisions[1].action, DecisionAction.RETRAIN)
        self.assertNotIn((0, 2), initial_candidates)
        self.assertEqual(model.fit_calls.count((0, 2)), 1)
        # This new candidate remains useful after measured-plan optimization.
        # Its measured baseline is .05 and its next loss is .3.
        # Teacher features must use these measurements, not the surrogate.
        self.assertFeatures(data.states[1], (0.25, 0.25, 1))
        self.assertAlmostEqual(data.plan.total_cost, 0.3)
        self.assertTrue(all(step <= 2 for _, step in model.eval_calls))

    def test_explicit_constant_initial_loss_does_not_train_or_evaluate_a_model(self):
        timeline = GuardedTimeline(2, visible_through=0)
        model = TableModel({})
        controller = OnlineController(
            timeline, model, RecordingPolicy(), PlannerConfig(initial_loss=0.7),
        )
        result = controller.step()
        self.assertEqual(result.loss, 0.7)
        self.assertEqual(model.fit_calls, [])
        self.assertEqual(model.eval_calls, [])

    def test_state_rejects_nonfinite_features_and_invalid_clock(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                OnlineState(1, value, 0, 1)
        for step, elapsed in ((-1, 0), (True, 0), (1, -1), (1, 0.5)):
            with self.subTest(step=step, elapsed=elapsed), self.assertRaises(ValueError):
                OnlineState(step, 0, 0, elapsed)


@unittest.skipUnless(find_spec("sklearn") is not None, "requires retrainopt[sklearn]")
class LearnedPolicyContracts(unittest.TestCase):
    def test_classifier_uses_all_rows_and_volume_regressor_only_positive_rows(self):
        states = [OnlineState(0, 0, 0, 0), OnlineState(1, -0.1, -0.1, 1), OnlineState(2, 0.4, 0.3, 2)]
        decisions = [KEEP, retrain(2), retrain(4)]
        classifier = Mock()
        regressor = Mock()
        classifier.predict.return_value = [1]
        regressor.predict.return_value = [3.5]
        with patch("sklearn.ensemble.RandomForestClassifier", return_value=classifier) as classifier_type, patch(
            "sklearn.ensemble.RandomForestRegressor", return_value=regressor,
        ) as regressor_type:
            policy = LearnedOnlinePolicy(random_seed=17).fit(states, decisions)
        self.assertEqual(classifier_type.call_args.kwargs["n_estimators"], 100)
        self.assertEqual(classifier_type.call_args.kwargs["random_state"], 17)
        self.assertEqual(regressor_type.call_args.kwargs["n_estimators"], 100)
        self.assertEqual(list(map(tuple, classifier.fit.call_args.args[0])), [features(s) for s in states])
        self.assertEqual(list(classifier.fit.call_args.args[1]), [0, 1, 1])
        self.assertEqual(list(map(tuple, regressor.fit.call_args.args[0])), [features(s) for s in states[1:]])
        self.assertEqual(list(regressor.fit.call_args.args[1]), [2, 4])
        self.assertEqual(policy.decide(states[-1], train_size=10).n_samples, 4)
        self.assertEqual(policy.decide(states[-1], train_size=3).n_samples, 3)
        regressor.predict.reset_mock()
        classifier.predict.return_value = [0]
        self.assertEqual(policy.decide(states[-1], train_size=3), KEEP)
        regressor.predict.assert_not_called()

    def test_all_keep_training_never_fits_or_calls_a_volume_model(self):
        regressor = Mock()
        with patch("sklearn.ensemble.RandomForestRegressor", return_value=regressor):
            policy = LearnedOnlinePolicy().fit([OnlineState(0, 0, 0, 0)], [KEEP])
            self.assertEqual(policy.decide(OnlineState(99, 10, 10, 99), train_size=4), KEEP)
        regressor.fit.assert_not_called()
        regressor.predict.assert_not_called()

    def test_unfit_and_invalid_predictions_fail_explicitly(self):
        state = OnlineState(1, 0.2, 0.2, 1)
        with self.assertRaises((ValueError, RuntimeError)):
            LearnedOnlinePolicy().decide(state, train_size=4)
        classifier, regressor = Mock(), Mock()
        classifier.predict.return_value = [1]
        with patch("sklearn.ensemble.RandomForestClassifier", return_value=classifier), patch(
            "sklearn.ensemble.RandomForestRegressor", return_value=regressor,
        ):
            policy = LearnedOnlinePolicy().fit([state], [retrain(2)])
        for value in (math.nan, math.inf, -1):
            regressor.predict.return_value = [value]
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.decide(state, train_size=4)


if __name__ == "__main__":
    unittest.main()
