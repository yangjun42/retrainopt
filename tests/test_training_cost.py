"""Custom training costs agree across planning and deployment."""

from itertools import cycle
import math
import unittest
from unittest.mock import Mock, patch

from retrainopt import (
    BaseOnlinePolicy,
    CandidateEvaluation,
    Decision,
    DecisionAction,
    OnlineController,
    PlannerConfig,
    RetrainingPlanner,
    collect_online_training_data,
)
from retrainopt.evaluation import EvaluationEngine
from retrainopt.interfaces import TimelineDataset


class VisibleTimeline(TimelineDataset):
    def __init__(self, steps=1, visible_through=0):
        self.steps = steps
        self.visible_through = visible_through
        self.reads = []

    @property
    def num_steps(self):
        return self.steps

    def _read(self, step):
        if not 0 <= step <= self.visible_through:
            raise AssertionError(f"Future data read at {step}")
        self.reads.append(step)

    def get_train_block(self, step):
        self._read(step)
        return [step, step]

    def get_eval_block(self, step):
        self._read(step)
        return [step]

    def train_size(self, step):
        self._read(step)
        return 2


class MeteredAdapter:
    def __init__(self):
        self.fitted_models = []
        self.eval_calls = []

    def fit(self, train_data, *, step, n_samples, random_seed):
        model = {"loss": 0.125, "resource_cost": 0.25, "step": step, "samples": n_samples}
        self.fitted_models.append(model)
        return model

    def evaluate(self, model, eval_data, *, step):
        self.eval_calls.append((model, step))
        return model["loss"]


class FixedPolicy:
    def __init__(self, n_samples=None):
        self.n_samples = n_samples

    def decide(self, state, *, train_size):
        if self.n_samples is None:
            return Decision(DecisionAction.KEEP)
        return Decision(DecisionAction.RETRAIN, self.n_samples, state.step)


class FixedCostProvider:
    def __init__(self, cost):
        self.cost = cost
        self.cached = {}

    def evaluate_initial_model(self, initial_model=None):
        return CandidateEvaluation(
            (-1, 0), 0, 0.0, 0.0, {step: 0.5 for step in range(3)},
        )

    def evaluate_candidate(self, step, n_samples):
        key = (step, n_samples)
        if key not in self.cached:
            self.cached[key] = self.evaluate_candidate_uncached(step, n_samples)
        return self.cached[key]

    def evaluate_candidate_uncached(self, step, n_samples):
        return CandidateEvaluation(
            (step, n_samples), 2, 0.5, self.cost,
            {future: 0.125 for future in range(step, 3)},
        )

    def iter_cached_candidates(self):
        return self.cached.values()


def deployment_state(controller):
    return (
        controller.current_model,
        controller.reference_loss,
        controller.last_retrain_step,
        controller.next_step,
        controller.accumulated_staleness,
    )


class TrainingCostContracts(unittest.TestCase):
    def test_default_cost_remains_fit_seconds_times_weight(self):
        config = PlannerConfig(initial_loss=0.5, time_cost_weight=4, loss_cost_weight=2)
        engine = EvaluationEngine(VisibleTimeline(), MeteredAdapter(), config)
        with patch("retrainopt.evaluation.time.perf_counter", side_effect=[10.0, 10.25]):
            candidate = engine.evaluate_candidate(0, 2)
        controller = OnlineController(VisibleTimeline(), MeteredAdapter(), FixedPolicy(2), config)
        with patch("retrainopt.online.time.perf_counter", side_effect=[20.0, 20.25]):
            result = controller.step()
        self.assertEqual(candidate.training_seconds, 0.25)
        self.assertEqual(candidate.training_cost, 1.0)
        self.assertEqual(result.training_seconds, 0.25)
        self.assertEqual(result.training_cost, 1.0)
        self.assertEqual(result.weighted_loss_cost, 0.25)
        self.assertEqual(result.total_cost, 1.25)

    def test_callback_receives_fitted_model_clipped_volume_and_fit_only_time(self):
        for mode in ("evaluation", "online"):
            with self.subTest(mode=mode):
                adapter = MeteredAdapter()
                calls = []

                def cost(*, step, n_samples, training_seconds, model):
                    self.assertEqual(timer.call_count, 2)
                    self.assertEqual(adapter.eval_calls, [])
                    self.assertIs(model, adapter.fitted_models[-1])
                    calls.append((step, n_samples, training_seconds))
                    return model["resource_cost"]

                config = PlannerConfig(initial_loss=0.5, time_cost_weight=7, loss_cost_weight=2)
                with patch(f"retrainopt.{mode}.time.perf_counter", side_effect=[5.0, 5.5]) as timer:
                    if mode == "evaluation":
                        engine = EvaluationEngine(VisibleTimeline(), adapter, config, training_cost_fn=cost)
                        result = engine.evaluate_candidate(0, 99)
                        self.assertEqual(result.model_id, (0, 2))
                    else:
                        controller = OnlineController(
                            VisibleTimeline(), adapter, FixedPolicy(99), config, training_cost_fn=cost,
                        )
                        result = controller.step()
                        self.assertEqual(result.decision.n_samples, 2)
                        self.assertEqual(result.total_cost, 0.5)
                self.assertEqual(calls, [(0, 2, 0.5)])
                self.assertEqual(result.training_seconds, 0.5)
                self.assertEqual(result.training_cost, 0.25)

    def test_custom_cost_reaches_exact_and_bayes_public_planners(self):
        for method in ("exact", "bayes"):
            with self.subTest(method=method):
                cost = Mock(return_value=0.25)
                config = PlannerConfig(
                    n_choices=(2,), initial_loss=0.5, time_cost_weight=8,
                    bayes_initial_samples=1, bayes_iterations=0, bayes_anchor_steps=(),
                )
                planner = RetrainingPlanner(
                    VisibleTimeline(), MeteredAdapter(), config, training_cost_fn=cost,
                )
                result = planner.plan(method)
                self.assertEqual(result.decisions, [Decision(DecisionAction.RETRAIN, 2, 0)])
                self.assertEqual(result.training_cost, 0.25)
                self.assertEqual(result.total_cost, 0.375)
                cost.assert_called_once()

    def test_custom_cost_changes_history_plan_and_matches_online_accounting(self):
        for method in ("exact", "bayes"):
            with self.subTest(method=method):
                config = PlannerConfig(
                    n_choices=(2,), initial_loss=0.5, time_cost_weight=8,
                    bayes_initial_samples=1, bayes_iterations=0, bayes_anchor_steps=(),
                )
                timeline = VisibleTimeline(steps=3)
                with patch("retrainopt.evaluation.time.perf_counter", side_effect=[0.0, 0.25]):
                    default = collect_online_training_data(
                        timeline, MeteredAdapter(), config, history_steps=1, method=method,
                    )
                self.assertEqual(default.decisions, (Decision(DecisionAction.KEEP),))
                cost_calls = []

                def cost(*, step, n_samples, training_seconds, model):
                    cost_calls.append((step, n_samples))
                    return model["resource_cost"]

                history = collect_online_training_data(
                    timeline, MeteredAdapter(), config, history_steps=1, method=method,
                    training_cost_fn=cost,
                )
                self.assertEqual(history.decisions, (Decision(DecisionAction.RETRAIN, 2, 0),))
                self.assertEqual(history.plan.total_cost, 0.375)
                self.assertEqual(history.states[0].features(), [0.0, 0.0, 0.0])
                self.assertTrue(all(step == 0 for step in timeline.reads))
                self.assertEqual(cost_calls, [(0, 2)])

                timeline.visible_through = 1
                controller = OnlineController(
                    timeline, MeteredAdapter(), BaseOnlinePolicy.from_plan(history.plan), config,
                    start_step=1, training_cost_fn=cost,
                )
                result = controller.step()
                self.assertEqual(result.decision, Decision(DecisionAction.RETRAIN, 2, 1))
                self.assertEqual(result.training_cost, history.plan.training_cost)
                self.assertEqual(result.total_cost, history.plan.total_cost)
                self.assertEqual(cost_calls, [(0, 2), (1, 2)])
                self.assertTrue(all(step <= 1 for step in timeline.reads))

    def test_bayes_unobserved_custom_costs_are_independent_of_time_weight(self):
        for entry_point in ("planner", "history", "external_evaluator"):
            for price in (0.25, 1.25):
                cost = Mock(return_value=price)
                expected_decisions = [Decision(DecisionAction.KEEP)] * 3
                expected_total = 1.5
                if price == 0.25:
                    expected_decisions[0] = Decision(DecisionAction.RETRAIN, 2, 0)
                    expected_total = 0.625
                for weight in (0, 7, 999):
                    with self.subTest(entry_point=entry_point, price=price, weight=weight):
                        timeline = VisibleTimeline(steps=5 if entry_point == "history" else 3, visible_through=2)
                        adapter = MeteredAdapter()
                        config = PlannerConfig(
                            n_choices=(2,), initial_loss=0.5, time_cost_weight=weight,
                            bayes_initial_samples=1, bayes_iterations=0, bayes_anchor_steps=(),
                        )
                        cost.reset_mock()
                        if entry_point == "history":
                            result = collect_online_training_data(
                                timeline, adapter, config, history_steps=3, method="bayes",
                                training_cost_fn=cost,
                            ).plan
                        elif entry_point == "external_evaluator":
                            result = RetrainingPlanner(
                                timeline, adapter, config, evaluator=FixedCostProvider(price),
                            ).plan("bayes")
                        else:
                            result = RetrainingPlanner(
                                timeline, adapter, config, training_cost_fn=cost,
                            ).plan("bayes")
                        self.assertEqual(result.metadata["candidate_count"], 3)
                        self.assertEqual(result.metadata["surrogate_evaluated_candidates"], 1)
                        self.assertEqual(result.metadata["bayes_iterations_completed"], 0)
                        initial_design = {tuple(item) for item in result.metadata["bayes_initial_candidates"]}
                        self.assertEqual(initial_design, {(1, 2)})
                        self.assertEqual(result.decisions, expected_decisions)
                        self.assertAlmostEqual(result.total_cost, expected_total)
                        self.assertAlmostEqual(result.estimated_total_cost, expected_total)
                        self.assertAlmostEqual(result.training_cost, price if price == 0.25 else 0.0)
                        if price == 0.25:
                            self.assertNotIn((0, 2), initial_design)
                            self.assertEqual(result.metadata["evaluated_candidates"], 2)
                        if entry_point != "external_evaluator":
                            self.assertEqual(cost.call_count, 2 if price == 0.25 else 1)
                        self.assertTrue(all(step <= 2 for step in timeline.reads))

    def test_cached_candidates_do_not_recompute_cost(self):
        adapter = MeteredAdapter()
        cost = Mock(return_value=0.25)
        engine = EvaluationEngine(VisibleTimeline(), adapter, PlannerConfig(), training_cost_fn=cost)
        first = engine.evaluate_candidate(0, 99)
        self.assertIs(engine.evaluate_candidate(0, 2), first)
        self.assertIs(engine.evaluate_candidate_uncached(0, 2), first)
        self.assertEqual(list(engine.iter_cached_candidates()), [first])
        self.assertEqual(len(adapter.fitted_models), 1)
        cost.assert_called_once()

        diagnostic = EvaluationEngine(VisibleTimeline(), adapter, PlannerConfig(), training_cost_fn=cost)
        diagnostic.evaluate_candidate_uncached(0, 2)
        self.assertEqual(list(diagnostic.iter_cached_candidates()), [])
        diagnostic.evaluate_candidate(0, 2)
        self.assertEqual(cost.call_count, 3)

    def test_keep_and_initial_bootstrap_do_not_call_custom_cost(self):
        cost = Mock(side_effect=AssertionError("No retraining cost expected"))
        adapter = MeteredAdapter()
        engine = EvaluationEngine(VisibleTimeline(), adapter, PlannerConfig(), training_cost_fn=cost)
        initial = engine.evaluate_initial_model()
        self.assertEqual(initial.training_cost, 0)
        self.assertEqual(len(adapter.fitted_models), 1)
        for policy, minimum in ((FixedPolicy(), 1), (FixedPolicy(2), 3)):
            controller = OnlineController(
                VisibleTimeline(), MeteredAdapter(), policy,
                PlannerConfig(initial_loss=0.5, min_train_size=minimum), training_cost_fn=cost,
            )
            with patch("retrainopt.online.time.perf_counter") as timer:
                result = controller.step()
            timer.assert_not_called()
            self.assertEqual(result.decision.action, DecisionAction.KEEP)
            self.assertEqual(result.training_cost, 0)
        cost.assert_not_called()

    def test_cost_failures_leave_candidate_cache_empty_and_allow_retry(self):
        invalid_values = (-1, math.inf, -math.inf, math.nan, True, None, "0.25")
        failure = RuntimeError("cost measurement failed")
        for invalid in (*invalid_values, failure):
            with self.subTest(value=invalid):
                adapter = MeteredAdapter()
                cost = Mock(side_effect=failure) if invalid is failure else Mock(return_value=invalid)
                engine = EvaluationEngine(VisibleTimeline(), adapter, PlannerConfig(), training_cost_fn=cost)
                expected = RuntimeError if invalid is failure else ValueError
                with self.assertRaises(expected) as raised:
                    engine.evaluate_candidate(0, 2)
                if invalid is failure:
                    self.assertIs(raised.exception, failure)
                self.assertEqual(list(engine.iter_cached_candidates()), [])
                self.assertEqual(adapter.eval_calls, [])
                cost.side_effect = None
                cost.return_value = 0.25
                result = engine.evaluate_candidate(0, 2)
                self.assertEqual(result.training_cost, 0.25)
                self.assertEqual(list(engine.iter_cached_candidates()), [result])
                self.assertEqual(len(adapter.fitted_models), 2)
                self.assertEqual(cost.call_count, 2)

    def test_cost_failures_preserve_nonzero_deployment_state_and_allow_retry(self):
        failure = RuntimeError("cost measurement failed")
        for invalid in (-1, math.inf, math.nan, True, None, "0.25", failure):
            with self.subTest(value=invalid):
                initial_model = {"loss": 0.75}
                adapter = MeteredAdapter()
                cost = Mock(side_effect=failure) if invalid is failure else Mock(return_value=invalid)
                controller = OnlineController(
                    VisibleTimeline(steps=3, visible_through=1), adapter, FixedPolicy(2),
                    PlannerConfig(time_cost_weight=7, loss_cost_weight=2),
                    initial_model=initial_model, initial_reference_loss=0.5, start_step=1,
                    initial_accumulated_staleness=-0.25, training_cost_fn=cost,
                )
                before = deployment_state(controller)
                expected = RuntimeError if invalid is failure else ValueError
                with patch("retrainopt.online.time.perf_counter", side_effect=cycle((1.0, 1.25))):
                    with self.assertRaises(expected) as raised:
                        controller.step()
                    if invalid is failure:
                        self.assertIs(raised.exception, failure)
                    self.assertEqual(deployment_state(controller), before)
                    self.assertIs(controller.current_model, initial_model)
                    cost.side_effect = None
                    cost.return_value = 0.25
                    recovered = controller.step()
                self.assertEqual(recovered.state.features(), [0.25, 0.0, 1.0])
                self.assertEqual(recovered.training_seconds, 0.25)
                self.assertEqual(recovered.training_cost, 0.25)
                self.assertEqual(recovered.total_cost, 0.5)
                self.assertIs(controller.current_model, adapter.fitted_models[-1])
                self.assertEqual(deployment_state(controller)[1:], (0.125, 1, 2, 0.0))
                self.assertEqual(cost.call_count, 2)

    def test_noncallable_cost_is_rejected_before_training_at_all_entry_points(self):
        for mode in ("engine", "planner", "history", "online"):
            with self.subTest(mode=mode):
                adapter = MeteredAdapter()
                timeline = VisibleTimeline()
                config = PlannerConfig(initial_loss=0.5)
                with self.assertRaises(TypeError):
                    if mode == "engine":
                        EvaluationEngine(timeline, adapter, config, training_cost_fn=object())
                    elif mode == "planner":
                        RetrainingPlanner(timeline, adapter, config, training_cost_fn=object())
                    elif mode == "history":
                        collect_online_training_data(
                            timeline, adapter, config, history_steps=1, training_cost_fn=object(),
                        )
                    else:
                        OnlineController(timeline, adapter, FixedPolicy(2), config, training_cost_fn=object())
                self.assertEqual(adapter.fitted_models, [])
                self.assertEqual(adapter.eval_calls, [])

    def test_injected_evaluator_and_training_cost_callback_are_mutually_exclusive(self):
        timeline, adapter, config = VisibleTimeline(), MeteredAdapter(), PlannerConfig(initial_loss=0.5)
        evaluator = EvaluationEngine(timeline, adapter, config)
        with self.assertRaises(ValueError):
            RetrainingPlanner(
                timeline, adapter, config, evaluator=evaluator, training_cost_fn=lambda **kwargs: 0.25,
            )
        self.assertEqual(adapter.fitted_models, [])
        self.assertEqual(adapter.eval_calls, [])


if __name__ == "__main__":
    unittest.main()
