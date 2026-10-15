"""Causal deployment of the paper's Base and Learned online strategies.

Historical tuning may inspect the supplied historical prefix. Deployment uses
only the current timestep, including its evaluation labels, before deciding.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from numbers import Integral, Real
import time
from typing import Any, Protocol, Sequence, runtime_checkable

from .config import PlannerConfig
from .evaluation import _nonnegative_measurement, _training_cost
from .interfaces import ModelAdapter, TimelineDataset, TrainingCostFunction
from .planner import RetrainingPlanner
from .types import Decision, DecisionAction, PlanResult


def _integer(value: Any, name: str, *, minimum: int | None = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or (minimum is not None and value < minimum):
        bound = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{name} must be an integer{bound}.")
    return int(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def _volume(value: float, train_size: int) -> int:
    """Round a positive volume half up, then limit it to available data."""
    value = _finite(value, "Predicted sample volume")
    if value <= 0:
        raise ValueError("Predicted sample volume must be positive.")
    return min(train_size, max(1, math.floor(value + 0.5)))


@dataclass(frozen=True)
class OnlineState:
    """The three paper features measured before the current action.

    Staleness is a raw loss difference and can be negative. Accumulated
    staleness includes the current timestep; elapsed time counts logical steps.
    Model age may exceed ``step`` when it was trained before this timeline.
    """

    step: int
    current_staleness: float
    accumulated_staleness: float
    time_since_last_retrain: int

    def __post_init__(self) -> None:
        _integer(self.step, "step")
        _integer(self.time_since_last_retrain, "time_since_last_retrain")
        _finite(self.current_staleness, "current_staleness")
        _finite(self.accumulated_staleness, "accumulated_staleness")

    def features(self) -> list[float]:
        """Return current staleness, accumulated staleness, and elapsed time."""
        return [
            float(self.current_staleness),
            float(self.accumulated_staleness),
            float(self.time_since_last_retrain),
        ]


@runtime_checkable
class OnlinePolicy(Protocol):
    """A policy proposes an action without modifying deployment state.

    The controller updates its state only after the action succeeds. Policies
    must therefore keep ``decide`` free of side effects to permit safe retries.
    """

    def decide(self, state: OnlineState, *, train_size: int) -> Decision: ...


def _policy_inputs(state: OnlineState, train_size: int) -> int:
    if not isinstance(state, OnlineState):
        raise TypeError("state must be an OnlineState.")
    return _integer(train_size, "train_size")


@dataclass(frozen=True)
class BaseOnlinePolicy:
    """Periodic retraining with the historical average interval and volume.

    ``period=T/R`` and ``n_samples`` is the mean selected retraining volume.
    Both may be fractional. Retraining occurs once elapsed time reaches the
    period; volume rounds half up to an integer and clips to current data size.
    A policy with both parameters ``None`` never retrains.
    """

    period: float | None
    n_samples: float | None

    def __post_init__(self) -> None:
        if self.period is None and self.n_samples is None:
            return
        if self.period is None or self.n_samples is None:
            raise ValueError("period and n_samples must both be positive or both be None.")
        if _finite(self.period, "period") <= 0 or _finite(self.n_samples, "n_samples") <= 0:
            raise ValueError("period and n_samples must be positive.")

    @classmethod
    def from_plan(cls, plan: PlanResult) -> BaseOnlinePolicy:
        """Tune from a plan computed exclusively on historical data."""
        if not isinstance(plan, PlanResult):
            raise TypeError("plan must be a PlanResult.")
        decisions = tuple(plan.decisions)
        if not decisions:
            raise ValueError("The historical plan must contain at least one decision.")
        for step, decision in enumerate(decisions):
            _validate_decision(decision, step)
        volumes = [d.n_samples for d in decisions if d.action == DecisionAction.RETRAIN]
        if not volumes:
            return cls(period=None, n_samples=None)
        return cls(period=len(decisions) / len(volumes), n_samples=sum(volumes) / len(volumes))

    def decide(self, state: OnlineState, *, train_size: int) -> Decision:
        """Propose a retrain when elapsed time reaches the fitted period."""
        size = _policy_inputs(state, train_size)
        if self.period is None or size == 0 or state.time_since_last_retrain < self.period:
            return Decision(DecisionAction.KEEP)
        return Decision(DecisionAction.RETRAIN, _volume(self.n_samples, size), state.step)


class LearnedOnlinePolicy:
    """Two-stage Random Forest policy fitted to historical features/actions.

    The classifier uses 100 trees as in the paper. This implementation also
    uses a 100-tree Random Forest regressor for positive retraining volumes.
    Requires the ``sklearn`` extra; imports it only when ``fit`` is called.
    """

    def __init__(self, *, random_seed: int = 42):
        self.random_seed = _integer(random_seed, "random_seed")
        self._classifier: Any = None
        self._regressor: Any = None
        self._is_fitted = False

    def fit(self, states: Sequence[OnlineState], decisions: Sequence[Decision]) -> LearnedOnlinePolicy:
        """Fit action labels on all rows and volume labels only on RETRAIN rows.

        Supply pre-action states and decisions from the same historical plan.
        An all-KEEP plan is valid and produces a policy that always keeps.
        """
        states = tuple(states)
        decisions = tuple(decisions)
        if not states or len(states) != len(decisions):
            raise ValueError("states and decisions must have the same nonzero length.")
        features = []
        labels = []
        positive_features = []
        volumes = []
        previous_step = -1
        for state, decision in zip(states, decisions):
            _policy_inputs(state, 0)
            if state.step <= previous_step:
                raise ValueError("Training states must be in strictly increasing step order.")
            previous_step = state.step
            _validate_decision(decision, state.step)
            row = state.features()
            features.append(row)
            retrain = decision.action == DecisionAction.RETRAIN
            labels.append(int(retrain))
            if retrain:
                positive_features.append(row)
                volumes.append(decision.n_samples)
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        except ImportError as exc:
            raise ImportError("LearnedOnlinePolicy requires the 'sklearn' extra: pip install 'retrainopt[sklearn]'.") from exc

        classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_seed)
        classifier.fit(features, labels)
        regressor = None
        if positive_features:
            regressor = RandomForestRegressor(n_estimators=100, random_state=self.random_seed)
            regressor.fit(positive_features, volumes)
        # A failed refit must not partially replace an already fitted policy.
        self._classifier = classifier
        self._regressor = regressor
        self._is_fitted = True
        return self

    def decide(self, state: OnlineState, *, train_size: int) -> Decision:
        """Predict an action and, when needed, a clipped integer volume."""
        size = _policy_inputs(state, train_size)
        if not self._is_fitted:
            raise RuntimeError("LearnedOnlinePolicy must be fitted before deciding.")
        if size == 0 or self._regressor is None:
            return Decision(DecisionAction.KEEP)
        prediction = self._classifier.predict([state.features()])[0]
        if prediction not in (0, 1):
            raise ValueError("The action classifier must predict 0 or 1.")
        if prediction == 0:
            return Decision(DecisionAction.KEEP)
        volume = self._regressor.predict([state.features()])[0]
        return Decision(DecisionAction.RETRAIN, _volume(volume, size), state.step)


def _validate_decision(decision: Decision, step: int) -> None:
    if not isinstance(decision, Decision):
        raise TypeError("The policy must return a Decision.")
    if decision.trained_at is not None and decision.trained_at != step:
        raise ValueError("A RETRAIN decision's trained_at must match the current step.")


@dataclass(frozen=True)
class OnlineStepResult:
    """An executed action, its pre-action features, and its realized cost.

    ``loss`` is the raw current-step loss of the model selected by the action.
    ``training_seconds`` times only the model fit. Sampling, evaluation and
    policy inference are excluded, just as in the offline objective.
    """

    state: OnlineState
    decision: Decision
    loss: float
    training_seconds: float
    training_cost: float
    weighted_loss_cost: float
    total_cost: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the row, including its pre-action feature values."""
        return {
            "schema_version": 1,
            "state": {
                "step": self.state.step,
                "current_staleness": self.state.current_staleness,
                "accumulated_staleness": self.state.accumulated_staleness,
                "time_since_last_retrain": self.state.time_since_last_retrain,
            },
            "decision": {
                "action": self.decision.action.value,
                "n_samples": self.decision.n_samples,
                "trained_at": self.decision.trained_at,
            },
            "loss": self.loss,
            "training_seconds": self.training_seconds,
            "training_cost": self.training_cost,
            "weighted_loss_cost": self.weighted_loss_cost,
            "total_cost": self.total_cost,
        }


class OnlineController:
    """Execute a policy one currently available timestep at a time.

    Supply either a real ``initial_model`` with ``initial_reference_loss`` or
    ``config.initial_loss`` for a synthetic constant-loss baseline. No initial
    model is implicitly trained. The initial model is treated as trained at
    step zero unless ``initial_trained_at`` is supplied. A negative training
    step denotes a model trained before the supplied timeline began.

    The timeline may grow between calls. ``step`` reads only the current train
    size/evaluation block, and samples training data only for RETRAIN. It never
    invokes the offline evaluation engine. Labels for the current evaluation
    block must be available before deciding. Model fitting must return a new
    artifact and evaluation must not mutate the deployed model.
    """

    def __init__(
        self,
        timeline: TimelineDataset,
        model_adapter: ModelAdapter,
        policy: OnlinePolicy,
        config: PlannerConfig | None = None,
        *,
        initial_model: Any | None = None,
        initial_reference_loss: float | None = None,
        start_step: int = 0,
        initial_trained_at: int = 0,
        initial_accumulated_staleness: float = 0.0,
        training_cost_fn: TrainingCostFunction | None = None,
    ):
        self.timeline = timeline
        self.model_adapter = model_adapter
        self.policy = policy
        self.config = copy.deepcopy(config) if config is not None else PlannerConfig()
        if training_cost_fn is not None and not callable(training_cost_fn):
            raise TypeError("training_cost_fn must be callable.")
        self.training_cost_fn = training_cost_fn
        if not isinstance(policy, OnlinePolicy):
            raise TypeError("policy must implement OnlinePolicy.decide.")
        self._next_step = _integer(start_step, "start_step")
        self._last_retrain_step = _integer(initial_trained_at, "initial_trained_at", minimum=None)
        if self._last_retrain_step > self._next_step:
            raise ValueError("initial_trained_at cannot be later than start_step.")
        self._accumulated_staleness = _finite(initial_accumulated_staleness, "initial_accumulated_staleness")
        _validate_initial_baseline(self.config, initial_model, initial_reference_loss, require_model=True)
        self._synthetic_initial = self.config.initial_loss is not None
        self._reference_loss = float(self.config.initial_loss) if self._synthetic_initial else float(initial_reference_loss)
        self._current_model = initial_model

    @property
    def current_model(self) -> Any:
        """Return the deployed artifact, or None for a synthetic baseline."""
        return self._current_model

    @property
    def next_step(self) -> int:
        """Return the timestep that the next successful call will process."""
        return self._next_step

    @property
    def reference_loss(self) -> float:
        """Return the deployed model's loss measured at its training step."""
        return self._reference_loss

    @property
    def last_retrain_step(self) -> int:
        """Return the training timestep of the deployed model."""
        return self._last_retrain_step

    @property
    def accumulated_staleness(self) -> float:
        """Return accumulated staleness after the most recent completed action."""
        return self._accumulated_staleness

    def step(self) -> OnlineStepResult:
        """Observe, decide and execute one step; commit state only on success.

        A RETRAIN uses the new model's current-step loss for both its cost and
        its future reference baseline. Insufficient training data forces KEEP.
        An unavailable next step raises IndexError and can be retried once the
        timeline grows. Adapter exceptions propagate without advancing state.
        """
        step = self._next_step
        available_steps = _integer(self.timeline.num_steps, "TimelineDataset.num_steps")
        if step >= available_steps:
            raise IndexError(f"Timestep {step} is not yet available.")
        train_size = _integer(self.timeline.train_size(step), f"train_size({step})")
        eval_data = self.timeline.get_eval_block(step)
        if hasattr(eval_data, "__len__") and len(eval_data) == 0:
            raise ValueError(f"Evaluation block at step {step} must not be empty.")
        current_loss = self._reference_loss if self._synthetic_initial else _nonnegative_measurement(
            self.model_adapter.evaluate(self._current_model, eval_data, step=step), f"Current model loss at step {step}"
        )
        staleness = current_loss - self._reference_loss
        state = OnlineState(step, staleness, self._accumulated_staleness + staleness, step - self._last_retrain_step)
        decision = self.policy.decide(state, train_size=train_size)
        _validate_decision(decision, step)
        if train_size < self.config.min_train_size:
            decision = Decision(DecisionAction.KEEP)

        model = self._current_model
        training_seconds = 0.0
        training_cost = 0.0
        loss = current_loss
        if decision.action == DecisionAction.RETRAIN:
            n_samples = min(train_size, decision.n_samples)
            if n_samples < self.config.min_train_size:
                raise ValueError("The proposed retraining volume is below min_train_size.")
            decision = Decision(DecisionAction.RETRAIN, n_samples, step)
            seed = self.config.random_seed + step * 1009 + n_samples
            train_data = self.timeline.sample_train_block(step, n_samples, seed=seed)
            started = time.perf_counter()
            model = self.model_adapter.fit(train_data, step=step, n_samples=n_samples, random_seed=seed)
            training_seconds = _nonnegative_measurement(time.perf_counter() - started, "Measured training time")
            training_cost = _training_cost(
                self.training_cost_fn, step=step, n_samples=n_samples,
                training_seconds=training_seconds, model=model, time_cost_weight=self.config.time_cost_weight,
            )
            loss = _nonnegative_measurement(
                self.model_adapter.evaluate(model, eval_data, step=step), f"Retrained model loss at step {step}"
            )
        weighted_loss = _nonnegative_measurement(loss * self.config.loss_cost_weight, "Weighted loss")
        total_cost = _nonnegative_measurement(training_cost + weighted_loss, "Total step cost")
        result = OnlineStepResult(state, decision, loss, training_seconds, training_cost, weighted_loss, total_cost)
        self._current_model = model
        self._next_step = step + 1
        if decision.action == DecisionAction.RETRAIN:
            self._synthetic_initial = False
            self._reference_loss = loss
            self._last_retrain_step = step
            self._accumulated_staleness = 0.0
        else:
            self._accumulated_staleness = state.accumulated_staleness
        return result


def _validate_initial_baseline(
    config: PlannerConfig, initial_model: Any, reference_loss: float | None, *, require_model: bool
) -> None:
    if config.initial_loss is not None:
        if initial_model is not None or reference_loss is not None:
            raise ValueError("initial_loss is mutually exclusive with initial_model and initial_reference_loss.")
    elif initial_model is not None:
        if reference_loss is None:
            raise ValueError("An explicit initial_model requires initial_reference_loss.")
        _nonnegative_measurement(reference_loss, "initial_reference_loss")
    elif require_model or reference_loss is not None:
        raise ValueError("Supply an initial_model with initial_reference_loss, or configure initial_loss.")


@dataclass(frozen=True)
class OnlineTrainingData:
    """A prefix-only offline plan paired with its pre-action feature rows."""

    plan: PlanResult
    states: tuple[OnlineState, ...]
    decisions: tuple[Decision, ...]


class _HistoryPrefix(TimelineDataset):
    """Expose exactly the historical prefix, including delegated sampling."""

    def __init__(self, timeline: TimelineDataset, history_steps: int):
        self._timeline = timeline
        self._history_steps = history_steps

    @property
    def num_steps(self) -> int:
        return self._history_steps

    def _check(self, step: int) -> None:
        if not 0 <= step < self._history_steps:
            raise IndexError(f"Timestep {step} lies outside the historical prefix.")

    def get_train_block(self, step: int) -> Any:
        self._check(step)
        return self._timeline.get_train_block(step)

    def get_eval_block(self, step: int) -> Any:
        self._check(step)
        return self._timeline.get_eval_block(step)

    def train_size(self, step: int) -> int:
        self._check(step)
        return self._timeline.train_size(step)

    def sample_train_block(self, step: int, n_samples: int, *, seed: int) -> Any:
        self._check(step)
        return self._timeline.sample_train_block(step, n_samples, seed=seed)


def collect_online_training_data(
    timeline: TimelineDataset,
    model_adapter: ModelAdapter,
    config: PlannerConfig | None = None,
    *,
    history_steps: int,
    method: str = "exact",
    initial_model: Any | None = None,
    initial_reference_loss: float | None = None,
    initial_model_age: int = 0,
    initial_accumulated_staleness: float = 0.0,
    training_cost_fn: TrainingCostFunction | None = None,
) -> OnlineTrainingData:
    """Plan exclusively on ``[0, history_steps)`` and collect teacher rows.

    The exact or Bayes planner is run anew on a bounded view of history. A
    full-horizon plan must never be sliced to produce causal tuning labels.
    Features use the deployed model before each historical action, including
    current signed staleness. After RETRAIN the reference becomes the selected
    candidate's loss on that same step and accumulated staleness resets.

    Candidate and initial losses come from the same planning evaluations;
    collecting features performs no extra fits or evaluations. As with the
    offline planner, omitting an initial model and initial_loss bootstraps a
    model on step zero; its step-zero loss supplies the initial reference.
    An explicit model requires its measured ``initial_reference_loss``.
    ``initial_model_age`` counts steps since its training at the window start;
    ``initial_accumulated_staleness`` carries the signed sum before that start.
    Nonzero carry requires an explicit model or a configured constant baseline.
    All returned step indices remain local to the supplied timeline.
    """
    steps = _integer(history_steps, "history_steps", minimum=1)
    available = _integer(timeline.num_steps, "TimelineDataset.num_steps")
    if steps > available:
        raise ValueError("history_steps cannot exceed the currently available timeline.")
    snapshot = copy.deepcopy(config) if config is not None else PlannerConfig()
    _validate_initial_baseline(snapshot, initial_model, initial_reference_loss, require_model=False)
    age = _integer(initial_model_age, "initial_model_age")
    accumulated = _finite(initial_accumulated_staleness, "initial_accumulated_staleness")
    if initial_model is None and snapshot.initial_loss is None and (age != 0 or accumulated != 0.0):
        raise ValueError("Nonzero initial history requires an initial_model or initial_loss; bootstrap starts at step zero.")
    prefix = _HistoryPrefix(timeline, steps)
    planner = RetrainingPlanner(prefix, model_adapter, snapshot, training_cost_fn=training_cost_fn)
    plan = planner.plan(method, initial_model=initial_model)
    initial = planner.engine._last_initial_evaluation
    if initial is None:
        raise RuntimeError("The historical planner did not evaluate its initial model.")
    current = initial
    reference = float(initial_reference_loss) if initial_reference_loss is not None else initial.future_losses[0]
    last_retrain_step = -age
    candidates = {evaluation.model_id: evaluation for evaluation in planner.engine.iter_cached_candidates()}
    states = []
    for step, decision in enumerate(plan.decisions):
        staleness = current.future_losses[step] - reference
        accumulated += staleness
        states.append(OnlineState(step, staleness, accumulated, step - last_retrain_step))
        if decision.action == DecisionAction.RETRAIN:
            current = candidates[(step, decision.n_samples)]
            reference = current.future_losses[step]
            last_retrain_step = step
            accumulated = 0.0
    return OnlineTrainingData(plan, tuple(states), tuple(plan.decisions))
