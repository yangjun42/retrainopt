"""Generate the public demo from RetrainOpt's exact planner, with no extras.

Run ``python examples/build_demo.py`` after installing this checkout. Use
``--check`` to verify the checked-in data without writing it. All losses and
training prices are synthetic inputs; no model fits or benchmarks are run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from retrainopt import (
    CandidateEvaluation,
    Decision,
    DecisionAction,
    ModelAdapter,
    PlannerConfig,
    RetrainingPlanner,
    TimelineDataset,
    __version__,
)


STEPS = 24
DATASET_SIZE = 100
VOLUMES = (25, 50, 75, 100)
QUALITY_WEIGHTS = (0.5, 1.0, 2.0, 4.0, 8.0)
TRAINING_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
PERIOD = 6
OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "_static" / "demo" / "data.json"
SCENARIOS = (
    ("sudden", "Sudden shifts", "Two abrupt changes make an old model less useful."),
    ("gradual", "Gradual drift", "The stream moves steadily away from its starting point."),
    ("recurring", "Recurring regimes", "Familiar regimes return, so keeping a model can pay off."),
)


def regime(pattern: str) -> tuple[float, ...]:
    if pattern == "sudden":
        return tuple(0.08 if t < 8 else 0.82 if t < 17 else 0.3 for t in range(STEPS))
    if pattern == "gradual":
        return tuple(round(0.08 + 0.82 * (t / (STEPS - 1)) ** 1.6, 8) for t in range(STEPS))
    if pattern == "recurring":
        return tuple(round(0.45 + 0.36 * math.sin(t / 2.4), 8) for t in range(STEPS))
    raise ValueError(f"Unknown demo scenario: {pattern}")


def synthetic_loss(values: tuple[float, ...], trained_at: int, n_samples: int, step: int) -> float:
    """Toy loss: base error + regime mismatch + small-data penalty + age."""
    volume = n_samples / DATASET_SIZE
    return round((
        0.04
        + 0.85 * abs(values[step] - values[trained_at])
        + 0.12 * (1.0 - volume)
        + 0.04 * (step - trained_at) / (STEPS - 1)
    ), 8)


def synthetic_training_cost(step: int, n_samples: int, multiplier: float) -> float:
    volume = n_samples / DATASET_SIZE
    return round(multiplier * (0.08 + volume**1.8) * (0.9 + 0.2 * step / (STEPS - 1)), 8)


class DemoTimeline(TimelineDataset):
    """Only candidate counts are needed when an evaluator supplies cost tables."""

    @property
    def num_steps(self) -> int:
        return STEPS

    def train_size(self, step: int) -> int:
        # Step 0 starts with an existing full-data model, shared by all policies.
        return 0 if step == 0 else DATASET_SIZE

    def get_train_block(self, step: int):
        raise AssertionError("The demo uses supplied costs, not training data.")

    def get_eval_block(self, step: int):
        raise AssertionError("The demo uses supplied costs, not evaluation data.")


class UnusedModelAdapter(ModelAdapter):
    def fit(self, train_data, *, step, n_samples, random_seed):
        raise AssertionError("The demo must not fit models.")

    def evaluate(self, model, eval_data, *, step):
        raise AssertionError("The demo must use its synthetic loss table.")


class DemoEvaluator:
    """Public EvaluationProvider implementation for the synthetic cost table."""

    def __init__(self, values: tuple[float, ...], training_multiplier: float):
        self.values = values
        self.training_multiplier = training_multiplier
        self.cache: dict[tuple[int, int], CandidateEvaluation] = {}

    def evaluate_initial_model(self, initial_model=None) -> CandidateEvaluation:
        return CandidateEvaluation(
            model_id=(-1, 0), dataset_size=DATASET_SIZE,
            training_seconds=0.0, training_cost=0.0,
            future_losses={t: synthetic_loss(self.values, 0, DATASET_SIZE, t) for t in range(STEPS)},
            metadata={"source": "synthetic_existing_model"},
        )

    def evaluate_candidate_uncached(self, step: int, n_samples: int) -> CandidateEvaluation:
        return CandidateEvaluation(
            model_id=(step, n_samples), dataset_size=DATASET_SIZE,
            training_seconds=0.0,
            training_cost=synthetic_training_cost(step, n_samples, self.training_multiplier),
            future_losses={t: synthetic_loss(self.values, step, n_samples, t) for t in range(step, STEPS)},
        )

    def evaluate_candidate(self, step: int, n_samples: int) -> CandidateEvaluation:
        key = (step, n_samples)
        if key not in self.cache:
            self.cache[key] = self.evaluate_candidate_uncached(step, n_samples)
        return self.cache[key]

    def iter_cached_candidates(self):
        return iter(self.cache.values())


def evaluate_schedule(decisions: list[Decision], evaluator: DemoEvaluator, quality_weight: float) -> dict:
    """Account for a supplied schedule; this does not optimize or duplicate DP."""
    current = evaluator.evaluate_initial_model()
    losses, costs, actions = [], [], []
    for step, decision in enumerate(decisions):
        cost = 0.0
        if decision.action == DecisionAction.RETRAIN:
            current = evaluator.evaluate_candidate(step, decision.n_samples)
            cost = current.training_cost
        losses.append(current.future_losses[step])
        costs.append(cost)
        actions.append(decision.n_samples if decision.action == DecisionAction.RETRAIN else 0)
    training = sum(costs)
    loss = sum(losses)
    return {
        "total_cost": training + quality_weight * loss,
        "training_cost": training,
        "raw_loss": loss,
        "weighted_loss_cost": quality_weight * loss,
        "retrain_count": sum(n > 0 for n in actions),
        "total_samples": sum(actions),
        "actions": actions,
        "step_losses": losses,
        "step_training_costs": costs,
    }


def build_case(pattern: str, quality_weight: float, training_multiplier: float) -> dict:
    evaluator = DemoEvaluator(regime(pattern), training_multiplier)
    config = PlannerConfig(n_choices=VOLUMES, loss_cost_weight=quality_weight)
    result = RetrainingPlanner(
        DemoTimeline(), UnusedModelAdapter(), config, evaluator=evaluator,
    ).plan("exact")
    optimized = evaluate_schedule(result.decisions, evaluator, quality_weight)
    for field in ("total_cost", "training_cost", "weighted_loss_cost"):
        if not math.isclose(optimized[field], getattr(result, field), rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"Demo accounting disagrees with the planner: {field}")
    policies = {"optimized": optimized}
    for name, should_retrain in (
        ("periodic", lambda step: step > 0 and step % PERIOD == 0),
        ("always", lambda step: step > 0),
        ("never", lambda step: False),
    ):
        decisions = [
            Decision(DecisionAction.RETRAIN, DATASET_SIZE, step)
            if should_retrain(step) else Decision(DecisionAction.KEEP)
            for step in range(STEPS)
        ]
        policies[name] = evaluate_schedule(decisions, evaluator, quality_weight)
    return {"scenario": pattern, "quality_weight": quality_weight,
            "training_multiplier": training_multiplier, "policies": policies}


def build_data() -> dict:
    return {
        "schema_version": 1,
        "package_version": __version__,
        "generator": "examples/build_demo.py",
        "provenance": "Synthetic cost tables; exact offline plans computed by RetrainOpt. Not paper measurements.",
        "steps": STEPS,
        "dataset_size": DATASET_SIZE,
        "sample_choices": list(VOLUMES),
        "period": PERIOD,
        "quality_weights": list(QUALITY_WEIGHTS),
        "training_multipliers": list(TRAINING_MULTIPLIERS),
        "scenarios": [{"id": key, "name": name, "description": description, "regime": list(regime(key))}
                      for key, name, description in SCENARIOS],
        "cases": [build_case(key, quality, training)
                  for key, _, _ in SCENARIOS
                  for quality in QUALITY_WEIGHTS
                  for training in TRAINING_MULTIPLIERS],
    }


def serialized_data() -> str:
    def rounded(value):
        if isinstance(value, float):
            return round(value, 8)
        if isinstance(value, list):
            return [rounded(item) for item in value]
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items()}
        return value

    # Quantize inputs above and serialized results here so libm's last bits do
    # not make the static artifact depend on an OS or Python minor version.
    return json.dumps(rounded(build_data()), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the artifact without modifying it.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    expected = serialized_data()
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != expected:
            parser.exit(1, "Demo data is stale. Run python examples/build_demo.py with this package version.\n")
        print(f"Demo data matches RetrainOpt {__version__}: 60 synthetic cases.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(expected, encoding="utf-8")
        print(f"Generated 60 synthetic cases with RetrainOpt {__version__}.")


if __name__ == "__main__":
    main()
