"""CLI entry point for RetrainOpt."""

from __future__ import annotations

import argparse
import json
import math
import re
from typing import List

from .adapters import BinaryTimelineAdapter, PrototypeModelAdapter, SklearnModelAdapter
from .config import PlannerConfig
from .planner import RetrainingPlanner


def _parse_n_choice(value: str) -> int | float:
    """Preserve integer-count versus decimal-fraction command-line intent."""
    try:
        if re.fullmatch(r"[+-]?\d+", value):
            count = int(value)
            if count >= 1:
                return count
        else:
            fraction = float(value)
            if math.isfinite(fraction) and 0 < fraction <= 1:
                return fraction
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Use a positive integer sample count (1) or a decimal fraction in (0, 1] (1.0 means all samples).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrainopt",
        description="RetrainOpt: Cost-aware retraining planning",
    )
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser(
        "plan",
        help="Plan a retraining schedule for a train/test binary timeline directory.",
    )
    plan_parser.add_argument("--dataset-dir", required=True, help="Path containing train/*.bin and test/*.bin.")
    plan_parser.add_argument("--num-features", type=int, default=None, help="Feature count if metadata.json is absent.")
    plan_parser.add_argument("--method", choices=["exact", "bayes"], default="exact")
    plan_parser.add_argument(
        "--bayes-backend",
        choices=["lightweight", "botorch"],
        default="lightweight",
        help="Surrogate backend used when --method=bayes.",
    )
    plan_parser.add_argument(
        "--model",
        choices=["prototype", "logistic_regression", "random_forest", "gradient_boosting"],
        default="prototype",
        help="Built-in model factory for CLI usage.",
    )
    plan_parser.add_argument(
        "--n-choice",
        dest="n_choices",
        action="append",
        type=_parse_n_choice,
        default=None,
        help="Repeat for integer sample counts or decimal fractions in (0, 1]; 1 means one sample, 1.0 means all.",
    )
    plan_parser.add_argument("--min-train-size", type=int, default=1)
    plan_parser.add_argument("--random-seed", type=int, default=42)
    plan_parser.add_argument("--time-cost-weight", type=float, default=1.0)
    plan_parser.add_argument("--loss-cost-weight", type=float, default=1.0)
    plan_parser.add_argument("--json", action="store_true", help="Emit the result as JSON.")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "plan":
        parser.print_help()
        return 0

    timeline = BinaryTimelineAdapter(args.dataset_dir, num_features=args.num_features)
    config = PlannerConfig(
        n_choices=args.n_choices or (0.25, 0.5, 1.0),
        min_train_size=args.min_train_size,
        random_seed=args.random_seed,
        time_cost_weight=args.time_cost_weight,
        loss_cost_weight=args.loss_cost_weight,
        bayes_backend=args.bayes_backend,
    )
    model_adapter = _build_model_adapter(args.model, args.random_seed)
    planner = RetrainingPlanner(timeline, model_adapter, config=config)
    result = planner.plan(method=args.method)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    else:
        print(f"RetrainOpt method={result.method} total_cost={result.total_cost:.6f}")
        for step, decision in enumerate(result.decisions):
            if decision.action.value == "keep":
                print(f"t={step}: keep")
            else:
                print(f"t={step}: retrain n={decision.n_samples}")
    return 0


def _build_sklearn_factory(model_name: str, random_seed: int):
    def _factory():
        try:
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The RetrainOpt CLI model presets require the `sklearn` extra."
            ) from exc

        if model_name == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=random_seed)
        if model_name == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=random_seed)
        if model_name == "gradient_boosting":
            return GradientBoostingClassifier(random_state=random_seed)
        raise ValueError(f"Unsupported CLI model preset: {model_name}")

    return _factory


def _build_model_adapter(model_name: str, random_seed: int):
    if model_name == "prototype":
        return PrototypeModelAdapter()
    return SklearnModelAdapter(_build_sklearn_factory(model_name, random_seed))
