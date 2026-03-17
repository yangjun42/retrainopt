"""Minimal RetrainOpt example using in-memory Python data."""

from retrainopt import PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter


def main() -> None:
    timeline = ArrayTimelineAdapter(
        train_blocks=[
            [((0.0,), 0), ((1.0,), 1)],
            [((0.1,), 0), ((0.9,), 1)],
            [((0.0,), 1), ((1.0,), 0)],
        ],
        eval_blocks=[
            [((0.05,), 0), ((0.95,), 1)],
            [((0.15,), 0), ((0.85,), 1)],
            [((0.05,), 1), ((0.95,), 0)],
        ],
    )

    planner = RetrainingPlanner(
        timeline,
        PrototypeModelAdapter(),
        config=PlannerConfig(
            n_choices=(0.5, 1.0),
            time_cost_weight=0.0,
            loss_cost_weight=1.0,
            bayes_iterations=2,
        ),
    )

    exact_result = planner.plan(method="exact")
    bayes_result = planner.plan(method="bayes")

    print("Exact")
    print(exact_result.to_dict())
    print()
    print("Bayes")
    print(bayes_result.to_dict())


if __name__ == "__main__":
    main()
