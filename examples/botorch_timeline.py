"""Tiny GP-guided search. Install retrainopt[botorch] first.

This demonstrates the backend; it is not a paper-reproduction experiment.
"""

from retrainopt import PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter


def main() -> None:
    first = [((0.0,), 0), ((0.2,), 0), ((0.8,), 1), ((1.0,), 1)]
    flipped = [(features, 1 - label) for features, label in first]
    blocks = [first, first, flipped]
    timeline = ArrayTimelineAdapter(blocks, blocks)
    config = PlannerConfig(
        n_choices=(2, 4), initial_loss=1.0, time_cost_weight=0.0,
        bayes_backend="botorch", bayes_anchor_steps=(),
        bayes_initial_samples=2, bayes_iterations=1,
        bayes_selection_strategy="sequence", acquisition="lcb",
    )
    result = RetrainingPlanner(timeline, PrototypeModelAdapter(), config).plan("bayes")
    print([decision.action.value for decision in result.decisions])
    print(f"Estimated objective: {result.estimated_total_cost:.6f}")
    print(f"Measured schedule objective: {result.total_cost:.6f}")


if __name__ == "__main__":
    main()
