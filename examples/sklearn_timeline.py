"""CPU-only scikit-learn example. Install retrainopt[sklearn] first."""

from sklearn.linear_model import LogisticRegression

from retrainopt import PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter, SklearnModelAdapter


def main() -> None:
    first = [((-2.0,), 0), ((-1.0,), 0), ((1.0,), 1), ((2.0,), 1)]
    second = [(features, 1 - label) for features, label in first]
    timeline = ArrayTimelineAdapter([first, second], [first, second])
    adapter = SklearnModelAdapter(LogisticRegression(random_state=42))
    config = PlannerConfig(n_choices=(1.0,), initial_loss=1.0, time_cost_weight=0.0)
    result = RetrainingPlanner(timeline, adapter, config).plan("exact")
    print([decision.action.value for decision in result.decisions])
    print(f"Selected schedule loss: {result.loss_cost:.6f}")


if __name__ == "__main__":
    main()
