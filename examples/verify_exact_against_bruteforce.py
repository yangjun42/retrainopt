"""Verify that RetrainOpt exact planning matches brute-force search."""

from __future__ import annotations

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from retrainopt import PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter
from retrainopt.core.costs import compute_step_cost
from retrainopt.interfaces import ModelAdapter


class PrototypeClassifierAdapter(ModelAdapter):
    """Tiny dependency-free classifier for correctness verification."""

    def fit(self, train_data, *, step, n_samples, random_seed):
        if not train_data:
            return {"prototypes": {}, "majority": 0}

        grouped = {}
        counts = {}
        for features, label in train_data:
            grouped.setdefault(int(label), []).append(float(features[0]))
            counts[int(label)] = counts.get(int(label), 0) + 1

        prototypes = {
            label: sum(values) / len(values)
            for label, values in grouped.items()
        }
        majority = max(counts.items(), key=lambda item: item[1])[0]
        return {"prototypes": prototypes, "majority": majority}

    def evaluate(self, model, eval_data, *, step):
        if not eval_data:
            return 0.0

        mistakes = 0
        prototypes = model["prototypes"]
        majority = model["majority"]
        for features, label in eval_data:
            feature_value = float(features[0])
            if prototypes:
                predicted = min(
                    prototypes.items(),
                    key=lambda item: abs(item[1] - feature_value),
                )[0]
            else:
                predicted = majority
            if int(predicted) != int(label):
                mistakes += 1
        return mistakes / len(eval_data)


def brute_force_cost(state_costs, num_steps, loss_weight, initial_model_id=(-1, 0)):
    best_cost = float("inf")

    choices_by_step = {}
    for model_id in state_costs:
        if model_id == initial_model_id:
            continue
        choices_by_step.setdefault(model_id[0], []).append(model_id[1])

    def recurse(step, current_model_id, running_cost):
        nonlocal best_cost
        if step == num_steps:
            best_cost = min(best_cost, running_cost)
            return

        keep_loss = state_costs[current_model_id]["future_losses"].get(step, float("inf"))
        recurse(
            step + 1,
            current_model_id,
            running_cost + compute_step_cost(0.0, keep_loss, loss_weight),
        )

        for n_value in choices_by_step.get(step, []):
            model_id = (step, n_value)
            info = state_costs[model_id]
            retrain_cost = info["training_cost"]
            loss_value = info["future_losses"].get(step, float("inf"))
            recurse(
                step + 1,
                model_id,
                running_cost + compute_step_cost(retrain_cost, loss_value, loss_weight),
            )

    recurse(0, initial_model_id, 0.0)
    return best_cost


def main() -> None:
    timeline = ArrayTimelineAdapter(
        train_blocks=[
            [((0.0,), 0), ((0.2,), 0), ((0.8,), 1), ((1.0,), 1)],
            [((0.1,), 0), ((0.3,), 0), ((0.7,), 1), ((0.9,), 1)],
            [((0.0,), 1), ((0.2,), 1), ((0.8,), 0), ((1.0,), 0)],
        ],
        eval_blocks=[
            [((0.05,), 0), ((0.95,), 1)],
            [((0.15,), 0), ((0.85,), 1)],
            [((0.05,), 1), ((0.95,), 0)],
        ],
    )

    config = PlannerConfig(
        n_choices=(0.5, 1.0),
        min_train_size=1,
        time_cost_weight=0.0,
        loss_cost_weight=1.0,
        random_seed=7,
    )
    planner = RetrainingPlanner(timeline, PrototypeClassifierAdapter(), config=config)
    result = planner.plan(method="exact")
    state_costs = planner._build_exact_state_costs()
    brute_force = brute_force_cost(
        state_costs=state_costs,
        num_steps=timeline.num_steps,
        loss_weight=config.loss_cost_weight,
    )

    print("Exact result")
    print(result.to_dict())
    print()
    print(f"Brute-force optimum: {brute_force:.6f}")

    if abs(result.total_cost - brute_force) > 1e-9:
        raise SystemExit(
            f"Verification failed: exact={result.total_cost:.12f}, brute_force={brute_force:.12f}"
        )

    print("VERIFIED: exact planner matches brute-force optimum")


if __name__ == "__main__":
    main()
