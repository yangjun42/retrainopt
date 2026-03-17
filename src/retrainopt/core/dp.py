"""Dynamic-programming schedule solver."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..types import Decision, DecisionAction, ModelId
from .costs import compute_step_cost


StateCosts = Dict[ModelId, Dict[str, Any]]


def solve_exact_schedule(
    num_steps: int,
    state_costs: StateCosts,
    *,
    loss_cost_weight: float = 1.0,
    initial_model_id: ModelId = (-1, 0),
) -> Tuple[float, List[Decision], Dict[int, Dict[ModelId, float]], Dict[int, Dict[ModelId, Decision]]]:
    """Solve the exact retraining schedule with backward dynamic programming."""
    if initial_model_id not in state_costs:
        raise ValueError("Initial model costs are missing from state_costs.")

    valid_model_ids = [
        model_id for model_id, info in state_costs.items() if "error" not in info and "future_losses" in info
    ]
    trained_model_ids = {model_id for model_id in valid_model_ids if model_id != initial_model_id}

    cost_to_go: Dict[int, Dict[ModelId, float]] = {num_steps: {model_id: 0.0 for model_id in valid_model_ids}}
    decisions: Dict[int, Dict[ModelId, Decision]] = {}

    for step in range(num_steps - 1, -1, -1):
        cost_to_go[step] = {}
        decisions[step] = {}

        best_retrain_total = float("inf")
        best_retrain_decision: Decision | None = None

        for n_value in sorted(n for t_value, n in trained_model_ids if t_value == step):
            model_id = (step, n_value)
            info = state_costs.get(model_id, {})
            training_cost = float(info.get("training_cost", float("inf")))
            immediate_loss = float(info.get("future_losses", {}).get(step, float("inf")))
            future_cost = cost_to_go[step + 1].get(model_id, float("inf"))
            total = compute_step_cost(training_cost, immediate_loss, loss_cost_weight) + future_cost
            if total < best_retrain_total:
                best_retrain_total = total
                best_retrain_decision = Decision(
                    action=DecisionAction.RETRAIN,
                    n_samples=n_value,
                    trained_at=step,
                )

        if best_retrain_decision is not None:
            retrained_id = (step, int(best_retrain_decision.n_samples or 0))
            retrained_loss = float(state_costs[retrained_id]["future_losses"].get(step, float("inf")))
            retrained_future = cost_to_go[step + 1].get(retrained_id, float("inf"))
            cost_to_go[step][retrained_id] = compute_step_cost(0.0, retrained_loss, loss_cost_weight) + retrained_future

        possible_current_models = [initial_model_id] + [
            model_id for model_id in trained_model_ids if model_id[0] < step
        ]
        for current_model_id in possible_current_models:
            info = state_costs.get(current_model_id, {})
            keep_loss = float(info.get("future_losses", {}).get(step, float("inf")))
            keep_total = compute_step_cost(0.0, keep_loss, loss_cost_weight) + cost_to_go[step + 1].get(
                current_model_id, float("inf")
            )

            if best_retrain_decision is not None and best_retrain_total < keep_total:
                cost_to_go[step][current_model_id] = best_retrain_total
                decisions[step][current_model_id] = best_retrain_decision
            else:
                cost_to_go[step][current_model_id] = keep_total
                decisions[step][current_model_id] = Decision(action=DecisionAction.KEEP)

    optimal_cost = cost_to_go[0][initial_model_id]
    optimal_sequence = _backtrack(num_steps, decisions, initial_model_id)
    return optimal_cost, optimal_sequence, cost_to_go, decisions


def _backtrack(
    num_steps: int,
    decisions: Dict[int, Dict[ModelId, Decision]],
    initial_model_id: ModelId,
) -> List[Decision]:
    sequence: List[Decision] = []
    current_model_id = initial_model_id
    for step in range(num_steps):
        decision = decisions[step].get(current_model_id)
        if decision is None:
            raise ValueError(
                f"Missing decision during backtracking at step {step} for model {current_model_id}."
            )
        sequence.append(decision)
        if decision.action == DecisionAction.RETRAIN:
            current_model_id = (step, int(decision.n_samples or 0))
    return sequence


def evaluate_decision_sequence(
    decisions: List[Decision],
    state_costs: StateCosts,
    *,
    loss_cost_weight: float = 1.0,
    initial_model_id: ModelId = (-1, 0),
) -> Tuple[float, float, float]:
    """Evaluate a decision sequence against a concrete cost matrix."""
    total_cost = 0.0
    training_cost_total = 0.0
    loss_total = 0.0
    current_model_id = initial_model_id

    for step, decision in enumerate(decisions):
        if decision.action == DecisionAction.KEEP:
            info = state_costs[current_model_id]
            loss_value = float(info["future_losses"].get(step, float("inf")))
            step_cost = compute_step_cost(0.0, loss_value, loss_cost_weight)
            total_cost += step_cost
            loss_total += loss_value
            continue

        new_model_id = (step, int(decision.n_samples or 0))
        info = state_costs[new_model_id]
        training_cost = float(info.get("training_cost", float("inf")))
        loss_value = float(info["future_losses"].get(step, float("inf")))
        step_cost = compute_step_cost(training_cost, loss_value, loss_cost_weight)
        total_cost += step_cost
        training_cost_total += training_cost
        loss_total += loss_value
        current_model_id = new_model_id

    return total_cost, training_cost_total, loss_total
