"""Dynamic programming over complete keep and retrain stage costs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Dict, List, Tuple

from ..types import Decision, DecisionAction, ModelId


StateCosts = Dict[ModelId, Dict[str, Any]]


@dataclass
class StageCosts:
    """Internal complete-stage costs; acquisition scores may be negative.

    A model trained at t may be kept only after t. Its cost at t belongs in
    ``retrain``, which already includes both training and evaluation.
    """

    keep: Dict[ModelId, Dict[int, float]]
    retrain: Dict[ModelId, float]


def _nonnegative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative real number.")
    return result


def _validate_horizon(num_steps: int) -> None:
    if isinstance(num_steps, bool) or not isinstance(num_steps, Integral) or num_steps < 0:
        raise ValueError("num_steps must be a nonnegative integer.")


def validate_model_costs(
    model_id: ModelId,
    info: Mapping[str, Any],
    num_steps: int,
    *,
    initial_model_id: ModelId = (-1, 0),
) -> Tuple[float, Dict[int, float]]:
    """Validate one external model cost record and return numeric components."""
    if (
        not isinstance(model_id, tuple)
        or len(model_id) != 2
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in model_id)
    ):
        raise ValueError(f"Invalid model id: {model_id!r}; expected (integer step, integer sample count).")
    is_initial = model_id == initial_model_id
    if not is_initial and not (0 <= model_id[0] < num_steps and model_id[1] > 0):
        raise ValueError(f"Candidate model id outside the timeline or nonpositive sample count: {model_id!r}.")
    if not isinstance(info, Mapping) or "error" in info:
        raise ValueError(f"Model {model_id!r} requires a valid cost record.")
    if not is_initial and "training_cost" not in info:
        raise ValueError(f"Model {model_id!r} is missing training_cost.")
    training_cost = _nonnegative_number(info.get("training_cost", 0.0), f"training_cost for {model_id!r}")
    if is_initial and training_cost != 0.0:
        raise ValueError("The initial model is a supplied baseline; its training_cost must be zero.")
    losses = info.get("future_losses")
    if not isinstance(losses, Mapping):
        raise ValueError(f"Model {model_id!r} is missing a future_losses mapping.")
    start = 0 if is_initial else model_id[0]
    expected_steps = set(range(start, num_steps))
    if any(isinstance(step, bool) or not isinstance(step, Integral) for step in losses):
        raise ValueError(f"future_losses for {model_id!r} must use integer timestep keys.")
    if set(losses) != expected_steps:
        raise ValueError(f"future_losses for {model_id!r} must cover exactly timesteps {start}..{num_steps - 1}.")
    return training_cost, {
        step: _nonnegative_number(losses[step], f"loss for model {model_id!r} at step {step}")
        for step in range(start, num_steps)
    }


def _objective_stages(
    num_steps: int,
    state_costs: StateCosts,
    loss_cost_weight: float,
    initial_model_id: ModelId,
) -> StageCosts:
    _validate_horizon(num_steps)
    weight = _nonnegative_number(loss_cost_weight, "loss_cost_weight")
    if not isinstance(state_costs, Mapping) or initial_model_id not in state_costs:
        raise ValueError("Initial model costs are missing from state_costs.")
    keep: Dict[ModelId, Dict[int, float]] = {}
    retrain: Dict[ModelId, float] = {}
    for model_id, info in state_costs.items():
        training, losses = validate_model_costs(model_id, info, num_steps, initial_model_id=initial_model_id)
        keep[model_id] = {step: weight * value for step, value in losses.items()}
        if model_id != initial_model_id:
            retrain[model_id] = training + keep[model_id][model_id[0]]
    return StageCosts(keep, retrain)


def solve_stage_schedule(
    num_steps: int,
    stages: StageCosts,
    *,
    initial_model_id: ModelId = (-1, 0),
) -> Tuple[float, List[Decision], Dict[int, Dict[ModelId, float]], Dict[int, Dict[ModelId, Decision]]]:
    """Solve additive stage costs, including signed acquisition scores."""
    if any(not math.isfinite(cost) for row in stages.keep.values() for cost in row.values()) or any(
        not math.isfinite(cost) for cost in stages.retrain.values()
    ):
        raise ValueError("Complete stage costs must be finite.")
    model_ids = list(stages.keep)
    candidates_by_step: Dict[int, List[ModelId]] = {step: [] for step in range(num_steps)}
    for model_id in stages.retrain:
        candidates_by_step[model_id[0]].append(model_id)
    cost_to_go = {num_steps: {model_id: 0.0 for model_id in model_ids}}
    decisions: Dict[int, Dict[ModelId, Decision]] = {}

    for step in range(num_steps - 1, -1, -1):
        cost_to_go[step] = {}
        decisions[step] = {}
        best_retrain_total = float("inf")
        best_retrain_decision = None
        for model_id in sorted(candidates_by_step[step]):
            total = stages.retrain[model_id] + cost_to_go[step + 1][model_id]
            if total < best_retrain_total:
                best_retrain_total = total
                best_retrain_decision = Decision(DecisionAction.RETRAIN, model_id[1], step)

        for model_id in model_ids:
            if model_id != initial_model_id and model_id[0] >= step:
                continue
            keep_total = stages.keep[model_id][step] + cost_to_go[step + 1][model_id]
            if best_retrain_decision is not None and best_retrain_total < keep_total:
                cost_to_go[step][model_id] = best_retrain_total
                decisions[step][model_id] = best_retrain_decision
            else:
                cost_to_go[step][model_id] = keep_total
                decisions[step][model_id] = Decision(DecisionAction.KEEP)

    optimal_cost = cost_to_go[0][initial_model_id]
    if not math.isfinite(optimal_cost):
        raise ValueError("Total schedule cost overflowed; rescale the cost units.")
    return optimal_cost, _backtrack(num_steps, decisions, initial_model_id), cost_to_go, decisions


def solve_exact_schedule(
    num_steps: int,
    state_costs: StateCosts,
    *,
    loss_cost_weight: float = 1.0,
    initial_model_id: ModelId = (-1, 0),
) -> Tuple[float, List[Decision], Dict[int, Dict[ModelId, float]], Dict[int, Dict[ModelId, Decision]]]:
    """Solve a complete finite objective matrix on its configured candidate grid.

    Costs and losses must be nonnegative. Every model supplies losses from its
    training step through the horizon; the initial model supplies all steps.
    Equal-cost choices keep the current model; tied retrains use the smaller n.
    """
    stages = _objective_stages(num_steps, state_costs, loss_cost_weight, initial_model_id)
    return solve_stage_schedule(num_steps, stages, initial_model_id=initial_model_id)


def _backtrack(
    num_steps: int,
    decisions: Dict[int, Dict[ModelId, Decision]],
    initial_model_id: ModelId,
) -> List[Decision]:
    sequence = []
    current_model_id = initial_model_id
    for step in range(num_steps):
        decision = decisions[step][current_model_id]
        sequence.append(decision)
        if decision.action == DecisionAction.RETRAIN:
            current_model_id = (step, decision.n_samples)
    return sequence


def evaluate_decision_sequence(
    decisions: List[Decision],
    state_costs: StateCosts,
    *,
    loss_cost_weight: float = 1.0,
    initial_model_id: ModelId = (-1, 0),
) -> Tuple[float, float, float]:
    """Return (total cost, weighted training cost, unweighted loss sum)."""
    stages = _objective_stages(len(decisions), state_costs, loss_cost_weight, initial_model_id)
    total_cost = training_total = loss_total = 0.0
    current_model_id = initial_model_id
    for step, decision in enumerate(decisions):
        if not isinstance(decision, Decision):
            raise ValueError(f"Invalid decision at step {step}: {decision!r}.")
        if decision.action == DecisionAction.RETRAIN:
            n = decision.n_samples
            if decision.trained_at is not None and decision.trained_at != step:
                raise ValueError(f"Retrain trained_at disagrees with decision step {step}.")
            current_model_id = (step, n)
            if current_model_id not in stages.retrain:
                raise ValueError(f"No candidate costs for retrain decision {current_model_id!r}.")
            training_total += float(state_costs[current_model_id]["training_cost"])
            total_cost += stages.retrain[current_model_id]
        else:
            total_cost += stages.keep[current_model_id][step]
        loss_total += float(state_costs[current_model_id]["future_losses"][step])
    if not all(math.isfinite(value) for value in (total_cost, training_total, loss_total)):
        raise ValueError("Total schedule cost overflowed; rescale the cost units.")
    return total_cost, training_total, loss_total
