"""Cost composition helpers."""

from __future__ import annotations


def compute_training_cost(training_time_seconds: float, time_cost_weight: float = 1.0) -> float:
    """Map raw training time into planner cost units."""
    return float(training_time_seconds) * float(time_cost_weight)


def compute_step_cost(training_cost: float, loss_value: float, loss_cost_weight: float = 1.0) -> float:
    """Compose training and loss terms into a scalar step cost."""
    return float(training_cost) + float(loss_cost_weight) * float(loss_value)

