"""Configuration objects for RetrainOpt."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Sequence, Union


NumberLike = Union[int, float]


@dataclass(frozen=True)
class PlannerConfig:
    """User-facing planner configuration.

    ``n_choices`` accepts float fractions in ``(0, 1]`` or integer sample
    counts ``>= 1``. Thus ``1`` means one sample and ``1.0`` means all samples.
    """

    n_choices: Sequence[NumberLike] = field(default_factory=lambda: (0.25, 0.5, 1.0))
    min_train_size: int = 1
    time_cost_weight: float = 1.0
    loss_cost_weight: float = 1.0
    random_seed: int = 42

    # If ``None``, the planner synthesizes the initial model by training on the
    # first timestep with ``initial_model_samples`` or the full first block.
    initial_loss: Optional[float] = None
    initial_model_samples: Optional[int] = None

    # Approximate "bayes" mode controls.
    bayes_iterations: int = 5
    bayes_initial_samples: int = 6
    bayes_candidates_per_iteration: int = 1
    bayes_selection_strategy: str = "sequence"
    bayes_anchor_steps: Sequence[float] = field(default_factory=lambda: (0.0, 0.5, 1.0))
    acquisition: str = "expected"
    bayes_backend: str = "lightweight"
    exploration_beta: float = 1.0
    improvement_xi: float = 0.01
    surrogate_k: int = 5
    surrogate_uncertainty_floor: float = 0.05
    bayes_backend_options: Dict[str, Any] = field(default_factory=dict)
    record_bayes_true_history: bool = False

    def __post_init__(self) -> None:
        choices = tuple(self.n_choices)
        if not choices:
            raise ValueError("n_choices must contain at least one sample count or fraction.")
        for choice in choices:
            if isinstance(choice, bool) or not isinstance(choice, Real):
                raise ValueError("n_choices must contain positive integers or float fractions in (0, 1].")
            if isinstance(choice, Integral):
                if choice < 1:
                    raise ValueError("Integer n_choices must be positive sample counts.")
            elif not math.isfinite(choice) or not 0 < choice <= 1:
                raise ValueError("Float n_choices must be finite fractions in (0, 1]; use an integer for a sample count.")
        object.__setattr__(self, "n_choices", choices)
        for name, minimum in (
            ("min_train_size", 1), ("random_seed", 0), ("bayes_iterations", 0),
            ("bayes_initial_samples", 1), ("bayes_candidates_per_iteration", 1), ("surrogate_k", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.initial_model_samples is not None and (
            isinstance(self.initial_model_samples, bool)
            or not isinstance(self.initial_model_samples, Integral)
            or self.initial_model_samples < 1
        ):
            raise ValueError("initial_model_samples must be a positive integer.")
        for name in ("time_cost_weight", "loss_cost_weight", "exploration_beta", "improvement_xi", "surrogate_uncertainty_floor", "initial_loss"):
            value = getattr(self, name)
            if value is None and name == "initial_loss":
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        anchors = tuple(self.bayes_anchor_steps)
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1 for value in anchors):
            raise ValueError("bayes_anchor_steps must contain finite fractions in [0, 1].")
        object.__setattr__(self, "bayes_anchor_steps", anchors)
        if not isinstance(self.bayes_backend_options, dict):
            raise ValueError("bayes_backend_options must be a dictionary.")
        object.__setattr__(self, "bayes_backend_options", copy.deepcopy(self.bayes_backend_options))

    def resolve_n_choices(self, dataset_size: int) -> List[int]:
        """Resolve configured sample choices for a concrete train block size."""
        if isinstance(dataset_size, bool) or not isinstance(dataset_size, Integral) or dataset_size < 0:
            raise ValueError("dataset_size must be a nonnegative integer.")
        if dataset_size < self.min_train_size:
            return []

        choices = set()
        for choice in self.n_choices:
            if isinstance(choice, Integral):
                n_value = min(dataset_size, choice)
            else:
                n_value = int(dataset_size * float(choice))

            if n_value >= self.min_train_size:
                choices.add(max(1, n_value))

        return sorted(choices)
