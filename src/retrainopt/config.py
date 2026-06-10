"""Configuration objects for RetrainOpt."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union


NumberLike = Union[int, float]


@dataclass
class PlannerConfig:
    """User-facing planner configuration.

    ``n_choices`` accepts either fractions in ``(0, 1]`` or absolute sample
    counts ``>= 1``. Fractions are resolved against each timestep's train size.
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

    def resolve_n_choices(self, dataset_size: int) -> List[int]:
        """Resolve configured sample choices for a concrete train block size."""
        if dataset_size < self.min_train_size:
            return []

        choices = set()
        for choice in self.n_choices:
            if isinstance(choice, bool):
                continue
            if isinstance(choice, int) and choice >= 1:
                n_value = min(dataset_size, choice)
            else:
                fraction = float(choice)
                if fraction <= 0:
                    continue
                if fraction <= 1.0:
                    n_value = int(dataset_size * fraction)
                else:
                    n_value = min(dataset_size, int(fraction))

            if n_value >= self.min_train_size:
                choices.add(max(1, n_value))

        if not choices and dataset_size >= self.min_train_size:
            choices.add(dataset_size)
        return sorted(choices)
