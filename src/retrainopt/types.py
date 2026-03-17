"""Shared public types for RetrainOpt."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


ModelId = Tuple[int, int]


class DecisionAction(str, Enum):
    """Supported planner actions."""

    KEEP = "keep"
    RETRAIN = "retrain"


@dataclass(frozen=True)
class Decision:
    """A single retraining decision."""

    action: DecisionAction
    n_samples: Optional[int] = None
    trained_at: Optional[int] = None


@dataclass
class PlanResult:
    """Structured planner output."""

    method: str
    decisions: List[Decision]
    total_cost: float
    training_cost: float
    loss_cost: float
    estimated_total_cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a plain dictionary."""
        return {
            "method": self.method,
            "decisions": [
                {
                    "action": decision.action.value,
                    "n_samples": decision.n_samples,
                    "trained_at": decision.trained_at,
                }
                for decision in self.decisions
            ],
            "total_cost": self.total_cost,
            "training_cost": self.training_cost,
            "loss_cost": self.loss_cost,
            "estimated_total_cost": self.estimated_total_cost,
            "metadata": dict(self.metadata),
        }


@dataclass
class CandidateEvaluation:
    """Observed or estimated cost information for a candidate model."""

    model_id: ModelId
    dataset_size: int
    training_seconds: float
    training_cost: float
    future_losses: Dict[int, float]
    observed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_state_cost(self) -> Dict[str, Any]:
        """Convert to the internal DP input format."""
        return {
            "training_seconds": self.training_seconds,
            "training_cost": self.training_cost,
            "future_losses": dict(self.future_losses),
            "observed": self.observed,
            "metadata": dict(self.metadata),
        }
