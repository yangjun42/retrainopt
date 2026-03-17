"""RetrainOpt public package.

RetrainOpt provides cost-aware retraining planning for time-sliced datasets.
The public API is intentionally small:

- ``TimelineDataset`` describes the retraining timeline.
- ``ModelAdapter`` abstracts model fitting and evaluation.
- ``PlannerConfig`` controls the search space and planning objective.
- ``RetrainingPlanner`` produces retraining decision sequences.
"""

from .config import PlannerConfig
from .interfaces import ModelAdapter, TimelineDataset
from .planner import RetrainingPlanner
from .types import Decision, DecisionAction, PlanResult

__version__ = "0.1.0"

__all__ = [
    "Decision",
    "DecisionAction",
    "ModelAdapter",
    "PlannerConfig",
    "PlanResult",
    "RetrainingPlanner",
    "TimelineDataset",
]
