"""RetrainOpt public package.

RetrainOpt provides cost-aware retraining planning for time-sliced datasets.
The public API is intentionally small:

- ``TimelineDataset`` describes the retraining timeline.
- ``ModelAdapter`` abstracts model fitting and evaluation.
- ``PlannerConfig`` controls the search space and planning objective.
- ``RetrainingPlanner`` produces retraining decision sequences.
"""

from importlib.metadata import version

from .config import PlannerConfig
from .interfaces import EvaluationProvider, ModelAdapter, TimelineDataset
from .planner import RetrainingPlanner
from .types import CandidateEvaluation, Decision, DecisionAction, PlanResult

__version__ = version("retrainopt")

__all__ = [
    "Decision",
    "CandidateEvaluation",
    "DecisionAction",
    "EvaluationProvider",
    "ModelAdapter",
    "PlannerConfig",
    "PlanResult",
    "RetrainingPlanner",
    "TimelineDataset",
]
