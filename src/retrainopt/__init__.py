"""RetrainOpt public package.

RetrainOpt provides cost-aware retraining planning for time-sliced datasets.
The public API is intentionally small:

- ``TimelineDataset`` describes the retraining timeline.
- ``ModelAdapter`` abstracts model fitting and evaluation.
- ``PlannerConfig`` controls the search space and planning objective.
- ``RetrainingPlanner`` produces retraining decision sequences.
- ``collect_online_training_data`` tunes on a bounded historical prefix.
- ``OnlineController`` deploys Base or Learned policies on current data.
"""

from importlib.metadata import version

from .config import PlannerConfig
from .interfaces import EvaluationProvider, ModelAdapter, TimelineDataset
from .planner import RetrainingPlanner
from .online import (
    BaseOnlinePolicy,
    LearnedOnlinePolicy,
    OnlineController,
    OnlinePolicy,
    OnlineState,
    OnlineStepResult,
    OnlineTrainingData,
    collect_online_training_data,
)
from .types import CandidateEvaluation, Decision, DecisionAction, PlanResult

__version__ = version("retrainopt")

__all__ = [
    "BaseOnlinePolicy",
    "Decision",
    "CandidateEvaluation",
    "DecisionAction",
    "EvaluationProvider",
    "ModelAdapter",
    "LearnedOnlinePolicy",
    "OnlineController",
    "OnlinePolicy",
    "OnlineState",
    "OnlineStepResult",
    "OnlineTrainingData",
    "PlannerConfig",
    "PlanResult",
    "RetrainingPlanner",
    "TimelineDataset",
    "collect_online_training_data",
]
