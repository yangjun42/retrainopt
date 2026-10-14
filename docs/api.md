# API reference

The public entry points below form the supported integration surface.
Underscore-prefixed helpers and internal DP cost tables may change between
releases. Use evaluator injection instead of replacing a planner's internal
engine or calling its private table builders.

```{eval-rst}
.. autoclass:: retrainopt.PlannerConfig
   :members:

.. autoclass:: retrainopt.RetrainingPlanner
   :members: plan

.. autoclass:: retrainopt.PlanResult
   :members: to_dict

.. autoclass:: retrainopt.Decision

.. autoclass:: retrainopt.DecisionAction

.. autoclass:: retrainopt.CandidateEvaluation
   :members:

.. autoclass:: retrainopt.EvaluationProvider
   :members:

.. autoclass:: retrainopt.TimelineDataset
   :members:

.. autoclass:: retrainopt.ModelAdapter
   :members:

.. automodule:: retrainopt.adapters
   :members:
```

