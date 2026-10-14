# Quick start

Python 3.10 or later is required. The core package has no mandatory third-party
runtime dependencies.

```shell
python -m pip install retrainopt
```

Each train block describes the data available for a candidate retraining at
that time. Blocks are not accumulated automatically: supply cumulative blocks
yourself if that is the intended experiment. Each evaluation block describes
the data on which that time's modeling loss is measured.

This example uses a constant initial loss, so no model is silently trained
before planning. Setting the training-time weight to zero makes the objective
independent of wall-clock timing in this tiny demonstration.

```{doctest}
>>> from retrainopt import PlannerConfig, RetrainingPlanner
>>> from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter
>>> timeline = ArrayTimelineAdapter(
...     train_blocks=[
...         [((0.0,), 0), ((1.0,), 1)],
...         [((0.0,), 1), ((1.0,), 0)],
...     ],
...     eval_blocks=[
...         [((0.1,), 0), ((0.9,), 1)],
...         [((0.1,), 1), ((0.9,), 0)],
...     ],
... )
>>> config = PlannerConfig(
...     n_choices=(1.0,), initial_loss=1.0,
...     time_cost_weight=0.0, loss_cost_weight=1.0,
... )
>>> result = RetrainingPlanner(timeline, PrototypeModelAdapter(), config).plan("exact")
>>> [decision.action.value for decision in result.decisions]
['retrain', 'retrain']
>>> result.total_cost
0.0
>>> result.to_dict()["schema_version"]
2
```

`n_choices=(1.0,)` means the full block. Integer `1` means one sample;
fractional choices in `(0, 1]` are resolved per block. `min_train_size` controls
the smallest candidate. Exact planning is optimal over these configured
candidates and the measured cost table, not over all conceivable models or
data-selection strategies.

For an approximate search, call `plan("bayes")`. Its default `lightweight`
backend uses a nearest-neighbor heuristic. For Gaussian-process surrogates,
install `retrainopt[botorch]` and set `bayes_backend="botorch"`.

The repository contains runnable examples:

```shell
python examples/minimal_array_timeline.py
python examples/verify_exact_against_bruteforce.py
```

The first prints both planners' schedules. The second independently enumerates
small schedules and checks the exact planner's objective.

