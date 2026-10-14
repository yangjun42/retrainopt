# RetrainOpt

Offline planning for **when to retrain a model and how much data to use**,
balancing training cost and predictive loss.

RetrainOpt provides exact dynamic programming and surrogate-guided approximate
search over a supplied timeline. It implements the offline core of
*Automated Retraining for Cost-Optimal ML* (Jun Yang and Michael Mathioudakis,
2026). It is an algorithm library: deployment, online drift detection and
experiment orchestration are outside its scope.

## Install

Python 3.10 or later. The core has no mandatory third-party runtime dependencies.

```shell
python -m pip install retrainopt
```

Optional integrations:

```shell
python -m pip install 'retrainopt[sklearn]'
python -m pip install 'retrainopt[torch]'
python -m pip install 'retrainopt[botorch]'
```

## Quick start

```python
from retrainopt import PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter

timeline = ArrayTimelineAdapter(
    train_blocks=[
        [((0.0,), 0), ((1.0,), 1)],
        [((0.0,), 1), ((1.0,), 0)],
    ],
    eval_blocks=[
        [((0.1,), 0), ((0.9,), 1)],
        [((0.1,), 1), ((0.9,), 0)],
    ],
)
config = PlannerConfig(
    n_choices=(1.0,),       # Full block; integer 1 would mean one sample.
    initial_loss=1.0,      # Explicit initial baseline; no bootstrap fit.
    time_cost_weight=0.0,  # Ignore wall-clock noise in this small example.
    loss_cost_weight=1.0,
)
result = RetrainingPlanner(timeline, PrototypeModelAdapter(), config).plan("exact")
print([decision.action.value for decision in result.decisions])
print(result.total_cost)
# ['retrain', 'retrain']
# 0.0
```

Each train block is the complete available dataset at that time; blocks are
not accumulated automatically. The planner can inspect the entire timeline,
including evaluation blocks. These are optimization inputs, so the selected
plan's cost is not an independent future-performance estimate.

## Choosing a method

| Method | What it does |
| --- | --- |
| `exact` | Measures every configured candidate and finds the minimum-cost schedule over that finite candidate set. |
| `bayes`, `lightweight` backend | Uses a dependency-free nearest-neighbor heuristic to select candidates. |
| `bayes`, `botorch` backend | Uses optional Gaussian-process surrogates for acquisition-guided search. |

For `bayes`, `estimated_total_cost` is the surrogate estimate, while
`total_cost` is measured for the selected schedule. Approximate search does not
guarantee the global optimum. Sequence selection can train more models than
`bayes_candidates_per_iteration`, which is a minimum target, not a hard budget.

`PlanResult.training_cost` is weighted training cost, `loss_cost` is raw summed
loss, and `weighted_loss_cost` is the loss contribution to `total_cost`.
Planning's exploratory fits are not part of the selected schedule's objective.

## Documentation and examples

- [Quick start](https://yangjun42.github.io/retrainopt/quickstart.html), [algorithm semantics](https://yangjun42.github.io/retrainopt/algorithms.html)
- [Initial models, costs and evaluator injection](https://yangjun42.github.io/retrainopt/evaluation.html)
- [Optional adapters and CLI](https://yangjun42.github.io/retrainopt/integrations.html), [API](https://yangjun42.github.io/retrainopt/api.html)
- [Migration from 0.1](https://yangjun42.github.io/retrainopt/migration.html), [changelog](https://github.com/yangjun42/retrainopt/blob/main/CHANGELOG.md)
- [Development and validation](https://yangjun42.github.io/retrainopt/development.html), [publishing](https://github.com/yangjun42/retrainopt/blob/main/PUBLISHING.md)

```shell
python examples/minimal_array_timeline.py
python examples/verify_exact_against_bruteforce.py
# After installing the corresponding extra:
python examples/sklearn_timeline.py
python examples/torch_timeline.py
python examples/botorch_timeline.py
```

Version 0.2 corrects acquisition composition and tightens public contracts.
It does not replace the paper's frozen code or the separate canonical
`retrainopt-paper-repro` pin `ecml-pkdd-2026-camera-ready-retrainopt`.
Use [CITATION.cff](https://github.com/yangjun42/retrainopt/blob/main/CITATION.cff) to cite this software and the paper.

Apache-2.0 licensed. Source and issues:
[github.com/yangjun42/retrainopt](https://github.com/yangjun42/retrainopt).
