# RetrainOpt

Decide **when to retrain a model and how much data to use**,
balancing training cost and predictive loss.

RetrainOpt combines exact dynamic programming and surrogate-guided offline
search with the Base and Learned online retraining strategies. It implements
algorithms from [*Automated Retraining for Cost-Optimal ML*](https://link.springer.com/chapter/10.1007/978-3-032-37654-1_14)
by Jun Yang and Michael Mathioudakis (ECML PKDD 2026).

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

## Offline quick start

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
not accumulated automatically. The offline planner can inspect the entire timeline,
including evaluation blocks. These are optimization inputs, so the selected
plan's cost is not an independent future-performance estimate.

## Offline planning methods

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

## Online retraining

The Base and Learned policies use an observed historical prefix to guide
subsequent retraining decisions. Base derives a retraining interval and sample
count from a historical plan. Learned fits a classifier for keep/retrain and
a regressor for the retraining sample count; it requires the `sklearn` extra.

`collect_online_training_data(..., history_steps=...)` limits policy training
to that prefix. `OnlineController.step()` then processes one deployment step
at a time, using the current model's measured loss and executing the policy's
decision. Supply the initial deployed model and its reference loss explicitly;
policy training does not deploy a historical candidate model automatically.
Current loss measurements require labels for the evaluation observations.

```shell
python examples/online_timeline.py
# Requires retrainopt[sklearn]:
python examples/learned_online_timeline.py
```

See [Online retraining](https://yangjun42.github.io/retrainopt/online.html)
for the API, initialization choices and policy assumptions. Neither online
strategy guarantees the globally optimal deployment schedule.

## Documentation and examples

- [Quick start](https://yangjun42.github.io/retrainopt/quickstart.html), [algorithm semantics](https://yangjun42.github.io/retrainopt/algorithms.html)
- [Initial models, costs and evaluator injection](https://yangjun42.github.io/retrainopt/evaluation.html)
- [Optional adapters and CLI](https://yangjun42.github.io/retrainopt/integrations.html), [API](https://yangjun42.github.io/retrainopt/api.html)
- [Migration guide](https://yangjun42.github.io/retrainopt/migration.html), [changelog](https://github.com/yangjun42/retrainopt/blob/main/CHANGELOG.md)
- [Development and validation](https://yangjun42.github.io/retrainopt/development.html), [publishing](https://github.com/yangjun42/retrainopt/blob/main/PUBLISHING.md)

```shell
python examples/minimal_array_timeline.py
python examples/verify_exact_against_bruteforce.py
# After installing the corresponding extra:
python examples/sklearn_timeline.py
python examples/torch_timeline.py
python examples/botorch_timeline.py
```

## Citation

If you use RetrainOpt in your work, please cite
[*Automated Retraining for Cost-Optimal ML*](https://link.springer.com/chapter/10.1007/978-3-032-37654-1_14).
The paper is included in the [ECML PKDD 2026 proceedings](https://ecmlpkdd.org/2026/proceedings/);
Springer lists its citation year as 2027. Machine-readable metadata is available in
[CITATION.cff](https://github.com/yangjun42/retrainopt/blob/main/CITATION.cff).
Also record the exact RetrainOpt version or release used in your work.

```bibtex
@inproceedings{yang2027automated,
  author = {Yang, Jun and Mathioudakis, Michael},
  title = {Automated Retraining for Cost-Optimal ML},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track},
  year = {2027},
  series = {Lecture Notes in Computer Science},
  volume = {16941},
  pages = {231--249},
  publisher = {Springer Nature Switzerland},
  address = {Cham},
  doi = {10.1007/978-3-032-37654-1_14},
  url = {https://doi.org/10.1007/978-3-032-37654-1_14}
}
```

Apache-2.0 licensed. Source and issues:
[github.com/yangjun42/retrainopt](https://github.com/yangjun42/retrainopt).
