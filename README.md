# RetrainOpt

**Cost-aware retraining schedules for machine learning.**

RetrainOpt helps you choose **when to retrain** and **how much data to use**,
balancing model quality against training cost. Plan a schedule from historical
data, or make retraining decisions as new data arrives.

[Try the interactive demo](https://yangjun42.github.io/retrainopt/_static/demo/index.html)
to see how changing training costs and data drift changes a retraining schedule.

[![Demo preview on a synthetic timeline: cost comparison and an optimized schedule with two retraining steps using 100% and 75% of available data.](https://raw.githubusercontent.com/yangjun42/retrainopt/main/docs/_static/demo/preview.jpg)](https://yangjun42.github.io/retrainopt/_static/demo/index.html)

Based on [*Automated Retraining for Cost-Optimal ML*](https://link.springer.com/chapter/10.1007/978-3-032-37654-1_14)
by Jun Yang and Michael Mathioudakis (ECML PKDD 2026).

## Install

Requires Python 3.10 or later. The core package has no external dependencies.

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

For `bayes`, the final schedule is reoptimized over all measured candidates.
`estimated_total_cost` is its estimate before the final measurements, while
`total_cost` is its measured cost. Approximate search does not
guarantee the global optimum. See the [algorithm guide](https://yangjun42.github.io/retrainopt/algorithms.html)
for cost accounting and search budgets.

## Online retraining

The Base and Learned policies use an initial period of historical data to guide
later retraining decisions. Base derives a retraining interval and sample
count from a historical plan. Learned fits a classifier for keep/retrain and
a regressor for the retraining sample count; it requires the `sklearn` extra.

`collect_online_training_data(..., history_steps=...)` limits policy training
to that period. `OnlineController.step()` then processes one deployment step
at a time, using the current model's measured loss and executing the policy's
decision. Supply the initial deployed model and its reference loss explicitly;
policy training does not deploy a historical candidate model automatically.
Current loss measurements require labels for the evaluation observations.
Custom training-cost functions and pre-window model history are supported in
historical tuning and deployment.

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
- [Development and validation](https://yangjun42.github.io/retrainopt/development.html), [publishing](https://yangjun42.github.io/retrainopt/development.html#publishing)

```shell
python examples/minimal_array_timeline.py
python examples/verify_exact_against_bruteforce.py
# After installing the corresponding extra:
python examples/sklearn_timeline.py
python examples/torch_timeline.py
python examples/botorch_timeline.py
```

## License

RetrainOpt is licensed under the
[Apache License 2.0](https://github.com/yangjun42/retrainopt/blob/main/LICENSE).
Copyright (c) 2026 Jun Yang and Michael Mathioudakis.

## Citation

If you use RetrainOpt in your research, please cite our paper in the
[ECML PKDD 2026 proceedings](https://ecmlpkdd.org/2026/proceedings/)
and report the software version you used. Citation metadata is available in
[CITATION.cff](https://github.com/yangjun42/retrainopt/blob/main/CITATION.cff).

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
