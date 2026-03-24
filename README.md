# RetrainOpt

Cost-aware retraining planning.

`RetrainOpt` is a standalone Python library for deciding when to retrain and
how much data to use at each retraining step on a time-sliced dataset.

License:

- Apache-2.0

Repository:

- [github.com/yangjun42/retrainopt](https://github.com/yangjun42/retrainopt)

## Requirements

- Python `>=3.10`

## Install

Minimal install:

```shell
pip install .
```

Editable install:

```shell
pip install -e .
```

Optional extras:

```shell
pip install -e .[sklearn]
pip install -e .[torch]
pip install -e .[botorch]
```

## Public API

Main concepts:

- `TimelineDataset`: your time-sliced train/eval data
- `ModelAdapter`: how a candidate model is trained and evaluated
- `PlannerConfig`: search space and objective weights
- `RetrainingPlanner`: produces retraining decision sequences

Minimal import:

```python
from retrainopt import PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter
```

## Quick Start

```python
from retrainopt import PlannerConfig, RetrainingPlanner
from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter

timeline = ArrayTimelineAdapter(
    train_blocks=[
        [((0.0,), 0), ((1.0,), 1)],
        [((0.1,), 0), ((0.9,), 1)],
    ],
    eval_blocks=[
        [((0.05,), 0), ((0.95,), 1)],
        [((0.15,), 0), ((0.85,), 1)],
    ],
)

planner = RetrainingPlanner(
    timeline,
    PrototypeModelAdapter(),
    config=PlannerConfig(n_choices=(1.0,), time_cost_weight=0.0, loss_cost_weight=1.0),
)

result = planner.plan(method="exact")
print(result.to_dict())
```

There is also a runnable example in
[`examples/minimal_array_timeline.py`](examples/minimal_array_timeline.py).
For a stricter correctness check, run
[`examples/verify_exact_against_bruteforce.py`](examples/verify_exact_against_bruteforce.py),
which verifies that `method="exact"` matches a brute-force optimum on a tiny
synthetic timeline.

## CLI

```shell
retrainopt --help
retrainopt plan --help
```

The built-in CLI consumes a binary timeline directory with:

- `train/*.bin`
- `test/*.bin`
- optional `metadata.json` with `{"num_features": ...}`

## Bayes Backends

- `lightweight`: dependency-free nearest-neighbor surrogate, fast startup
- `botorch`: GP-based surrogate backend used by the paper-oriented workflow and
  validated in CI via the `botorch` extra

`bayes` planning returns an estimated search result first and then exactly
re-evaluates the selected decision sequence before returning `total_cost`.

## Built-in Adapters

- `ArrayTimelineAdapter` for in-memory Python data
- `BinaryTimelineAdapter` for `train/*.bin` and `test/*.bin` directories
- `TorchTimelineAdapter` for PyTorch datasets
- `PrototypeModelAdapter` for zero-dependency experimentation
- `SklearnModelAdapter` and `TorchModelAdapter` as optional integrations

## Tests

Run the standalone test suite with:

```shell
python -m unittest tests.test_retrainopt_planner tests.test_retrainopt_smoke tests.test_retrainopt_example
python examples/verify_exact_against_bruteforce.py
python -m unittest tests.test_retrainopt_botorch
```

The last command requires `pip install -e .[botorch]`.

## Offline Paper Reproduction

Phase 1 adds an internal offline reproduction harness under `retrainopt.paper_repro`.
It is intentionally separate from the public planner API and is meant for
latest-paper offline studies only.

List the bundled Phase 1 runs:

```shell
python -m retrainopt.paper_repro.cli list
```

Run one manifest entry and write normalized artifacts:

```shell
python -m retrainopt.paper_repro.cli run \
  --run-id phase1_exact_extreme_sudden_M_seed42 \
  --artifact-root ./artifacts/offline-phase1
```

Render a Puhti batch script for one run:

```shell
python -m retrainopt.paper_repro.cli render-puhti \
  --run-id phase1_exact_extreme_sudden_M_seed42 \
  --artifact-root /scratch/project_2005865/jun/retrainopt-artifacts \
  --project-root /scratch/project_2005865/jun/retrainopt
```

Coverage and explicit Phase 2 omissions are documented in
[`docs/paper_repro_coverage.md`](docs/paper_repro_coverage.md).

## Release

For GitHub and PyPI release steps, see
[`PUBLISHING.md`](PUBLISHING.md).
