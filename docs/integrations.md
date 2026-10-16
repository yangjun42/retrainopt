# Integrations and examples

## Model and timeline adapters

`TimelineDataset` supplies train/evaluation blocks, candidate train-set sizes
and reproducible subsampling. `ModelAdapter.fit()` produces a new model for a
candidate; `evaluate()` returns a scalar loss, with lower values better.

| Integration | Use |
| --- | --- |
| `ArrayTimelineAdapter` | In-memory sequences of samples. |
| `BinaryTimelineAdapter` | The documented binary train/test directory layout. |
| `TorchTimelineAdapter` | PyTorch datasets, including reproducible subset selection. |
| `PrototypeModelAdapter` | Tiny dependency-free classification examples. |
| `SklearnModelAdapter` | A scikit-learn estimator or factory; classification loss. |
| `TorchModelAdapter` | A PyTorch classification model factory, optimizer and fit settings. |

The built-in classifiers use labeled samples `(features, label)`. Custom
adapters can support other data structures and modeling tasks. Keep
preprocessing fitted on training data; fitting it on the full evaluation
timeline changes the experiment and can leak evaluation information.

Prototype and sklearn adapters accept integer class labels, including finite
integer-valued real numbers such as `1.0`. Fractional, non-finite, boolean and
string labels raise `ValueError`. Prototype features must be finite real
scalars or nonempty flat lists/tuples; every sample must have the same dimension,
including evaluation samples. The sklearn estimator and PyTorch model retain
their own feature and target requirements.

### scikit-learn class coverage

For probability classifiers, declare the task's class space when a training
sample may omit classes or later evaluation blocks may contain additional
classes:

```python
from sklearn.ensemble import RandomForestClassifier
from retrainopt.adapters import SklearnModelAdapter

adapter = SklearnModelAdapter(RandomForestClassifier(), classes=(0, 1, 2))
```

`classes` must contain at least two distinct integer class labels. The adapter
copies and sorts them, aligns `predict_proba` columns with that class space,
and assigns zero probability to classes absent from the fitted model.
Log loss clips zero probabilities to the prediction dtype's machine precision,
so a missing class incurs a large finite loss. Supply classes from the task's
known label schema before planning; the adapter never discovers them by
reading future evaluation blocks. Training, model and evaluation labels
outside the declared space raise an error.

Without `classes`, log loss uses the fitted model's classes, requires at least
two of them, and rejects evaluation labels the model has not learned. These
checks apply consistently across supported scikit-learn versions. Classifiers
without `predict_proba` retain their classification error-rate loss.

The default timeline sampler is uniform, without class stratification. Small
candidates can therefore contain a single class. A random forest can fit such
a candidate and be scored using declared classes; an estimator such as logistic
regression still requires multiple training classes. Declaring `classes` does
not change its fitting requirements or add training samples. A failed fit
stops planning and retains the original error, with timestep and sample-count
context for single-class candidates. Choose feasible candidate sizes or supply
a `TimelineDataset.sample_train_block()` implementation with the sampling
policy your task needs; the adapter does not retry or silently resample.

### PyTorch devices

`device="cuda"` resolves to the current CUDA device at the start of each adapter
call. Training uses that same device for model and batch placement, RNG seeding
and restoration, and synchronization. Use an explicit device such as
`device="cuda:1"` when calls must stay on a particular GPU even if the process's
current device changes. An externally supplied model must already be on the
adapter's device when evaluated.

The optional examples use synthetic, in-memory data and run on CPU:

```shell
python -m pip install 'retrainopt[sklearn]'
python examples/sklearn_timeline.py

python -m pip install 'retrainopt[torch]'
python examples/torch_timeline.py

python -m pip install 'retrainopt[botorch]'
python examples/botorch_timeline.py
```

They exercise adapters and imports, not the published paper's large-scale
benchmarks. Fixed seeds control supported random sources; floating-point
results, model-library versions, and training timings may still differ.

## Command line

```shell
retrainopt --help
retrainopt plan --help
```

The CLI accepts a directory with matching time slices:

```text
timeline/
  metadata.json
  train/0.bin
  train/1.bin
  test/0.bin
  test/1.bin
```

`metadata.json` supplies `{"num_features": 2}`, or pass the feature count using
the CLI option shown by `--help`. Each binary record contains `num_features`
features followed by one label, all encoded as little-endian 32-bit floats.
Train and evaluation slices must align; malformed records and invalid metadata
are errors. Use the Python API for richer data formats and custom evaluators.

