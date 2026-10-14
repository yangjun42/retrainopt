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

