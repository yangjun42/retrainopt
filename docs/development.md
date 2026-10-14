# Development and validation

Dependency declarations live in `pyproject.toml`; `uv.lock` records the
reproducible development/CI environment. Runtime extras (`sklearn`, `torch`,
`botorch`) are separate from development groups (`test`, `build`, `docs`).
CI pins uv 0.13.1. The Linux lock uses CPU PyTorch wheels; published extras do
not force a particular accelerator wheel index.

From a local checkout:

```shell
uv sync --locked --all-extras --group test --group build --group docs
uv run --no-sync python -m pytest
uv run --no-sync python examples/minimal_array_timeline.py
uv run --no-sync python examples/verify_exact_against_bruteforce.py
```

Optional integration examples are listed in [Integrations](integrations.md).
Changing a dependency declaration requires refreshing and checking `uv.lock`.
Do not infer dependencies from an existing virtual environment.

## Documentation and distributions

Build generated files outside the checkout when it lives in a synchronized
folder:

```shell
RETRAINOPT_BUILD_DIR=$(mktemp -d)
uv run --no-sync sphinx-build -W --keep-going -b html docs "$RETRAINOPT_BUILD_DIR/html"
uv run --no-sync sphinx-build -W --keep-going -b doctest docs "$RETRAINOPT_BUILD_DIR/doctest"
SOURCE_DATE_EPOCH=$(git log -1 --format=%ct) uv run --locked --group build python scripts/build_release.py --outdir "$RETRAINOPT_BUILD_DIR/dist"
uv run --no-sync python -m twine check "$RETRAINOPT_BUILD_DIR"/dist/*
```

The build output directory must be empty or absent. The release script uses
the locked build backend, builds the wheel from the source distribution, and
normalizes source archive timestamps, owner IDs and permissions. CI builds
twice with the same commit timestamp and requires identical wheel and source
archive bytes before uploading the distributions.

The documentation imports the installed package, not a manually inserted source
path. Validate both the wheel and a wheel rebuilt from the source distribution
in clean environments, from a directory outside the checkout. Importing from
`src/` alone cannot establish that the published package is complete.

The reusable `validation.yml` workflow runs source tests, optional integration
checks, documentation/doctests and installed-distribution checks. CI invokes
it for branch pushes and pull requests; publishing invokes it for the release
tag. Small deterministic fixtures test DP optimality and acquisition semantics;
full paper experiments are a separate reproducibility workflow.

## Minimum direct dependencies

The `minimum-direct-dependencies` CI job uses Linux and Python 3.10 to exercise
the lowest compatible versions of the declared runtime extras. It resolves
with `uv sync --resolution lowest-direct --upgrade --all-extras --group test`,
requires and reports all optional backends, and runs the full test suite and
the sklearn, Torch and BoTorch examples. Transitive dependencies remain at the
newest compatible versions; this checks direct lower bounds, not every possible
dependency combination. The sklearn floor is 1.4.2 because that release adds
[NumPy 2 compatibility](https://scikit-learn.org/1.4/whats_new/v1.4.html#version-1-4-2).

This job deliberately resolves a temporary lockfile instead of using
`--locked`; `--upgrade` removes preferences for versions in the committed
lockfile. Its checkout and environment are disposable, and it does not build
or upload release artifacts. The regular matrix and release builds continue
to use the committed `uv.lock`. Reproduce this lower-bound check only in a
separate local checkout so it does not replace the normal development lock.

## External evaluator compatibility

`scripts/check_repro_compat.py` exercises the real `OfflinePaperRunner` and
`PaperEvaluationEngine` from a separate paper-repro checkout. It runs a tiny
synthetic exact/approximate case, then uses the public `evaluator=` interface
and compares the measured objectives. Each invocation creates a unique
artifact directory; it does not use historical caches or edit the reproduction
project's dependency pin.

Run it in a separate environment with the candidate wheel and the reproduction
project's declared base dependencies, excluding its pinned `retrainopt`
dependency. Do not run its ordinary dependency synchronization against this
compatibility environment, which would restore the historical package.

```shell
python scripts/check_repro_compat.py \
  --repro-root /path/to/isolated/retrainopt-paper-repro \
  --artifact-root /path/to/local/compatibility-artifacts
```

The JSON report records the installed package path/version, reproduction
commit, artifact namespace and both objectives. This checks API compatibility;
it does not establish numerical reproduction of published experiments.

## Runtime and artifacts

Keep virtual environments, package caches, builds and scratch output outside
synchronized source trees, using a machine-local runtime/cache or a local
checkout. Follow workspace runtime configuration when present. The repository
contains source, dependency declarations, tests, documentation and small
examples; it does not require downloading benchmark datasets or generating
large model artifacts for its release checks.

Release artifacts are the wheel, source distribution and their checksums, built
from the release tag. Retain those together with the workflow result. Generated
HTML is published separately by the documentation workflow. Publication is
complete only when index installation and the corresponding release checks
succeed; a local build alone is not a published release.

## Paper provenance and citation

The canonical experiment project remains pinned to
`ecml-pkdd-2026-camera-ready-retrainopt` (its documented short commit is
`1fc9f59`). Version 0.2 does not rewrite that tag, upgrade the reproduction
project, or replace paper artifacts.

`CITATION.cff` records the software and the paper *Automated Retraining for
Cost-Optimal ML*, Jun Yang and Michael Mathioudakis, 2026. It intentionally
omits unverified DOI, page and volume metadata.
