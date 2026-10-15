# Development and validation

`pyproject.toml` declares package metadata, dependencies and the setuptools
build backend used by standard tools such as pip, uv and `python -m build`;
`uv.lock` records the reproducible development/CI environment.
Runtime extras (`sklearn`, `torch`, `botorch`) are separate from development
groups (`test`, `build`, `docs`).
CI pins uv 0.13.1. The Linux lock uses CPU PyTorch wheels; published extras do
not force a particular accelerator wheel index.

From a local checkout:

```shell
uv sync --locked --all-extras --group test --group build --group docs
uv run --no-sync python -m pytest
uv run --no-sync python examples/minimal_array_timeline.py
uv run --no-sync python examples/verify_exact_against_bruteforce.py
uv run --no-sync python examples/online_timeline.py
uv run --no-sync python examples/learned_online_timeline.py
```

Optional integration examples are listed in [Integrations](integrations.md).
Changing a dependency declaration requires refreshing and checking `uv.lock`.
Do not infer dependencies from an existing virtual environment.

## Documentation and distributions

Build the documentation and release distributions in a temporary directory:

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
tag. Small deterministic fixtures test DP optimality, acquisition semantics,
online decision timing and the boundary between historical and deployment data;
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

## Publishing

Publishing uses GitHub Actions and PyPI Trusted Publishing (OIDC). Before the
first release, the account owner must configure:

- TestPyPI and PyPI project `retrainopt` (or a pending publisher), repository
  `yangjun42/retrainopt`, workflow `publish.yml`, and GitHub environments
  `testpypi` and `pypi`, respectively. Names must match exactly.
- Those GitHub environments and their protection rules, restricting production
  publishing to reviewed release tags; the repository workflows must be enabled.
- GitHub Pages with GitHub Actions as its source and the `github-pages`
  environment for documentation deployment.

The workflow does not create these accounts or require a `PYPI_API_TOKEN`.
Keep Trusted Publishing as the upload route.

1. Prepare an `X.Y.ZrcN` candidate in `pyproject.toml`, align `CITATION.cff`,
   refresh `uv.lock` as needed, and review the changelog and migration notes.
   Complete the locked validation and distribution checks above. Keep the
   paper reproduction pin and frozen artifacts unchanged. Commit the
   reviewed changes, then tag that revision with its exact package version:

   ```shell
   VERSION=$(uv version --short)
   git tag -a "v$VERSION" -m "RetrainOpt $VERSION"
   git push origin "v$VERSION"
   ```

2. Follow the `Publish` workflow. It accepts only `vX.Y.ZrcN` or `vX.Y.Z` tags
   matching `pyproject.toml`, and reruns the full reusable validation workflow
   on that tagged revision. Candidates use TestPyPI/environment `testpypi`;
   stable versions use PyPI/environment `pypi`.
3. Confirm upload and verification succeeded. The workflow verifies
   `SHA256SUMS`, uploads the validated wheel and source distribution, then
   installs the exact version anonymously with `--no-cache --no-deps` in a
   clean environment. `scripts/verify_index.py` compares index digests with
   those artifacts; `scripts/installed_smoke.py` runs outside the checkout.
   Only then does it create the GitHub release or prerelease with both
   distributions and checksums. Retain these with the workflow result.
4. After candidate verification, update package/citation metadata to `X.Y.Z`,
   refresh the lock as needed, and merge the reviewed final changes into
   `main`. Repeat the tag commands from that revision. Stable tags must point
   to a commit reachable from `origin/main`; the final artifacts undergo the
   same validation and public-index installation checks.

For an additional manual core-only installation check, set `VERSION` to the
published version and run in a clean environment outside the source checkout:

```shell
RETRAINOPT_VERIFY_DIR=$(mktemp -d)
uv venv "$RETRAINOPT_VERIFY_DIR/venv"
uv pip install --no-cache --no-deps --python "$RETRAINOPT_VERIFY_DIR/venv/bin/python" \
  --index-url https://test.pypi.org/simple/ "retrainopt==$VERSION"
(cd "$RETRAINOPT_VERIFY_DIR" && ./venv/bin/python -I -c 'import retrainopt; print(retrainopt.__version__)')
```

Use `https://pypi.org/simple/` for a stable release. Optional integration
dependencies belong in the normal locked validation environment, not TestPyPI.
Documentation deploys separately on a `main` push or a manual documentation
workflow dispatch targeting `main`; uploading a release does not deploy it.

If configuration or any check fails, preserve the run and artifacts, fix the
cause, and report publication as incomplete. A local build or green pull
request alone is not a release. Index versions cannot be overwritten: changed
artifacts require a new version, never moving an already published tag.
