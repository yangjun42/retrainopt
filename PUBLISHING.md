# Publishing RetrainOpt

This document gives a minimal end-to-end workflow for testing, tagging, and
publishing `retrainopt`.

## One-Time Setup

1. Push this repository to GitHub.
2. In the GitHub repository settings, add the secret `PYPI_API_TOKEN`.
3. Confirm that the workflows in `.github/workflows/` are present on the
   default branch.
4. Confirm that the PyPI project name `retrainopt` is still available before
   the first upload.

## Local Validation Before Release

Create a fresh environment:

```shell
python3.13 -m venv .venv
.venv/bin/python -m pip install -e .
```

Run the full test set:

```shell
.venv/bin/python -m unittest tests.test_retrainopt_planner tests.test_retrainopt_smoke tests.test_retrainopt_example
PYTHONPATH=src python3.13 examples/verify_exact_against_bruteforce.py
```

If you want to validate the paper backend locally as well:

```shell
.venv/bin/python -m pip install -e .[botorch]
.venv/bin/python -m unittest tests.test_retrainopt_botorch
```

Build local artifacts:

```shell
.venv/bin/python -m pip install build setuptools wheel
.venv/bin/python -m build --no-isolation
```

If you are working inside a cloud-synced directory and hit timestamp or
`egg-info` update errors during build, build from a clean local clone or a
temporary non-synced directory instead.

Expected outputs:

- `dist/retrainopt-<version>.tar.gz`
- `dist/retrainopt-<version>-py3-none-any.whl`

The GitHub CI workflow also validates:

- source install on Python 3.10-3.13
- wheel install smoke from built artifacts
- the BoTorch backend on Python 3.13 with `.[botorch]`

## Releasing a New Version

1. Update the version in `pyproject.toml`.
2. Update the version in `src/retrainopt/__init__.py`.
3. Commit the changes.
4. Create an annotated tag:

```shell
git tag -a v0.1.0 -m "RetrainOpt 0.1.0"
```

5. Push the branch and the tag:

```shell
git push origin main
git push origin v0.1.0
```

The `publish.yml` workflow will build the package and publish it to PyPI using
`PYPI_API_TOKEN`.

## Manual Fallback

If you want to publish manually instead of GitHub Actions:

```shell
.venv/bin/python -m pip install twine
.venv/bin/python -m twine upload dist/*
```
