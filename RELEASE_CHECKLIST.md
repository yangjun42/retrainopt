# Release Checklist

## Before First Publish

- Fill in the copyright owner/year in `LICENSE` if you want the appendix boilerplate customized
- Confirm the PyPI name `retrainopt` one more time right before first upload
- Decide whether to keep version `0.1.0` or start from another initial version
- Decide whether you want `pip install retrainopt[botorch]` documented as experimental or fully supported
- Add the `PYPI_API_TOKEN` GitHub Actions secret if you want to use the bundled publish workflow

## Recommended Repo Setup

- Initialize a fresh git repository in this directory
- Create a clean virtual environment
- Run `pip install -e .`
- Run `python -m unittest tests.test_retrainopt_planner tests.test_retrainopt_smoke`

## Optional But Useful

- Tune the bundled GitHub Actions workflows for your repository conventions
- Add project URLs and author metadata to `pyproject.toml`
- Add a changelog
- Add a couple of benchmark or real-dataset examples
