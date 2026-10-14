# Publishing RetrainOpt

Publishing uses GitHub Actions and PyPI Trusted Publishing (OIDC). A local
build or green pull request does not upload a release. The publishing workflow
runs only on `v*` tags, checks that the tag exactly matches `pyproject.toml`, and
validates that tagged revision before uploading its artifacts.

## Repository and publisher setup

The account owner must configure these external settings before publication:

1. On TestPyPI and PyPI, configure the project `retrainopt` (or a pending
   publisher for a new project) with repository `yangjun42/retrainopt`, workflow
   `publish.yml`, and GitHub environment `testpypi` or `pypi`, respectively.
2. Create the corresponding GitHub environments and configure their protection
   rules. Restrict production publishing to reviewed release tags.
3. Enable GitHub Pages with GitHub Actions as its source. Documentation deploys
   through the `github-pages` environment.
4. Confirm the workflows are enabled and that the publisher names, repository
   ownership and environment names match exactly.

The checked-in workflow expects this setup; it does not create publisher
accounts. It does not require a `PYPI_API_TOKEN` secret. Do not replace the
Trusted Publishing configuration with an unreviewed manual upload workaround.

## Validate before tagging

Use the locked groups and commands in [Development](docs/development.md).
The release checks cover:

- core tests and optional sklearn/Torch/BoTorch integrations;
- exact-vs-exhaustive verification, acquisition regression cases, and online
  policy/causality checks;
- warning-free HTML and doctest builds;
- built wheel and source-distribution validation;
- installation and smoke execution outside the source checkout.

Review [CHANGELOG.md](CHANGELOG.md), migration notes and `CITATION.cff`. Version
metadata must match the intended release. Keep the paper's frozen reproduction
pin unchanged.

## Release candidate

For a candidate such as `0.3.0rc1`, set the package version to `0.3.0rc1` and use
the corresponding tag `v0.3.0rc1`. After the reviewed commit is ready:

```shell
git tag -a v0.3.0rc1 -m 'RetrainOpt 0.3.0rc1'
git push origin v0.3.0rc1
```

The workflow routes `rc` versions to **TestPyPI** using environment `testpypi`.
It reuses the validated distributions, verifies their checksums, uploads them,
then tests an anonymous installation from that index. Only after verification
succeeds does it create the GitHub prerelease with artifacts and checksums.

This core-only manual verification is useful after the workflow completes:

```shell
python -m pip install --no-deps --index-url https://test.pypi.org/simple/ 'retrainopt==0.3.0rc1'
python -c 'import retrainopt; print(retrainopt.__version__)'
```

Use a clean environment and run outside the source checkout. TestPyPI is for
the candidate package; optional integration dependencies should come from
their normal package indexes through the locked validation environment.

## Final release

After candidate verification, update the package and citation version to
`0.3.0`, refresh the lock as needed, and merge the reviewed release changes to
`main`. Create and push `v0.3.0` from that revision. The workflow requires a
stable release tag to be an ancestor of `origin/main`.

Stable versions route to **PyPI** using environment `pypi`. Upload, anonymous
index-install verification, and GitHub release creation follow the same order
as the candidate. The documentation workflow deploys `main` on a push or an
explicit dispatch targeting `main`; release uploads do not independently
publish documentation.

If publisher configuration is missing or a check fails, preserve the failed
run and artifacts and fix the cause. Report publication as incomplete. PyPI
versions are not overwriteable; use a new candidate version for changed
candidate artifacts rather than moving an already published tag.
