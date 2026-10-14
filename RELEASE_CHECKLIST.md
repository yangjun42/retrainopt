# Release checklist

## Source and behavior

- [ ] Release changes reviewed; working tree and tag identify the intended revision.
- [ ] Package version, citation version and tag agree.
- [ ] Changelog and migration notes describe the release's public API and behavior.
- [ ] Exact DP passes independent exhaustive checks on small fixtures.
- [ ] Whole-stage acquisition, weighted variance, observed entries and zero-variance EI/PI pass regression checks.
- [ ] Evaluator, configuration, data and cost contracts pass tests.
- [ ] Online features, policy training, model transitions and current-step cost accounting pass tests.
- [ ] Historical policy training and online execution cannot read future data.
- [ ] Paper reproduction tag/pin and frozen artifacts remain unchanged.

## Distribution and documentation

- [ ] Locked test/build/docs groups resolve and validate.
- [ ] Minimal, online and optional sklearn/Torch/BoTorch examples pass.
- [ ] Sphinx HTML and doctests build without warnings.
- [ ] Wheel and sdist build, metadata checks pass, and checksums are recorded.
- [ ] Wheel and sdist-derived installations run outside the source checkout.
- [ ] Public API, assumptions, cost accounting and limitations are documented.

## External configuration

- [ ] TestPyPI Trusted Publisher matches `publish.yml` and `testpypi`.
- [ ] PyPI Trusted Publisher matches `publish.yml` and `pypi`.
- [ ] GitHub environments and protection rules are configured.
- [ ] GitHub Pages uses Actions and its deployment environment is ready.

## Publication

- [ ] Candidate tag exactly matches the `rc` package version.
- [ ] Tagged validation and TestPyPI upload succeed.
- [ ] Anonymous TestPyPI install succeeds; GitHub prerelease contains verified artifacts/checksums.
- [ ] Final version is reviewed on `main`; stable tag matches package version.
- [ ] PyPI upload and anonymous PyPI installation succeed.
- [ ] GitHub release contains verified artifacts/checksums and links to migration notes.
- [ ] Documentation deployment succeeds and shows the intended release.

Unchecked publisher/index/deployment steps mean publication is pending, even
when local validation is complete. See [PUBLISHING.md](PUBLISHING.md).
