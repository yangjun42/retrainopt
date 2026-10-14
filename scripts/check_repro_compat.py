"""Smoke-test a separate paper-repro checkout against the installed library.

This is a tiny compatibility check, not a paper-result reproduction. It reads
the supplied checkout without modifying its dependency pin, and creates a new
artifact namespace on every invocation. Install the repro project's declared
base dependencies separately, excluding its historical retrainopt pin.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    repro_root = args.repro_root.resolve()
    if not (repro_root / "src/retrainopt_paper_repro/runner.py").is_file():
        parser.error("--repro-root must contain src/retrainopt_paper_repro/runner.py")
    repro_commit = subprocess.check_output(
        ["git", "-C", str(repro_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repro_root / "src"))

    import retrainopt
    from retrainopt import EvaluationProvider, RetrainingPlanner
    from retrainopt.adapters import PrototypeModelAdapter
    from retrainopt_paper_repro.cache import ArtifactLayout
    from retrainopt_paper_repro.manifest import DatasetSpec, PaperManifest, RunSpec
    from retrainopt_paper_repro.runner import OfflinePaperRunner, PaperEvaluationEngine
    from retrainopt_paper_repro.synthetic import build_synthetic_timeline

    namespace = args.artifact_root.resolve() / f"retrainopt-{version('retrainopt')}-{uuid4().hex}"
    namespace.mkdir(parents=True, exist_ok=False)
    dataset = DatasetSpec(
        name="compat_synthetic", source="synthetic", num_features=1,
        num_steps=3, samples_per_step=32, test_split_ratio=0.25,
        drift_pattern="sudden", drift_timestep=2,
    )
    runs = [
        RunSpec(
            run_id=f"compat_{method}", study="compatibility", dataset=dataset.name,
            method=method, model="prototype", seed=42, n_choices=(0.5, 1.0),
            time_cost_weight=0.0, bayes_backend="lightweight",
            bayes_iterations=1, bayes_initial_samples=2,
            bayes_candidates_per_iteration=1, bayes_anchor_steps=(),
        )
        for method in ("exact", "bayes")
    ]
    manifest = PaperManifest(
        version="compatibility-0.2", acceptance_baseline="compatibility-only",
        datasets={dataset.name: dataset}, runs=runs,
    )
    runner = OfflinePaperRunner(manifest, artifact_root=namespace / "runner")
    runner_summaries = {}
    for run in runs:
        summary = runner.run(run.run_id)
        if not math.isfinite(summary.total_cost) or len(summary.decisions) != dataset.num_steps:
            raise AssertionError(f"Invalid OfflinePaperRunner output for {run.method}")
        runner_summaries[run.method] = summary

    injected_results = {}
    for run in runs:
        timeline = build_synthetic_timeline(dataset, seed=run.seed)
        adapter = PrototypeModelAdapter()
        config = run.planner_config()
        provider = PaperEvaluationEngine(
            dataset_spec=dataset, timeline=timeline, model_adapter=adapter,
            config=config, layout=ArtifactLayout(namespace / "injected"),
            cache_key=run.run_id,
        )
        if not isinstance(provider, EvaluationProvider):
            raise AssertionError("The external engine does not implement EvaluationProvider")
        result = RetrainingPlanner(timeline, adapter, config, evaluator=provider).plan(run.method)
        if not math.isfinite(result.total_cost) or len(result.decisions) != dataset.num_steps:
            raise AssertionError(f"Invalid injected-provider output for {run.method}")
        if not list(provider.iter_cached_candidates()):
            raise AssertionError("The external engine did not evaluate any candidates")
        if not math.isclose(result.total_cost, runner_summaries[run.method].total_cost, abs_tol=1e-12):
            raise AssertionError(f"Runner/injection objective mismatch for {run.method}")
        injected_results[run.method] = result

    print(json.dumps({
        "status": "passed",
        "check": "tiny synthetic API compatibility; not published-result reproduction",
        "retrainopt_version": version("retrainopt"),
        "retrainopt_import": str(Path(retrainopt.__file__).resolve()),
        "repro_commit": repro_commit,
        "artifact_namespace": str(namespace),
        "methods": {
            method: {
                "runner_total_cost": runner_summaries[method].total_cost,
                "injected_total_cost": result.total_cost,
                "result_schema_version": result.to_dict()["schema_version"],
                "initial_model_source": result.metadata["initial_model_source"],
            }
            for method, result in injected_results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
