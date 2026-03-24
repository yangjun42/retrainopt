"""Tests for aggregation, reference comparison, and Puhti/rsync rendering."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

import sys

sys.path.insert(0, str(SRC_ROOT))

from retrainopt.paper_repro.aggregate import aggregate_summaries, write_aggregate_reports
from retrainopt.paper_repro.figures import generate_phase1_svg_figures
from retrainopt.paper_repro.cli import main as paper_repro_main
from retrainopt.paper_repro.manifest import PaperManifest, default_phase1_manifest
from retrainopt.paper_repro.references import latest_paper_exact_targets, latest_paper_report
from retrainopt.paper_repro.runner import RunSummary
from retrainopt.paper_repro.transfer import render_puhti_bundle, render_rsync_command, write_puhti_bundle


class RetrainOptPaperReportingTests(unittest.TestCase):
    def test_latest_paper_targets_include_all_exact_seed42_runs(self):
        targets = latest_paper_exact_targets()

        for run_id in (
            "phase1_exact_extreme_sudden_M_seed42",
            "phase1_exact_gradual_drift_M_seed42",
            "phase1_exact_recurring_drift_M_seed42",
            "phase1_exact_airlines_seed42",
            "phase1_exact_electricity_seed42",
            "phase1_exact_covertype_seed42",
        ):
            self.assertIn(run_id, targets)

    def test_aggregate_reports_write_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir)
            summaries_dir = artifact_root / "summaries"
            summaries_dir.mkdir(parents=True)

            summary = RunSummary(
                run_id="phase1_exact_extreme_sudden_M_seed42",
                study="objective-validity",
                dataset="extreme_sudden_M",
                method="exact",
                model="paper_mlp",
                seed=42,
                timeline_steps=50,
                total_cost=2.8480,
                training_cost=1.0,
                loss_cost=1.8480,
                estimated_total_cost=2.8480,
                evaluated_candidates=10,
                evaluation_training_seconds=10.0,
                oracle_schedule_training_seconds=5.0,
                rog=0.0,
                rmt=2.0,
                acquisition="expected",
                bayes_backend="lightweight",
                cache_fingerprint="abc123",
                manifest_version="phase1-latest-offline-v1",
                decisions=[],
            )
            (summaries_dir / f"{summary.run_id}.json").write_text(
                json.dumps(summary.to_dict(), indent=2),
                encoding="utf-8",
            )

            report = aggregate_summaries(artifact_root)
            self.assertEqual(report["run_count"], 1)

            paths = write_aggregate_reports(artifact_root)
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["csv"].exists())

            with paths["csv"].open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], summary.run_id)

    def test_latest_paper_report_marks_missing_and_drift(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir)
            summaries_dir = artifact_root / "summaries"
            summaries_dir.mkdir(parents=True)

            summary = RunSummary(
                run_id="phase1_exact_extreme_sudden_M_seed42",
                study="objective-validity",
                dataset="extreme_sudden_M",
                method="exact",
                model="paper_mlp",
                seed=42,
                timeline_steps=50,
                total_cost=99.0,
                training_cost=1.0,
                loss_cost=98.0,
                estimated_total_cost=99.0,
                evaluated_candidates=10,
                evaluation_training_seconds=10.0,
                oracle_schedule_training_seconds=5.0,
                rog=0.0,
                rmt=2.0,
                acquisition="expected",
                bayes_backend="lightweight",
                cache_fingerprint="abc123",
                manifest_version="phase1-latest-offline-v1",
                decisions=[],
            )
            (summaries_dir / f"{summary.run_id}.json").write_text(
                json.dumps(summary.to_dict(), indent=2),
                encoding="utf-8",
            )

            report = latest_paper_report(artifact_root)
            self.assertEqual(report["status"], "drift")
            self.assertTrue(report["missing"])
            self.assertTrue(report["mismatches"])

    def test_render_rsync_command_mentions_paths(self):
        command = render_rsync_command(
            local_project_root="/Users/example/retrainopt",
            local_artifact_root="/Users/example/retrainopt-artifacts",
            puhti_target="puhti:/scratch/project_2005865/jun",
        )

        self.assertIn("rsync -avz", command)
        self.assertIn("/Users/example/retrainopt", command)
        self.assertIn("puhti:/scratch/project_2005865/jun", command)

    def test_render_puhti_bundle_creates_submit_helper(self):
        manifest = default_phase1_manifest()
        bundle = render_puhti_bundle(
            manifest=manifest,
            artifact_root="/scratch/project_2005865/jun/retrainopt-artifacts",
            project_root="/scratch/project_2005865/jun/retrainopt",
        )

        self.assertIn("submit_all.sh", bundle)
        self.assertIn("prepare_real_data.sbatch", bundle)
        self.assertIn("postprocess_phase1.sbatch", bundle)
        self.assertIn("phase1_exact_extreme_sudden_M_seed42.sbatch", bundle)
        self.assertIn("sbatch", bundle["submit_all.sh"])
        self.assertIn("--dependency=afterok:", bundle["submit_all.sh"])
        self.assertIn("set -eo pipefail", bundle["prepare_real_data.sbatch"])
        self.assertIn("set -eo pipefail", bundle["postprocess_phase1.sbatch"])
        self.assertIn("PYTHONPATH=src python -m retrainopt.paper_repro.cli prepare-real-data", bundle["prepare_real_data.sbatch"])
        self.assertIn("PYTHONPATH=src python -m retrainopt.paper_repro.cli aggregate", bundle["postprocess_phase1.sbatch"])

    def test_write_puhti_bundle_writes_files(self):
        manifest = default_phase1_manifest()
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = write_puhti_bundle(
                manifest=manifest,
                output_dir=Path(tmp_dir),
                artifact_root="/scratch/project_2005865/jun/retrainopt-artifacts",
                project_root="/scratch/project_2005865/jun/retrainopt",
            )

            self.assertIn("submit_all.sh", paths)
            self.assertTrue(paths["submit_all.sh"].exists())
            self.assertTrue(paths["prepare_real_data.sbatch"].exists())
            self.assertTrue(paths["postprocess_phase1.sbatch"].exists())
            self.assertTrue(paths["phase1_exact_extreme_sudden_M_seed42.sbatch"].exists())

    def test_cli_aggregate_command_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir)
            summaries_dir = artifact_root / "summaries"
            summaries_dir.mkdir(parents=True)
            summary = RunSummary(
                run_id="phase1_exact_extreme_sudden_M_seed42",
                study="objective-validity",
                dataset="extreme_sudden_M",
                method="exact",
                model="paper_mlp",
                seed=42,
                timeline_steps=50,
                total_cost=2.8480,
                training_cost=1.0,
                loss_cost=1.8480,
                estimated_total_cost=2.8480,
                evaluated_candidates=10,
                evaluation_training_seconds=10.0,
                oracle_schedule_training_seconds=5.0,
                rog=0.0,
                rmt=2.0,
                acquisition="expected",
                bayes_backend="lightweight",
                cache_fingerprint="abc123",
                manifest_version="phase1-latest-offline-v1",
                decisions=[],
            )
            (summaries_dir / f"{summary.run_id}.json").write_text(
                json.dumps(summary.to_dict(), indent=2),
                encoding="utf-8",
            )

            exit_code = paper_repro_main(["aggregate", "--artifact-root", str(artifact_root)])
            self.assertEqual(exit_code, 0)
            self.assertTrue((artifact_root / "comparisons" / "aggregate_summary.json").exists())
            self.assertTrue((artifact_root / "comparisons" / "aggregate_summary.csv").exists())

    def test_generate_phase1_svg_figures_writes_svg(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir)
            summaries_dir = artifact_root / "summaries"
            summaries_dir.mkdir(parents=True)
            summary = RunSummary(
                run_id="phase1_exact_extreme_sudden_M_seed42",
                study="objective-validity",
                dataset="extreme_sudden_M",
                method="exact",
                model="paper_mlp",
                seed=42,
                timeline_steps=50,
                total_cost=2.8480,
                training_cost=1.0,
                loss_cost=1.8480,
                estimated_total_cost=2.8480,
                evaluated_candidates=10,
                evaluation_training_seconds=10.0,
                oracle_schedule_training_seconds=5.0,
                rog=0.0,
                rmt=2.0,
                acquisition="expected",
                bayes_backend="lightweight",
                cache_fingerprint="abc123",
                manifest_version="phase1-latest-offline-v1",
                decisions=[],
            )
            (summaries_dir / f"{summary.run_id}.json").write_text(
                json.dumps(summary.to_dict(), indent=2),
                encoding="utf-8",
            )

            outputs = generate_phase1_svg_figures(artifact_root)
            self.assertIn("total_cost_by_run.svg", outputs)
            self.assertTrue(outputs["total_cost_by_run.svg"].exists())
            svg_text = outputs["total_cost_by_run.svg"].read_text(encoding="utf-8")
            self.assertIn("<svg", svg_text)
            self.assertIn(summary.run_id, svg_text)


if __name__ == "__main__":
    unittest.main()
