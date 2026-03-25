"""Tests for the internal offline paper reproduction harness."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

import sys

sys.path.insert(0, str(SRC_ROOT))

from retrainopt.interfaces import ModelAdapter
from retrainopt.paper_repro.legacy import build_initial_future_losses, sample_legacy_subset
from retrainopt.paper_repro.compare import ComparisonTarget, compare_summaries
from retrainopt.paper_repro.manifest import DatasetSpec, PaperManifest, RunSpec, default_phase1_manifest
from retrainopt.paper_repro.runner import OfflinePaperRunner
from retrainopt.paper_repro.slurm import render_puhti_batch_script
from retrainopt.paper_repro.synthetic import build_synthetic_timeline


class CountingPrototypeAdapter(ModelAdapter):
    """Prototype adapter that counts fit calls for cache reuse tests."""

    fit_calls = 0

    def fit(self, train_data, *, step, n_samples, random_seed):
        type(self).fit_calls += 1
        if not train_data:
            return {"prototypes": {}, "majority": 0}

        grouped = {}
        counts = {}
        for features, label in train_data:
            grouped.setdefault(int(label), []).append(float(features[0]))
            counts[int(label)] = counts.get(int(label), 0) + 1

        prototypes = {
            label: sum(values) / len(values)
            for label, values in grouped.items()
        }
        majority = max(counts.items(), key=lambda item: item[1])[0]
        return {"prototypes": prototypes, "majority": majority}

    def evaluate(self, model, eval_data, *, step):
        if not eval_data:
            return 0.0

        mistakes = 0
        prototypes = model["prototypes"]
        majority = model["majority"]

        for features, label in eval_data:
            feature_value = float(features[0])
            if prototypes:
                predicted = min(
                    prototypes.items(),
                    key=lambda item: abs(item[1] - feature_value),
                )[0]
            else:
                predicted = majority
            if int(predicted) != int(label):
                mistakes += 1
        return mistakes / len(eval_data)


class RetrainOptPaperReproTests(unittest.TestCase):
    def test_default_phase1_manifest_covers_expected_scope(self):
        manifest = default_phase1_manifest()

        self.assertEqual(manifest.acceptance_baseline, "latest-paper")
        self.assertIn("extreme_sudden_M", manifest.datasets)
        self.assertIn("gradual_drift_M", manifest.datasets)
        self.assertIn("recurring_drift_M", manifest.datasets)
        self.assertIn("airlines", manifest.datasets)
        self.assertIn("electricity", manifest.datasets)
        self.assertIn("covertype", manifest.datasets)

        studies = {run.study for run in manifest.runs}
        self.assertIn("objective-validity", studies)
        self.assertIn("convergence", studies)
        self.assertIn("scalability", studies)
        self.assertIn("kappa-sensitivity", studies)
        self.assertIn("parameter-sensitivity", studies)
        self.assertEqual(manifest.datasets["extreme_sudden_M"].initial_model_behavior, "DEGRADING_PERFORMANCE")
        self.assertEqual(manifest.datasets["extreme_sudden_M"].m0_degradation_rate, 0.1)
        self.assertEqual(manifest.datasets["gradual_drift_M"].m0_degradation_rate, 0.05)
        self.assertEqual(manifest.datasets["recurring_drift_M"].m0_degradation_rate, 0.06)
        self.assertEqual(manifest.datasets["airlines"].m0_degradation_rate, 0.05)

    def test_build_initial_future_losses_supports_degrading_m0(self):
        losses = build_initial_future_losses(
            num_steps=4,
            behavior="DEGRADING_PERFORMANCE",
            base_loss=0.5,
            degradation_rate=0.2,
            effective_train_time=-1,
        )

        self.assertEqual(losses, {0: 0.7, 1: 0.9, 2: 1.1, 3: 1.3})

    def test_runner_supports_degrading_initial_model_without_step0_training(self):
        manifest = PaperManifest(
            version="test-v1",
            acceptance_baseline="latest-paper",
            datasets={
                "toy_degrading_m0": DatasetSpec(
                    name="toy_degrading_m0",
                    source="synthetic",
                    num_features=1,
                    num_steps=4,
                    samples_per_step=6,
                    drift_pattern="sudden",
                    drift_timestep=2,
                    initial_model_behavior="DEGRADING_PERFORMANCE",
                    m0_base_loss=0.5,
                    m0_degradation_rate=0.2,
                ),
            },
            runs=[
                RunSpec(
                    run_id="degrading_m0_only",
                    study="objective-validity",
                    dataset="toy_degrading_m0",
                    method="exact",
                    model="prototype",
                    seed=3,
                    n_choices=(),
                    min_train_size=999,
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = OfflinePaperRunner(manifest, artifact_root=Path(tmp_dir))
            summary = runner.run("degrading_m0_only")

            self.assertEqual(summary.training_cost, 0.0)
            self.assertEqual(summary.loss_cost, 4.0)
            self.assertEqual(summary.total_cost, 4.0)

    def test_legacy_subset_sampling_matches_torch_random_split(self):
        try:
            import torch
            from torch.utils.data import random_split
        except ImportError:
            self.skipTest("Torch is required for the legacy subset sampling test.")

        block = [((float(index),), index % 2) for index in range(10)]
        generator = torch.Generator().manual_seed(29)
        expected_subset, _ = random_split(block, [4, 6], generator=generator)
        expected = [block[index] for index in expected_subset.indices]

        sampled = sample_legacy_subset(block, n_samples=4, seed=29)

        self.assertEqual(sampled, expected)

    def test_synthetic_run_writes_cache_and_summary(self):
        manifest = PaperManifest(
            version="test-v1",
            acceptance_baseline="latest-paper",
            datasets={
                "toy_sudden": DatasetSpec(
                    name="toy_sudden",
                    source="synthetic",
                    num_features=1,
                    num_steps=4,
                    samples_per_step=12,
                    drift_pattern="sudden",
                    drift_timestep=2,
                    pattern_params={"feature_shift": 2.0, "noise_std": 0.5},
                ),
            },
            runs=[
                RunSpec(
                    run_id="toy_exact",
                    study="objective-validity",
                    dataset="toy_sudden",
                    method="exact",
                    model="prototype",
                    seed=7,
                    n_choices=(0.5, 1.0),
                    expected_artifacts=("summary", "cache"),
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = OfflinePaperRunner(manifest, artifact_root=Path(tmp_dir))
            summary = runner.run("toy_exact")

            self.assertEqual(summary.run_id, "toy_exact")
            self.assertEqual(summary.dataset, "toy_sudden")
            self.assertEqual(summary.method, "exact")

            summary_path = runner.layout.summary_path("toy_exact")
            self.assertTrue(summary_path.exists())

            candidate_files = list(runner.layout.cache_candidates_dir("toy_exact").glob("*.json"))
            self.assertTrue(candidate_files)

    def test_binary_run_uses_binary_timeline_adapter(self):
        manifest = PaperManifest(
            version="test-v1",
            acceptance_baseline="latest-paper",
            datasets={
                "toy_binary": DatasetSpec(
                    name="toy_binary",
                    source="binary",
                    num_features=1,
                    binary_dir="DATASET_DIR",
                ),
            },
            runs=[
                RunSpec(
                    run_id="binary_exact",
                    study="objective-validity",
                    dataset="toy_binary",
                    method="exact",
                    model="prototype",
                    seed=11,
                    n_choices=(1.0,),
                    expected_artifacts=("summary",),
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            (dataset_root / "train").mkdir(parents=True)
            (dataset_root / "test").mkdir(parents=True)
            (dataset_root / "metadata.json").write_text('{"num_features": 1}', encoding="utf-8")
            self._write_block(dataset_root / "train" / "0.bin", [((0.0,), 0), ((1.0,), 1)])
            self._write_block(dataset_root / "train" / "1.bin", [((0.2,), 0), ((0.9,), 1)])
            self._write_block(dataset_root / "test" / "0.bin", [((0.1,), 0), ((0.8,), 1)])
            self._write_block(dataset_root / "test" / "1.bin", [((0.3,), 0), ((0.7,), 1)])

            manifest.datasets["toy_binary"] = DatasetSpec(
                name="toy_binary",
                source="binary",
                num_features=1,
                binary_dir=str(dataset_root),
            )

            runner = OfflinePaperRunner(manifest, artifact_root=Path(tmp_dir) / "artifacts")
            summary = runner.run("binary_exact")

            self.assertEqual(summary.dataset, "toy_binary")
            self.assertEqual(summary.timeline_steps, 2)
            self.assertEqual(summary.method, "exact")

    def test_synthetic_timeline_supports_sliding_window_state(self):
        manifest = PaperManifest(
            version="test-v1",
            acceptance_baseline="latest-paper",
            datasets={
                "toy_windowed": DatasetSpec(
                    name="toy_windowed",
                    source="synthetic",
                    num_features=1,
                    num_steps=4,
                    samples_per_step=5,
                    test_split_ratio=0.2,
                    drift_pattern="sudden",
                    drift_timestep=2,
                    pattern_params={"feature_shift": 2.0, "noise_std": 0.5},
                    data_state_strategy="sliding_window",
                    sliding_window_k=2,
                ),
            },
            runs=[],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = OfflinePaperRunner(manifest, artifact_root=Path(tmp_dir))
            timeline = runner._build_timeline(manifest.datasets["toy_windowed"], seed=7)

            self.assertEqual([timeline.train_size(step) for step in range(timeline.num_steps)], [4, 8, 8, 8])
            self.assertEqual([len(timeline.get_eval_block(step)) for step in range(timeline.num_steps)], [1, 2, 2, 2])

    def test_binary_timeline_supports_sliding_window_state(self):
        manifest = PaperManifest(
            version="test-v1",
            acceptance_baseline="latest-paper",
            datasets={
                "toy_binary_windowed": DatasetSpec(
                    name="toy_binary_windowed",
                    source="binary",
                    num_features=1,
                    binary_dir="DATASET_DIR",
                    data_state_strategy="sliding_window",
                    sliding_window_k=2,
                ),
            },
            runs=[],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "dataset"
            (dataset_root / "train").mkdir(parents=True)
            (dataset_root / "test").mkdir(parents=True)
            (dataset_root / "metadata.json").write_text('{"num_features": 1}', encoding="utf-8")
            self._write_block(dataset_root / "train" / "0.bin", [((0.0,), 0), ((1.0,), 1)])
            self._write_block(dataset_root / "train" / "1.bin", [((0.2,), 0), ((0.9,), 1)])
            self._write_block(dataset_root / "train" / "2.bin", [((0.3,), 0), ((0.7,), 1)])
            self._write_block(dataset_root / "test" / "0.bin", [((0.1,), 0)])
            self._write_block(dataset_root / "test" / "1.bin", [((0.8,), 1)])
            self._write_block(dataset_root / "test" / "2.bin", [((0.6,), 1)])

            manifest.datasets["toy_binary_windowed"] = DatasetSpec(
                name="toy_binary_windowed",
                source="binary",
                num_features=1,
                binary_dir=str(dataset_root),
                data_state_strategy="sliding_window",
                sliding_window_k=2,
            )

            runner = OfflinePaperRunner(manifest, artifact_root=Path(tmp_dir) / "artifacts")
            timeline = runner._build_timeline(manifest.datasets["toy_binary_windowed"], seed=11)

            self.assertEqual([timeline.train_size(step) for step in range(timeline.num_steps)], [2, 4, 4])
            self.assertEqual([len(timeline.get_eval_block(step)) for step in range(timeline.num_steps)], [1, 2, 2])

    def test_cache_reuse_avoids_duplicate_training(self):
        manifest = PaperManifest(
            version="test-v1",
            acceptance_baseline="latest-paper",
            datasets={
                "toy_gradual": DatasetSpec(
                    name="toy_gradual",
                    source="synthetic",
                    num_features=1,
                    num_steps=3,
                    samples_per_step=8,
                    drift_pattern="gradual",
                    drift_timestep=1,
                    pattern_params={"drift_duration": 1, "final_threshold": 1.0},
                ),
            },
            runs=[
                RunSpec(
                    run_id="cache_exact",
                    study="objective-validity",
                    dataset="toy_gradual",
                    method="exact",
                    model="prototype",
                    seed=5,
                    n_choices=(0.5, 1.0),
                ),
            ],
        )

        CountingPrototypeAdapter.fit_calls = 0
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = OfflinePaperRunner(
                manifest,
                artifact_root=Path(tmp_dir),
                model_overrides={"prototype": CountingPrototypeAdapter()},
            )
            runner.run("cache_exact")
            first_calls = CountingPrototypeAdapter.fit_calls
            runner.run("cache_exact")
            second_calls = CountingPrototypeAdapter.fit_calls

            self.assertGreater(first_calls, 0)
            self.assertEqual(first_calls, second_calls)

    def test_runner_recovers_from_corrupt_candidate_cache(self):
        manifest = PaperManifest(
            version="test-v1",
            acceptance_baseline="latest-paper",
            datasets={
                "toy_gradual": DatasetSpec(
                    name="toy_gradual",
                    source="synthetic",
                    num_features=1,
                    num_steps=3,
                    samples_per_step=8,
                    drift_pattern="gradual",
                    drift_timestep=1,
                    pattern_params={"drift_duration": 1, "final_threshold": 1.0},
                ),
            },
            runs=[
                RunSpec(
                    run_id="corrupt_cache_exact",
                    study="objective-validity",
                    dataset="toy_gradual",
                    method="exact",
                    model="prototype",
                    seed=5,
                    n_choices=(1.0,),
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = OfflinePaperRunner(manifest, artifact_root=Path(tmp_dir))
            timeline = build_synthetic_timeline(manifest.datasets["toy_gradual"], seed=5)
            cache_path = runner.layout.candidate_path("corrupt_cache_exact", 0, timeline.train_size(0))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text('{"broken": true}\n{"still": "broken"}\n', encoding="utf-8")

            summary = runner.run("corrupt_cache_exact")

            self.assertEqual(summary.run_id, "corrupt_cache_exact")
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertIn("model_id", payload)
            self.assertIn("training_cost", payload)

    def test_synthetic_paper_mlp_run_supports_list_backed_timelines(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("Torch is required for the paper_mlp integration test.")

        manifest = PaperManifest(
            version="test-v1",
            acceptance_baseline="latest-paper",
            datasets={
                "toy_sudden_mlp": DatasetSpec(
                    name="toy_sudden_mlp",
                    source="synthetic",
                    num_features=2,
                    num_steps=3,
                    samples_per_step=12,
                    drift_pattern="sudden",
                    drift_timestep=1,
                    pattern_params={"feature_shift": 1.5, "noise_std": 0.2},
                    model_defaults={"hidden_dims": [8], "epochs": 1, "learning_rate": 0.01, "batch_size": 4},
                ),
            },
            runs=[
                RunSpec(
                    run_id="paper_mlp_exact",
                    study="objective-validity",
                    dataset="toy_sudden_mlp",
                    method="exact",
                    model="paper_mlp",
                    seed=3,
                    n_choices=(1.0,),
                    expected_artifacts=("summary",),
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = OfflinePaperRunner(manifest, artifact_root=Path(tmp_dir))
            summary = runner.run("paper_mlp_exact")

            self.assertEqual(summary.run_id, "paper_mlp_exact")
            self.assertEqual(summary.method, "exact")
            self.assertTrue((Path(tmp_dir) / "summaries" / "paper_mlp_exact.json").exists())

    def test_comparison_report_flags_drift(self):
        manifest = PaperManifest(
            version="test-v1",
            acceptance_baseline="latest-paper",
            datasets={
                "toy_sudden_compare": DatasetSpec(
                    name="toy_sudden_compare",
                    source="synthetic",
                    num_features=1,
                    num_steps=3,
                    samples_per_step=10,
                    drift_pattern="sudden",
                    drift_timestep=1,
                    pattern_params={"feature_shift": 2.0, "noise_std": 0.5},
                ),
            },
            runs=[
                RunSpec(
                    run_id="compare_exact",
                    study="objective-validity",
                    dataset="toy_sudden_compare",
                    method="exact",
                    model="prototype",
                    seed=13,
                    n_choices=(0.5, 1.0),
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = OfflinePaperRunner(manifest, artifact_root=Path(tmp_dir))
            summary = runner.run("compare_exact")

            report = compare_summaries(
                [summary],
                [
                    ComparisonTarget(
                        run_id=summary.run_id,
                        total_cost=summary.total_cost + 0.5,
                        tolerance=0.01,
                    )
                ],
            )

            self.assertEqual(report["status"], "drift")
            self.assertEqual(len(report["mismatches"]), 1)

    def test_render_puhti_batch_script_contains_expected_paths(self):
        manifest = default_phase1_manifest()
        script = render_puhti_batch_script(
            manifest=manifest,
            run_id="phase1_exact_extreme_sudden_M_seed42",
            artifact_root="/scratch/project_2005865/jun/retrainopt-artifacts",
            project_root="/scratch/project_2005865/jun/retrainopt",
        )

        self.assertIn("#SBATCH --job-name=retrainopt-phase1", script)
        self.assertIn("set -eo pipefail", script)
        self.assertNotIn("set -euo pipefail", script)
        self.assertIn("module load pytorch/2.7", script)
        self.assertIn("PYTHONPATH=src python -m retrainopt.paper_repro.cli run", script)
        self.assertIn("/scratch/project_2005865/jun/retrainopt-artifacts", script)

    @staticmethod
    def _write_block(path: Path, samples):
        with path.open("wb") as handle:
            for features, label in samples:
                handle.write(struct.pack("<2f", float(features[0]), float(label)))


if __name__ == "__main__":
    unittest.main()
