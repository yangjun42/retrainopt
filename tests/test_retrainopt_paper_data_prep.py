"""Tests for clean real-dataset preparation in the Phase 1 repro harness."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

import sys

sys.path.insert(0, str(SRC_ROOT))

from retrainopt.adapters import BinaryTimelineAdapter
from retrainopt.paper_repro.cli import main as paper_repro_main
from retrainopt.paper_repro.data_prep import prepare_real_dataset


class RetrainOptPaperDataPrepTests(unittest.TestCase):
    def test_prepare_real_dataset_airlines_writes_binary_timeline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_root = Path(tmp_dir) / "raw"
            output_root = Path(tmp_dir) / "prepared"
            airlines_path = raw_root / "airlines" / "raw.csv"
            airlines_path.parent.mkdir(parents=True)
            self._write_csv(
                airlines_path,
                header=[
                    "Month",
                    "DayofMonth",
                    "DayOfWeek",
                    "CRSDepTime",
                    "CRSArrTime",
                    "Distance",
                    "AirTime",
                    "Delay",
                ],
                rows=[
                    [1, 1, 1, 900, 1030, 500, 90, 0],
                    [1, 2, 2, 930, 1100, 600, 95, 1],
                    [1, 3, 3, 1000, 1130, 650, 97, 0],
                    [1, 4, 4, 1030, 1200, 700, 100, 1],
                    [1, 5, 5, 1100, 1230, 750, 104, 0],
                    [1, 6, 6, 1130, 1300, 800, 108, 1],
                ],
            )

            result = prepare_real_dataset(
                "airlines",
                raw_root=raw_root,
                output_root=output_root,
                num_steps=3,
                seed=7,
            )

            self.assertEqual(result["dataset"], "airlines")
            self.assertEqual(result["num_features"], 7)
            dataset_dir = output_root / "airlines"
            self.assertTrue((dataset_dir / "metadata.json").exists())
            self.assertEqual(len(list((dataset_dir / "train").glob("*.bin"))), 3)
            self.assertEqual(len(list((dataset_dir / "test").glob("*.bin"))), 3)

            timeline = BinaryTimelineAdapter(dataset_dir)
            self.assertEqual(timeline.num_steps, 3)

    def test_prepare_real_dataset_covertype_parses_libsvm(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_root = Path(tmp_dir) / "raw"
            output_root = Path(tmp_dir) / "prepared"
            covtype_path = raw_root / "covertype" / "raw" / "covtype.libsvm.binary.scale"
            covtype_path.parent.mkdir(parents=True)
            covtype_path.write_text(
                "\n".join(
                    [
                        "1 1:0.1 3:0.4 54:0.9",
                        "2 1:0.2 3:0.5 54:0.8",
                        "1 1:0.3 3:0.6 54:0.7",
                        "2 1:0.4 3:0.7 54:0.6",
                        "1 1:0.5 3:0.8 54:0.5",
                        "2 1:0.6 3:0.9 54:0.4",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = prepare_real_dataset(
                "covertype",
                raw_root=raw_root,
                output_root=output_root,
                num_steps=3,
                seed=11,
            )

            self.assertEqual(result["dataset"], "covertype")
            self.assertEqual(result["num_features"], 54)
            timeline = BinaryTimelineAdapter(output_root / "covertype")
            self.assertEqual(timeline.num_steps, 3)

    def test_cli_prepare_real_data_command_writes_expected_tree(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_root = Path(tmp_dir) / "raw"
            output_root = Path(tmp_dir) / "prepared"
            electricity_path = raw_root / "elec" / "raw.csv"
            electricity_path.parent.mkdir(parents=True)
            self._write_csv(
                electricity_path,
                header=["day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer", "class"],
                rows=[
                    [1, 1, 0.1, 0.2, 0.3, 0.4, 0.5, "UP"],
                    [1, 2, 0.2, 0.3, 0.4, 0.5, 0.6, "DOWN"],
                    [1, 3, 0.3, 0.4, 0.5, 0.6, 0.7, "UP"],
                    [1, 4, 0.4, 0.5, 0.6, 0.7, 0.8, "DOWN"],
                ],
            )

            exit_code = paper_repro_main(
                [
                    "prepare-real-data",
                    "--dataset",
                    "electricity",
                    "--raw-root",
                    str(raw_root),
                    "--output-root",
                    str(output_root),
                    "--timesteps",
                    "2",
                    "--seed",
                    "5",
                ]
            )

            self.assertEqual(exit_code, 0)
            timeline = BinaryTimelineAdapter(output_root / "electricity")
            self.assertEqual(timeline.num_steps, 2)

    @staticmethod
    def _write_csv(path: Path, header, rows) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
