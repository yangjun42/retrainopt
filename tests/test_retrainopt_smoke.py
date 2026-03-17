"""Smoke tests for imports and CLI help."""

from __future__ import annotations

import os
import json
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"


class RetrainOptSmokeTests(unittest.TestCase):
    def test_import_public_package(self):
        sys.path.insert(0, str(SRC_ROOT))
        import retrainopt
        import retrainopt.adapters

        self.assertIn("RetrainingPlanner", retrainopt.__all__)
        self.assertEqual(retrainopt.__version__, "0.1.0")
        self.assertTrue(hasattr(retrainopt.adapters, "BinaryTimelineAdapter"))

        pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text, flags=re.MULTILINE)
        self.assertIsNotNone(version_match)
        self.assertEqual(retrainopt.__version__, version_match.group(1))

    def test_cli_help(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SRC_ROOT)
        result = subprocess.run(
            [sys.executable, "-m", "retrainopt", "--help"],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("RetrainOpt", result.stdout)

    def test_cli_plan_runs_with_dependency_free_prototype_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "train").mkdir()
            (root / "test").mkdir()
            (root / "metadata.json").write_text('{"num_features": 1}')
            self._write_block(root / "train" / "0.bin", [((0.0,), 0), ((1.0,), 1)])
            self._write_block(root / "test" / "0.bin", [((0.1,), 0), ((0.9,), 1)])

            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(SRC_ROOT)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "retrainopt",
                    "plan",
                    "--dataset-dir",
                    str(root),
                    "--model",
                    "prototype",
                    "--method",
                    "exact",
                    "--json",
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["method"], "exact")
            self.assertEqual(len(payload["decisions"]), 1)

    @staticmethod
    def _write_block(path: Path, samples):
        with path.open("wb") as handle:
            for features, label in samples:
                handle.write(struct.pack("<2f", float(features[0]), float(label)))


if __name__ == "__main__":
    unittest.main()
