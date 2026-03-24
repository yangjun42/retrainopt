"""Example smoke tests for the standalone RetrainOpt repository."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"


class RetrainOptExampleTests(unittest.TestCase):
    def test_minimal_example_runs_without_manual_pythonpath(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "examples" / "minimal_array_timeline.py")],
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Exact", result.stdout)
        self.assertIn("Bayes", result.stdout)

    def test_minimal_example_runs(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "examples" / "minimal_array_timeline.py")],
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(SRC_ROOT),
            },
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Exact", result.stdout)
        self.assertIn("Bayes", result.stdout)

    def test_exact_verification_example_runs_without_manual_pythonpath(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "examples" / "verify_exact_against_bruteforce.py")],
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("VERIFIED", result.stdout)

    def test_exact_verification_example_runs(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "examples" / "verify_exact_against_bruteforce.py")],
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(SRC_ROOT),
            },
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("VERIFIED", result.stdout)


if __name__ == "__main__":
    unittest.main()
