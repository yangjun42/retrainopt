"""Regression checks that keep the public package separate from paper repro."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src" / "retrainopt"


class RetrainOptPackageBoundaryTests(unittest.TestCase):
    def test_source_tree_excludes_paper_repro_package(self):
        self.assertFalse((SRC_ROOT / "paper_repro").exists())

    def test_readme_points_paper_reproduction_to_companion_repo(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("retrainopt-paper-repro", readme)
        self.assertNotIn("python -m retrainopt.paper_repro.cli", readme)

    def test_public_algorithm_mapping_doc_exists(self):
        doc_path = PROJECT_ROOT / "docs" / "paper_algorithm_mapping.md"
        self.assertTrue(doc_path.exists())

        text = doc_path.read_text(encoding="utf-8")
        self.assertIn('RetrainingPlanner.plan(method="exact")', text)
        self.assertIn('RetrainingPlanner.plan(method="bayes")', text)


if __name__ == "__main__":
    unittest.main()
