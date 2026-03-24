"""Allow examples to run from a source checkout without manual PYTHONPATH."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
