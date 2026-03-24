"""Helpers for moving clean Phase 1 runs to and on Puhti."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .manifest import PaperManifest
from .slurm import render_puhti_batch_script


def render_rsync_command(
    *,
    local_project_root: str,
    local_artifact_root: str,
    puhti_target: str,
) -> str:
    return (
        "rsync -avz --exclude='.git' --exclude='__pycache__' "
        f"{local_project_root.rstrip('/')}/ {puhti_target.rstrip('/')}/retrainopt/ && "
        f"rsync -avz {local_artifact_root.rstrip('/')}/ {puhti_target.rstrip('/')}/retrainopt-artifacts/"
    )


def render_puhti_bundle(
    *,
    manifest: PaperManifest,
    artifact_root: str,
    project_root: str,
) -> Dict[str, str]:
    bundle: Dict[str, str] = {}

    for run in manifest.runs:
        filename = f"{run.run_id}.sbatch"
        bundle[filename] = render_puhti_batch_script(
            manifest=manifest,
            run_id=run.run_id,
            artifact_root=artifact_root,
            project_root=project_root,
        )

    submit_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        f"cd {project_root}",
    ]
    for run in manifest.runs:
        submit_lines.append(f"sbatch {run.run_id}.sbatch")
    bundle["submit_all.sh"] = "\n".join(submit_lines) + "\n"
    return bundle


def write_puhti_bundle(
    *,
    manifest: PaperManifest,
    output_dir: Path | str,
    artifact_root: str,
    project_root: str,
) -> Dict[str, Path]:
    bundle = render_puhti_bundle(
        manifest=manifest,
        artifact_root=artifact_root,
        project_root=project_root,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    for filename, content in bundle.items():
        path = root / filename
        path.write_text(content, encoding="utf-8")
        paths[filename] = path
    return paths
