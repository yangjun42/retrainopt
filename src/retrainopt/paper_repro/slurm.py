"""Puhti SLURM rendering for offline paper repro jobs."""

from __future__ import annotations

from .manifest import PaperManifest


def render_puhti_batch_script(
    *,
    manifest: PaperManifest,
    run_id: str,
    artifact_root: str,
    project_root: str,
) -> str:
    run = manifest.get_run(run_id)
    return f"""#!/bin/bash
#SBATCH --job-name=retrainopt-phase1
#SBATCH --account=project_2005865
#SBATCH --partition=small
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output={artifact_root}/runs/{run.run_id}.out
#SBATCH --error={artifact_root}/runs/{run.run_id}.err

set -euo pipefail

source /appl/profile/zz-csc-env.sh
module load pytorch/2.7

cd {project_root}

PYTHONPATH=src python -m retrainopt.paper_repro.cli run \\
  --run-id {run.run_id} \\
  --artifact-root {artifact_root}
"""


def render_puhti_prepare_data_script(
    *,
    project_root: str,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=retrainopt-prep
#SBATCH --account=project_2005865
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output={project_root}/logs/prepare_real_data.out
#SBATCH --error={project_root}/logs/prepare_real_data.err

set -euo pipefail

source /appl/profile/zz-csc-env.sh
module load pytorch/2.7

cd {project_root}
mkdir -p logs

PYTHONPATH=src python -m retrainopt.paper_repro.cli prepare-real-data \\
  --dataset all \\
  --raw-root {project_root}/data/raw \\
  --output-root {project_root}/data \\
  --timesteps 50 \\
  --seed 42 \\
  --download-missing
"""


def render_puhti_postprocess_script(
    *,
    artifact_root: str,
    project_root: str,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=retrainopt-post
#SBATCH --account=project_2005865
#SBATCH --partition=small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output={artifact_root}/runs/postprocess_phase1.out
#SBATCH --error={artifact_root}/runs/postprocess_phase1.err

set -euo pipefail

source /appl/profile/zz-csc-env.sh
module load pytorch/2.7

cd {project_root}

PYTHONPATH=src python -m retrainopt.paper_repro.cli aggregate --artifact-root {artifact_root}
PYTHONPATH=src python -m retrainopt.paper_repro.cli compare-latest --artifact-root {artifact_root}
PYTHONPATH=src python -m retrainopt.paper_repro.cli generate-figures --artifact-root {artifact_root}
"""
