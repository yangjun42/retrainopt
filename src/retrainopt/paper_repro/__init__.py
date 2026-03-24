"""Internal offline paper reproduction helpers for RetrainOpt."""

from .aggregate import aggregate_summaries, write_aggregate_reports
from .compare import ComparisonTarget, compare_summaries
from .manifest import DatasetSpec, PaperManifest, RunSpec, default_phase1_manifest
from .references import latest_paper_exact_targets, latest_paper_report
from .runner import OfflinePaperRunner, RunSummary
from .slurm import render_puhti_batch_script
from .transfer import render_puhti_bundle, render_rsync_command

__all__ = [
    "ComparisonTarget",
    "DatasetSpec",
    "OfflinePaperRunner",
    "PaperManifest",
    "RunSpec",
    "RunSummary",
    "aggregate_summaries",
    "compare_summaries",
    "default_phase1_manifest",
    "latest_paper_exact_targets",
    "latest_paper_report",
    "render_puhti_batch_script",
    "render_puhti_bundle",
    "render_rsync_command",
    "write_aggregate_reports",
]
