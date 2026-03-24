"""Internal CLI for Phase 1 offline paper reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import write_aggregate_reports
from .figures import generate_phase1_svg_figures
from .manifest import default_phase1_manifest
from .references import latest_paper_report
from .runner import OfflinePaperRunner
from .slurm import render_puhti_batch_script
from .transfer import render_puhti_bundle, render_rsync_command, write_puhti_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m retrainopt.paper_repro.cli",
        description="Internal offline paper reproduction utilities.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Execute a single run from the Phase 1 manifest.")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--artifact-root", required=True)

    list_parser = subparsers.add_parser("list", help="List available Phase 1 run IDs.")
    list_parser.add_argument("--json", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate existing run summaries.")
    aggregate_parser.add_argument("--artifact-root", required=True)

    compare_parser = subparsers.add_parser("compare-latest", help="Compare summaries against latest-paper exact targets.")
    compare_parser.add_argument("--artifact-root", required=True)

    figures_parser = subparsers.add_parser("generate-figures", help="Generate dependency-light SVG figures from summaries.")
    figures_parser.add_argument("--artifact-root", required=True)

    render_parser = subparsers.add_parser("render-puhti", help="Render a Puhti batch script for one run.")
    render_parser.add_argument("--run-id", required=True)
    render_parser.add_argument("--artifact-root", required=True)
    render_parser.add_argument("--project-root", required=True)

    render_all_parser = subparsers.add_parser("render-puhti-bundle", help="Render all Puhti batch scripts and submit helper.")
    render_all_parser.add_argument("--artifact-root", required=True)
    render_all_parser.add_argument("--project-root", required=True)

    write_bundle_parser = subparsers.add_parser("write-puhti-bundle", help="Write the full Puhti bundle to an output directory.")
    write_bundle_parser.add_argument("--artifact-root", required=True)
    write_bundle_parser.add_argument("--project-root", required=True)
    write_bundle_parser.add_argument("--output-dir", required=True)

    rsync_parser = subparsers.add_parser("print-rsync", help="Print the recommended rsync command for Puhti sync.")
    rsync_parser.add_argument("--local-project-root", required=True)
    rsync_parser.add_argument("--local-artifact-root", required=True)
    rsync_parser.add_argument("--puhti-target", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = default_phase1_manifest()

    if args.command == "run":
        runner = OfflinePaperRunner(manifest, artifact_root=Path(args.artifact_root))
        summary = runner.run(args.run_id)
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    if args.command == "list":
        payload = [run.run_id for run in manifest.runs]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for run_id in payload:
                print(run_id)
        return 0

    if args.command == "aggregate":
        paths = write_aggregate_reports(args.artifact_root)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return 0

    if args.command == "compare-latest":
        report = latest_paper_report(args.artifact_root)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "generate-figures":
        outputs = generate_phase1_svg_figures(args.artifact_root)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return 0

    if args.command == "render-puhti":
        print(
            render_puhti_batch_script(
                manifest=manifest,
                run_id=args.run_id,
                artifact_root=args.artifact_root,
                project_root=args.project_root,
            )
        )
        return 0

    if args.command == "render-puhti-bundle":
        bundle = render_puhti_bundle(
            manifest=manifest,
            artifact_root=args.artifact_root,
            project_root=args.project_root,
        )
        print(json.dumps(bundle, indent=2))
        return 0

    if args.command == "write-puhti-bundle":
        paths = write_puhti_bundle(
            manifest=manifest,
            output_dir=args.output_dir,
            artifact_root=args.artifact_root,
            project_root=args.project_root,
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return 0

    if args.command == "print-rsync":
        print(
            render_rsync_command(
                local_project_root=args.local_project_root,
                local_artifact_root=args.local_artifact_root,
                puhti_target=args.puhti_target,
            )
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
