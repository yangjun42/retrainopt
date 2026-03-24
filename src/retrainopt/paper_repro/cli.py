"""Internal CLI for Phase 1 offline paper reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import default_phase1_manifest
from .runner import OfflinePaperRunner
from .slurm import render_puhti_batch_script


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

    render_parser = subparsers.add_parser("render-puhti", help="Render a Puhti batch script for one run.")
    render_parser.add_argument("--run-id", required=True)
    render_parser.add_argument("--artifact-root", required=True)
    render_parser.add_argument("--project-root", required=True)

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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
