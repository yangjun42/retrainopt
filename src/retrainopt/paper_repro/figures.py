"""Dependency-light figure generation for Phase 1 offline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def generate_phase1_svg_figures(artifact_root: Path | str) -> Dict[str, Path]:
    root = Path(artifact_root)
    summaries_dir = root / "summaries"
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    if summaries_dir.exists():
        for path in sorted(summaries_dir.glob("*.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))

    outputs: Dict[str, Path] = {}
    total_cost_svg = figures_dir / "total_cost_by_run.svg"
    total_cost_svg.write_text(_build_bar_chart_svg(rows), encoding="utf-8")
    outputs[total_cost_svg.name] = total_cost_svg
    return outputs


def _build_bar_chart_svg(rows: List[dict]) -> str:
    width = 960
    height = 480
    margin_left = 220
    margin_bottom = 80
    plot_width = width - margin_left - 40
    plot_height = height - 60 - margin_bottom
    max_value = max((float(row.get("total_cost", 0.0)) for row in rows), default=1.0)
    if max_value <= 0:
        max_value = 1.0

    bar_height = 24
    gap = 10
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:12px;} .title{font-size:16px;font-weight:bold;} .bar{fill:#1f77b4;} .axis{stroke:#444;stroke-width:1;}</style>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" class="title">Phase 1 Total Cost by Run</text>',
        f'<line x1="{margin_left}" y1="50" x2="{margin_left}" y2="{50 + plot_height}" class="axis" />',
        f'<line x1="{margin_left}" y1="{50 + plot_height}" x2="{margin_left + plot_width}" y2="{50 + plot_height}" class="axis" />',
    ]

    for index, row in enumerate(rows):
        label = str(row.get("run_id", f"run-{index}"))
        value = float(row.get("total_cost", 0.0))
        bar_width = 0 if max_value == 0 else (value / max_value) * plot_width
        y = 60 + index * (bar_height + gap)
        svg_lines.append(
            f'<text x="{margin_left - 10}" y="{y + 16}" text-anchor="end">{_escape(label)}</text>'
        )
        svg_lines.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" class="bar" />'
        )
        svg_lines.append(
            f'<text x="{margin_left + bar_width + 8:.2f}" y="{y + 16}">{value:.4f}</text>'
        )

    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
