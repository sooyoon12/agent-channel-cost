"""Dependency-free SVG renderers for the two required paper figures."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Mapping


COLORS = {"B0": "#64748b", "I-H": "#2563eb", "T-H": "#0f9f78", "O": "#d97706"}


def _svg_bar_chart(title: str, labels: list[str], values: list[float], colors: list[str], y_label: str) -> str:
    width, height = 760, 430
    left, top, bottom = 90, 65, 70
    plot_height = height - top - bottom
    maximum = max([abs(value) for value in values] + [1e-12])
    zero_y = top + plot_height if min(values, default=0) >= 0 else top + plot_height / 2
    scale = (plot_height if min(values, default=0) >= 0 else plot_height / 2) / maximum
    spacing = (width - left - 30) / max(1, len(values))
    bars = []
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        bar_width = spacing * 0.55
        x = left + spacing * index + (spacing - bar_width) / 2
        pixels = abs(value) * scale
        y = zero_y - pixels if value >= 0 else zero_y
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{pixels:.1f}" fill="{color}" rx="3"/>'
            f'<text x="{x + bar_width / 2:.1f}" y="{zero_y + 24:.1f}" text-anchor="middle" font-size="13">{escape(label)}</text>'
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 7 if value >= 0 else y + pixels + 16:.1f}" text-anchor="middle" font-size="11">{value:.4g}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="19" font-weight="600">{escape(title)}</text>'
        f'<text transform="translate(20 {height / 2}) rotate(-90)" text-anchor="middle" font-size="12">{escape(y_label)}</text>'
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width - 30}" y2="{zero_y:.1f}" stroke="#334155"/>'
        + "".join(bars)
        + '</svg>\n'
    )


def write_required_figures(analysis: Mapping[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    conditions = [value for value in ("B0", "I-H", "T-H", "O") if value in analysis["conditions"]]
    baseline = analysis["conditions"].get("B0", {})
    baseline_channels = {
        "I": baseline.get("mean_local_i_input_tokens") or 0.0,
        "T": baseline.get("mean_local_t_input_tokens") or 0.0,
        "O": baseline.get("mean_provider_output_tokens") or 0.0,
    }
    token_labels: list[str] = []
    token_savings: list[float] = []
    token_colors = {"I": "#2563eb", "T": "#0f9f78", "O": "#d97706"}
    savings_colors: list[str] = []
    for condition in conditions:
        summary = analysis["conditions"][condition]
        observed = {
            "I": summary.get("mean_local_i_input_tokens") or 0.0,
            "T": summary.get("mean_local_t_input_tokens") or 0.0,
            "O": summary.get("mean_provider_output_tokens") or 0.0,
        }
        for channel in ("I", "T", "O"):
            token_labels.append(f"{condition}-{channel}")
            token_savings.append(baseline_channels[channel] - observed[channel])
            savings_colors.append(token_colors[channel])
    figure1 = _svg_bar_chart(
        "I/T/O nominal token saving (end-to-end cost shown below)",
        token_labels,
        token_savings,
        savings_colors,
        "Mean tokens saved vs B0",
    )
    cost_text = " | ".join(
        f"{condition}: ${analysis['conditions'][condition].get('mean_model_list_cost_usd', 0):.6f}"
        for condition in conditions
    )
    figure1 = figure1.replace(
        "</svg>",
        f'<text x="380" y="420" text-anchor="middle" font-size="12">Mean end-to-end list cost — {escape(cost_text)}</text></svg>',
    )
    first = directory / "figure1_channel_saving.svg"
    first.write_text(figure1, encoding="utf-8")

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for condition in ("I-H", "T-H"):
        if condition not in analysis["cost_transfer"]:
            continue
        aggregate = analysis["cost_transfer"][condition]["aggregate"]
        labels.extend((f"{condition} G", f"{condition} B"))
        values.extend((aggregate["gross_saving"], aggregate["behavioral_displacement"]))
        colors.extend((COLORS[condition], "#dc2626"))
    figure2 = _svg_bar_chart(
        "Gross saving and behavioral displacement",
        labels,
        values,
        colors,
        "Aggregate list-price USD",
    )
    cdr_text = " | ".join(
        f"{condition} CDR={analysis['cost_transfer'][condition]['aggregate'].get('cdr')}"
        for condition in ("I-H", "T-H")
        if condition in analysis["cost_transfer"]
    )
    figure2 = figure2.replace(
        "</svg>",
        f'<text x="380" y="420" text-anchor="middle" font-size="12">{escape(cdr_text)}</text></svg>',
    )
    second = directory / "figure2_cdr_components.svg"
    second.write_text(figure2, encoding="utf-8")
    return first, second


def write_followup_success_figure(
    comparison: list[Mapping[str, Any]],
    ratio_analysis: Mapping[str, Any],
    guard_analysis: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Plot success across ratios and the contract-preserving architecture."""

    initial = {row["condition"]: row for row in comparison}
    cells = {
        (cell["condition"], float(cell["target_ratio"])): cell
        for cell in ratio_analysis["cells"]
    }
    guard = next(
        value for value in guard_analysis["methods"] if value["method"] == "contract_guard"
    )
    labels = ["I .4", "I .6", "I .8", "T .4", "T .6", "T .8", "Guard"]
    values = [
        100 * float(initial["I-H"]["success_rate"]),
        100 * float(cells[("I-H", 0.6)]["success_rate_planned"]),
        100 * float(cells[("I-H", 0.8)]["success_rate_planned"]),
        100 * float(initial["T-H"]["success_rate"]),
        100 * float(cells[("T-H", 0.6)]["success_rate_planned"]),
        100 * float(cells[("T-H", 0.8)]["success_rate_planned"]),
        100 * float(guard["success_rate_planned"]),
    ]
    colors = ["#2563eb"] * 3 + ["#0f9f78"] * 3 + ["#d97706"]
    width, height = 800, 430
    left, top, bottom = 70, 55, 70
    plot_height = height - top - bottom
    spacing = (width - left - 25) / len(values)
    bars = []
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        bar_width = spacing * 0.58
        x = left + index * spacing + (spacing - bar_width) / 2
        bar_height = plot_height * value / 100
        y = top + plot_height - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="3"/>'
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-size="12">{value:.1f}%</text>'
            f'<text x="{x + bar_width / 2:.1f}" y="{top + plot_height + 23:.1f}" text-anchor="middle" font-size="12">{escape(label)}</text>'
        )
    safety_y = top + plot_height * 0.05
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="19" font-weight="600">Success recovery and contract-preserving compression</text>'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - 25}" y2="{top + plot_height}" stroke="#334155"/>'
        f'<line x1="{left}" y1="{safety_y:.1f}" x2="{width - 25}" y2="{safety_y:.1f}" stroke="#dc2626" stroke-dasharray="6 5"/>'
        f'<text x="{width - 30}" y="{safety_y - 6:.1f}" text-anchor="end" font-size="11" fill="#dc2626">95% safety threshold</text>'
        f'<text transform="translate(18 {height / 2}) rotate(-90)" text-anchor="middle" font-size="12">Planned-run success rate</text>'
        + "".join(bars)
        + '</svg>\n'
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path
