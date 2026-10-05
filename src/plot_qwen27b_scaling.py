"""Create the P0/P1 Qwen 4B-versus-27B scaling plot for the talk."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from talk_plot_style import GRID, MODEL_COLORS, MUTED, TEXT, light_tint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "figures" / "reproduced_qwen27b_scaling_eval_v2.png"
HARD_DIFFICULTY = "hard"


@dataclass(frozen=True)
class ModelRun:
    label: str
    color_key: str
    group: str
    p0_path: Path
    p1_path: Path
    emphasise: bool = False


def scored_path(scope: str, run: str, condition: str) -> Path:
    return ROOT / "results" / "evaluation_v2_silver" / scope / run / f"{condition}.jsonl"


MODELS = (
    ModelRun(
        "Grok 4.6 (Frontier-Referenz)",
        "grok",
        "frontier",
        scored_path("frontier", "grok-4.6-low-reasoning", "p0"),
        scored_path("frontier", "grok-4.6-low-reasoning", "p1"),
    ),
    ModelRun(
        "Qwen 3.5 4B vanilla",
        "vanilla",
        "qwen_4b",
        scored_path("offline", "eval-v2-qwen35-4b-vanilla", "p0"),
        scored_path("offline", "eval-v2-qwen35-4b-vanilla", "p1"),
    ),
    ModelRun(
        "Qwen 3.5 4B + SFT",
        "qwen",
        "qwen_4b",
        scored_path("offline", "eval-v2-qwen35-4b-sft-v4-1", "p0"),
        scored_path("offline", "eval-v2-qwen35-4b-sft-v4-1", "p1"),
        emphasise=True,
    ),
    ModelRun(
        "Qwen 3.8 27B vanilla",
        "vanilla",
        "qwen_27b",
        scored_path("offline", "eval-v2-qwen38-27b-vanilla-q4", "p0"),
        scored_path("offline", "eval-v2-qwen38-27b-vanilla-q4", "p1"),
    ),
    ModelRun(
        "Qwen 3.8 27B + SFT",
        "qwen",
        "qwen_27b",
        scored_path("offline", "eval-v2-qwen38-27b-sft-v4-r16-usc1", "p0"),
        scored_path("offline", "eval-v2-qwen38-27b-sft-v4-r16-usc1", "p1"),
        emphasise=True,
    ),
)


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"No scored rows in {path}")
    return rows


def accuracy(rows: list[dict]) -> tuple[int, int, float]:
    correct = sum(row.get("result_equivalent") is True for row in rows)
    return correct, len(rows), correct / len(rows)


def hard_accuracy(rows: list[dict]) -> tuple[int, int, float]:
    hard_rows = [row for row in rows if row.get("difficulty") == HARD_DIFFICULTY]
    if not hard_rows:
        raise ValueError(f"No {HARD_DIFFICULTY!r} rows found")
    return accuracy(hard_rows)


def make_plot(output: Path, show: bool = False) -> None:
    values = []
    for model in MODELS:
        p0_rows = load_rows(model.p0_path)
        p1_rows = load_rows(model.p1_path)
        values.append((model, accuracy(p0_rows), accuracy(p1_rows), hard_accuracy(p0_rows)))

    total = values[0][1][1]
    if any(p0[1] != total or p1[1] != total for _, p0, p1, _ in values):
        raise ValueError("All plotted models must use the same eval set size")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": TEXT,
            "axes.edgecolor": GRID,
        }
    )
    fig = plt.figure(figsize=(12.8, 5.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.22, 0.17, 0.53, 0.76])
    ax.set_facecolor("white")

    y_positions = []
    y = 0.0
    previous_group = None
    for model, _, _, _ in values:
        if previous_group is not None and model.group != previous_group:
            y += 0.65
        y_positions.append(y)
        previous_group = model.group
        y += 1.0

    bar_height = 0.30
    p0_offset = -0.18
    p1_offset = 0.18
    for y_position, (model, p0, p1, _) in zip(y_positions, values):
        model_color = MODEL_COLORS[model.color_key]
        fontweight = "bold" if model.emphasise else "normal"
        p0_bar = ax.barh(
            y_position + p0_offset,
            p0[2],
            height=bar_height,
            color=light_tint(model_color),
            zorder=3,
        )
        p1_bar = ax.barh(
            y_position + p1_offset,
            p1[2],
            height=bar_height,
            color=model_color,
            zorder=3,
        )
        for bar, value in ((p0_bar[0], p0[2]), (p1_bar[0], p1[2])):
            ax.text(
                value + 0.012,
                bar.get_y() + bar_height / 2,
                f"{value:.0%}",
                va="center",
                ha="left",
                fontsize=10.5,
                fontweight=fontweight,
                color=TEXT,
            )

    ax.set_yticks(y_positions, [model.label for model, _, _, _ in values])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.88)
    ax.xaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11.5)
    for tick_label, (model, _, _, _) in zip(ax.get_yticklabels(), values):
        if model.emphasise:
            tick_label.set_fontweight("bold")
    ax.tick_params(axis="x", length=0, labelsize=10, colors="#8A8A96")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))

    legend = [
        Patch(facecolor=light_tint(MODEL_COLORS["qwen"]), label="P0 · nur Schema"),
        Patch(facecolor=MODEL_COLORS["qwen"], label="P1 · Schema + Geschäftsregeln"),
    ]
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.22, 0.015),
        ncol=2,
        frameon=False,
        fontsize=11,
        labelcolor=MUTED,
        handlelength=1.6,
        handleheight=1.0,
    )

    # A compact, slide-ready result for difficult two-stage SQL.
    qwen4_hard = values[2][3]
    qwen27_hard = values[4][3]
    callout = fig.add_axes([0.80, 0.32, 0.18, 0.35])
    callout.set_facecolor("#F6F8FB")
    for spine in callout.spines.values():
        spine.set_visible(False)
    callout.set_xticks([])
    callout.set_yticks([])
    callout.text(0.08, 0.86, "HARD · CTE / WINDOW", transform=callout.transAxes,
                 fontsize=9.5, fontweight="bold", color=MUTED, va="top")
    callout.text(0.08, 0.59, f"4B + SFT   {qwen4_hard[2]:.0%}", transform=callout.transAxes,
                 fontsize=12.5, color=TEXT, va="center")
    callout.text(0.08, 0.39, f"27B + SFT  {qwen27_hard[2]:.0%}", transform=callout.transAxes,
                 fontsize=12.5, fontweight="bold", color=TEXT, va="center")
    callout.text(0.08, 0.15, f"+{qwen27_hard[2] - qwen4_hard[2]:.0%} Punkte", transform=callout.transAxes,
                 fontsize=14, fontweight="bold", color=MODEL_COLORS["qwen"], va="center")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    make_plot(args.output, show=args.show)