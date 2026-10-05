"""Create the P0/P1 SFT-versus-frontier comparison plot for the talk."""

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
DEFAULT_OUTPUT = ROOT / "results" / "figures" / "reproduced_sft_4b_competitive_eval_v2.png"


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
        "open_weights",
        scored_path("offline", "eval-v2-qwen35-4b-vanilla", "p0"),
        scored_path("offline", "eval-v2-qwen35-4b-vanilla", "p1"),
    ),
    ModelRun(
        "Qwen 3.5 4B + SFT",
        "qwen",
        "open_weights",
        scored_path("offline", "eval-v2-qwen35-4b-sft-v4-1", "p0"),
        scored_path("offline", "eval-v2-qwen35-4b-sft-v4-1", "p1"),
        emphasise=True,
    ),
    # Add Qwen 3.8 27B + SFT here once its evaluation has completed.
    ModelRun(
        "Gemma 3 4B vanilla",
        "vanilla",
        "open_weights",
        scored_path("offline", "eval-v2-gemma4-e4b-vanilla", "p0"),
        scored_path("offline", "eval-v2-gemma4-e4b-vanilla", "p1"),
    ),
    ModelRun(
        "Gemma 3 4B + SFT",
        "gemma",
        "open_weights",
        scored_path("offline", "eval-v2-gemma4-e4b-sft-v4", "p0"),
        scored_path("offline", "eval-v2-gemma4-e4b-sft-v4", "p1"),
        emphasise=True,
    ),
)


def load_accuracy(path: Path) -> tuple[int, int, float]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"No scored rows in {path}")
    correct = sum(row.get("result_equivalent") is True for row in rows)
    return correct, len(rows), correct / len(rows)


def make_plot(output: Path, show: bool = False) -> None:
    values = [(model, load_accuracy(model.p0_path), load_accuracy(model.p1_path)) for model in MODELS]
    total = values[0][1][1]
    if any(p0[1] != total or p1[1] != total for _, p0, p1 in values):
        raise ValueError("All plotted models must use the same eval set size")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": TEXT,
            "axes.edgecolor": GRID,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_positions = []
    y = 0.0
    previous_group = None
    group_positions: dict[str, float] = {}
    for model, _, _ in values:
        if previous_group is not None and model.group != previous_group:
            y += 0.65
        y_positions.append(y)
        group_positions.setdefault(model.group, y)
        previous_group = model.group
        y += 1.0

    bar_height = 0.30
    p0_offset = -0.18
    p1_offset = 0.18
    for y_position, (model, p0, p1) in zip(y_positions, values):
        model_color = MODEL_COLORS[model.color_key]
        p0_value = p0[2]
        p1_value = p1[2]

        p0_bar = ax.barh(
            y_position + p0_offset,
            p0_value,
            height=bar_height,
            color=light_tint(model_color),
            zorder=3,
        )
        p1_bar = ax.barh(
            y_position + p1_offset,
            p1_value,
            height=bar_height,
            color=model_color,
            zorder=3,
        )

        fontweight = "bold" if model.emphasise else "normal"
        ax.text(
            p0_value + 0.012,
            p0_bar[0].get_y() + bar_height / 2,
            f"{p0_value:.0%}",
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight=fontweight,
            color=TEXT,
        )
        ax.text(
            p1_value + 0.012,
            p1_bar[0].get_y() + bar_height / 2,
            f"{p1_value:.0%}",
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight=fontweight,
            color=TEXT,
        )

    # The intentionally blank row after Grok separates the frontier reference
    # from the open-weight models without adding another visual hierarchy.

    ax.set_yticks(y_positions, [model.label for model, _, _ in values])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.84)
    ax.xaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=12)
    for tick_label, (model, _, _) in zip(ax.get_yticklabels(), values):
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
        loc="lower center",
        bbox_to_anchor=(0.50, 0.01),
        ncol=2,
        frameon=False,
        fontsize=11.5,
        labelcolor=MUTED,
        handlelength=1.6,
        handleheight=1.0,
    )
    plt.tight_layout(rect=[0.08, 0.08, 1, 1])

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
