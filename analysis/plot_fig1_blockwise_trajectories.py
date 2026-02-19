#!/usr/bin/env python3
"""Figure 1 – Block-wise learning trajectories by Group × Day × Condition.

Publication-ready 2×2 panel figure. Output:
  analysis/outputs/04_figures/fig1_blockwise_trajectories.pdf
  analysis/outputs/04_figures/fig1_blockwise_trajectories_600dpi.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
BLOCK_CSV = ROOT / "analysis" / "outputs" / "02_metrics" / "block_level_metrics.csv"
OUT_DIR = ROOT / "analysis" / "outputs" / "04_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib / font settings (publication-ready)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,        # TrueType → editable in Illustrator
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
    }
)

# ---------------------------------------------------------------------------
# Colour specification (colour-blind-friendly)
# ---------------------------------------------------------------------------
COLOURS = {
    "structured": "#0072B2",   # blue
    "random":     "#D55E00",   # orange
}
CI_ALPHA = 0.20
INDIVIDUAL_ALPHA = 0.10
INDIVIDUAL_LW = 0.6
MAIN_LW = 2.0

# ---------------------------------------------------------------------------
# Load & aggregate
# ---------------------------------------------------------------------------
def load_and_aggregate(
    path: Path, min_coverage_frac: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (panel_stats, participant_block) DataFrames.

    min_coverage_frac: blocks where fewer than this fraction of the panel's
    participants have data for a given condition are suppressed (set to NaN).
    This prevents single-participant spikes caused by anomalous block
    assignments (e.g. restarted sessions with duplicate block numbering).
    Only blocks 1–119 are used.
    """
    df = pd.read_csv(path)

    # Restrict to the canonical block range
    df = df[df["BlockNumber"].between(1, 119)].copy()

    # Step 1: participant-level mean per block × condition
    # (each participant may have multiple sequence rows per block for structured)
    part_block = (
        df.groupby(["PID", "Group", "day", "BlockNumber", "condition"], dropna=False)[
            "meanRT_hit_ms"
        ]
        .mean()
        .reset_index()
        .rename(columns={"meanRT_hit_ms": "rt_participant"})
    )

    # Count total participants per Group × Day panel
    n_panel = (
        part_block.groupby(["Group", "day"])["PID"]
        .nunique()
        .reset_index()
        .rename(columns={"PID": "n_total"})
    )

    # Step 2: group-level stats across participants
    def agg_fn(g: pd.DataFrame) -> pd.Series:
        vals = g["rt_participant"].dropna().to_numpy()
        n = len(vals)
        m = float(np.mean(vals)) if n > 0 else np.nan
        se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        return pd.Series({"mean_rt": m, "se_rt": se, "n_participants": n})

    panel_stats = (
        part_block.groupby(["Group", "day", "BlockNumber", "condition"])
        .apply(agg_fn, include_groups=False)
        .reset_index()
    )

    # Drop cells where coverage is below threshold (single-participant
    # anomalies from PID 9's double session or other edge cases).
    # Rows are dropped rather than set to NaN so that matplotlib connects
    # adjacent valid points without leaving explicit gaps in the line.
    panel_stats = panel_stats.merge(n_panel, on=["Group", "day"], how="left")
    panel_stats = panel_stats[
        panel_stats["n_participants"] >= panel_stats["n_total"] * min_coverage_frac
    ].copy()

    panel_stats["ci_lo"] = panel_stats["mean_rt"] - 1.96 * panel_stats["se_rt"]
    panel_stats["ci_hi"] = panel_stats["mean_rt"] + 1.96 * panel_stats["se_rt"]

    return panel_stats, part_block


# ---------------------------------------------------------------------------
# Determine shared y-limits (P1–P99 of block means + 5 % padding)
# ---------------------------------------------------------------------------
def compute_ylim(panel_stats: pd.DataFrame) -> tuple[float, float]:
    vals = panel_stats["mean_rt"].dropna().to_numpy()
    p1, p99 = np.percentile(vals, 1), np.percentile(vals, 99)
    pad = (p99 - p1) * 0.05
    return p1 - pad, p99 + pad


# ---------------------------------------------------------------------------
# Draw one panel
# ---------------------------------------------------------------------------
def draw_panel(
    ax: plt.Axes,
    panel_stats: pd.DataFrame,
    part_block: pd.DataFrame,
    group: str,
    day: int,
    show_individuals: bool = True,
) -> None:
    sub_stats = panel_stats[
        (panel_stats["Group"] == group) & (panel_stats["day"] == day)
    ]
    sub_part = part_block[
        (part_block["Group"] == group)
        & (part_block["day"] == day)
        & (part_block["BlockNumber"].between(1, 119))
    ]

    # Individual participant lines (z-order below main lines)
    if show_individuals:
        for pid, pid_df in sub_part.groupby("PID"):
            for cond, cond_df in pid_df.groupby("condition"):
                cond_sorted = cond_df.sort_values("BlockNumber")
                ax.plot(
                    cond_sorted["BlockNumber"],
                    cond_sorted["rt_participant"],
                    color=COLOURS[cond],
                    lw=INDIVIDUAL_LW,
                    alpha=INDIVIDUAL_ALPHA,
                    zorder=1,
                    rasterized=True,
                )

    # CI bands and main lines
    for cond in ["structured", "random"]:
        sub_cond = sub_stats[sub_stats["condition"] == cond].sort_values("BlockNumber")
        if sub_cond.empty:
            continue
        colour = COLOURS[cond]
        ax.fill_between(
            sub_cond["BlockNumber"],
            sub_cond["ci_lo"],
            sub_cond["ci_hi"],
            color=colour,
            alpha=CI_ALPHA,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            sub_cond["BlockNumber"],
            sub_cond["mean_rt"],
            color=colour,
            lw=MAIN_LW,
            solid_capstyle="round",
            antialiased=True,
            zorder=3,
            label="Structured (repeating)" if cond == "structured" else "Random",
        )

    # n annotation (top-right, inside axes)
    n_participants = int(
        sub_part["PID"].nunique()
    )
    ax.text(
        0.98,
        0.95,
        f"$n$ = {n_participants}",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="right",
        va="top",
    )


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def make_figure(
    panel_stats: pd.DataFrame,
    part_block: pd.DataFrame,
    ylim: tuple[float, float],
    show_individuals: bool = True,
) -> plt.Figure:
    # Panel layout: rows = groups (A top, B bottom), cols = days (1 left, 2 right)
    PANELS = [
        ("A", 1, 0, 0),   # (group, day, row_idx, col_idx)
        ("A", 2, 0, 1),
        ("B", 1, 1, 0),
        ("B", 2, 1, 1),
    ]
    PANEL_LETTERS = ["A", "B", "C", "D"]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(7.09, 4.72),        # 180 × 120 mm
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(wspace=0.15, hspace=0.18)

    x_ticks = [1, 20, 40, 60, 80, 100, 119]

    group_labels = {"A": "Group A (VR+SRTT)", "B": "Group B (SRTT-only)"}
    day_labels   = {1: "Day 1", 2: "Day 2"}

    for panel_idx, (group, day, ri, ci) in enumerate(PANELS):
        ax = axes[ri, ci]
        draw_panel(ax, panel_stats, part_block, group, day, show_individuals=show_individuals)

        # Panel letter (top-left, outside plot area style)
        ax.text(
            -0.08, 1.06,
            PANEL_LETTERS[panel_idx],
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
        )
        # Panel title
        title = f"{group_labels[group]} – {day_labels[day]}"
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=4)

        # Axes formatting
        ax.set_ylim(*ylim)
        ax.set_xlim(1, 119)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis="both", labelsize=9)

        # Axis labels only on outer panels
        if ri == 1:
            ax.set_xlabel("Block number", fontsize=11)
        if ci == 0:
            ax.set_ylabel("Reaction time (ms)", fontsize=11)

        # Subtle y-grid only
        ax.yaxis.grid(True, linewidth=0.4, color="#cccccc", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)

    # Single shared legend, placed outside grid (top centre)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Deduplicate (individual lines add duplicates)
    seen: dict[str, object] = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    fig.legend(
        seen.values(),
        seen.keys(),
        loc="upper center",
        ncol=2,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        handlelength=2.0,
    )

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading block-level metrics…")
    panel_stats, part_block = load_and_aggregate(BLOCK_CSV)

    ylim = compute_ylim(panel_stats)
    print(f"Shared y-limits: {ylim[0]:.0f}–{ylim[1]:.0f} ms")

    # Individual participant lines create a "spaghetti" effect with this
    # many participants and should not be shown (per spec: weglassen wenn unruhig).
    show_individuals = False

    print("Rendering figure…")
    fig = make_figure(panel_stats, part_block, ylim, show_individuals=show_individuals)

    pdf_path = OUT_DIR / "fig1_blockwise_trajectories.pdf"
    png_path = OUT_DIR / "fig1_blockwise_trajectories_600dpi.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", transparent=True)
    print(f"Saved PDF → {pdf_path}")

    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight")
    print(f"Saved PNG → {png_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
