#!/usr/bin/env python3
"""Figure 2 – Model-based simple effects of the structured advantage.

Reads EMM-based contrasts from the primary LMM output and produces a
two-panel dot plot (Panel A: simple effects per Group×Day; Panel B: DoD).

Output:
  analysis/outputs/04_figures/fig2_emm_simple_effects.pdf
  analysis/outputs/04_figures/fig2_emm_simple_effects_600dpi.png
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
CONTRASTS_CSV = (
    ROOT / "analysis" / "outputs" / "05_additional_analyses"
    / "planned_contrasts_simple_effects.csv"
)
DOD_CSV = (
    ROOT / "analysis" / "outputs" / "05_additional_analyses"
    / "supp_time_trend_DoD_linear.csv"
)
OUT_DIR = ROOT / "analysis" / "outputs" / "04_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib / font settings (consistent with Figure 1)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
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

COLOURS = {"A": "#0072B2", "B": "#D55E00"}
MARKER_SIZE = 7
ERRORBAR_LW = 1.5
CAPSIZE = 3
LINE_LW = 1.5
REF_LINE_COLOUR = "#555555"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_simple_effects(path: Path) -> pd.DataFrame:
    """Return the four structured_minus_random rows as % values."""
    df = pd.read_csv(path)
    se = df[df["contrast"] == "structured_minus_random"].copy()
    se["pct"] = se["pct_change"] * 100
    se["pct_ci_low"] = se["pct_ci_low"] * 100
    se["pct_ci_high"] = se["pct_ci_high"] * 100
    se["day"] = se["day"].astype(int)
    return se[["Group", "day", "pct", "pct_ci_low", "pct_ci_high"]].reset_index(drop=True)


def load_dod(path: Path) -> dict[str, float]:
    """Return DoD as % with CI."""
    df = pd.read_csv(path)
    row = df.iloc[0]
    return {
        "pct": float(row["pct_change"]) * 100,
        "ci_low": float(row["pct_ci_low"]) * 100,
        "ci_high": float(row["pct_ci_high"]) * 100,
    }


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
def make_figure(se: pd.DataFrame, dod: dict[str, float]) -> plt.Figure:
    # Two-panel layout with width_ratios 3:1
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(7.09, 3.3),
        gridspec_kw={"width_ratios": [3, 1]},
        sharey=True,
    )
    fig.subplots_adjust(wspace=0.25, left=0.10, right=0.98, top=0.92, bottom=0.20)

    # ------------------------------------------------------------------
    # Determine shared y-limits from all data
    # ------------------------------------------------------------------
    all_vals = (
        list(se["pct_ci_low"])
        + list(se["pct_ci_high"])
        + [dod["ci_low"], dod["ci_high"]]
    )
    ymin = min(all_vals) - 1.5
    ymax = max(all_vals) + 1.5

    # ------------------------------------------------------------------
    # Panel A – simple effects dot plot
    # ------------------------------------------------------------------
    # x positions: A-Day1=0, A-Day2=1, B-Day1=3, B-Day2=4  (gap between groups)
    x_map = {("A", 1): 0, ("A", 2): 1, ("B", 1): 3, ("B", 2): 4}
    x_labels = {0: "A  Day 1", 1: "A  Day 2", 3: "B  Day 1", 4: "B  Day 2"}

    for _, row in se.iterrows():
        grp, day = row["Group"], int(row["day"])
        x = x_map[(grp, day)]
        col = COLOURS[grp]
        ax_a.errorbar(
            x, row["pct"],
            yerr=[[row["pct"] - row["pct_ci_low"]], [row["pct_ci_high"] - row["pct"]]],
            fmt="o",
            color=col,
            markersize=MARKER_SIZE,
            elinewidth=ERRORBAR_LW,
            capsize=CAPSIZE,
            capthick=ERRORBAR_LW,
            zorder=3,
        )

    # Connection lines within each group
    for grp in ("A", "B"):
        sub = se[se["Group"] == grp].sort_values("day")
        xs = [x_map[(grp, int(r["day"]))] for _, r in sub.iterrows()]
        ys = list(sub["pct"])
        ax_a.plot(xs, ys, color=COLOURS[grp], lw=LINE_LW, zorder=2, alpha=0.8)

    # Reference line at y=0
    ax_a.axhline(0, color=REF_LINE_COLOUR, lw=0.8, alpha=0.6, zorder=1)

    # Axes formatting
    ax_a.set_xlim(-0.7, 4.7)
    ax_a.set_ylim(ymin, ymax)
    ax_a.set_xticks(list(x_labels.keys()))
    ax_a.set_xticklabels(list(x_labels.values()), fontsize=9)
    ax_a.set_ylabel("Structured advantage\n(Structured–Random, % RT)", fontsize=11)
    ax_a.tick_params(axis="y", labelsize=9)
    ax_a.tick_params(axis="x", length=0)   # suppress x tick marks, labels suffice

    # Panel label
    ax_a.text(
        -0.08, 1.06, "A",
        transform=ax_a.transAxes,
        fontsize=11, fontweight="bold", va="top", ha="left",
    )

    # Legend (proxy artists, upper right)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color=COLOURS["A"], lw=LINE_LW,
               markersize=MARKER_SIZE - 1, label="Group A (VR+SRTT)"),
        Line2D([0], [0], marker="o", color=COLOURS["B"], lw=LINE_LW,
               markersize=MARKER_SIZE - 1, label="Group B (SRTT-only)"),
    ]
    ax_a.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
        frameon=False,
        handlelength=1.8,
    )

    # ------------------------------------------------------------------
    # Panel B – DoD
    # ------------------------------------------------------------------
    ax_b.axhline(0, color=REF_LINE_COLOUR, lw=0.8, alpha=0.6, zorder=1)

    dod_x = 0
    ax_b.errorbar(
        dod_x, dod["pct"],
        yerr=[[dod["pct"] - dod["ci_low"]], [dod["ci_high"] - dod["pct"]]],
        fmt="D",                    # diamond marker
        color="#444444",
        markersize=MARKER_SIZE,
        elinewidth=ERRORBAR_LW,
        capsize=CAPSIZE,
        capthick=ERRORBAR_LW,
        zorder=3,
    )

    # Value annotation to the right of the point
    ax_b.text(
        dod_x + 0.15,
        dod["pct"],
        f"{dod['pct']:.1f}%",
        fontsize=10,
        color="#444444",
        va="center",
        ha="left",
    )

    # Axes formatting
    ax_b.set_xlim(-0.8, 0.9)
    ax_b.set_xticks([dod_x])
    ax_b.set_xticklabels(["DoD"], fontsize=9)
    ax_b.tick_params(axis="x", length=0)
    ax_b.set_title("Difference-in-\ndifferences", fontsize=10, pad=4)

    # Panel label
    ax_b.text(
        -0.18, 1.06, "B",
        transform=ax_b.transAxes,
        fontsize=11, fontweight="bold", va="top", ha="left",
    )

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading contrasts…")
    se = load_simple_effects(CONTRASTS_CSV)
    dod = load_dod(DOD_CSV)

    print("Simple effects (%):")
    print(se.to_string(index=False))
    print(f"\nDoD: {dod['pct']:.2f}%  [{dod['ci_low']:.2f}%, {dod['ci_high']:.2f}%]")

    print("\nRendering figure…")
    fig = make_figure(se, dod)

    pdf_path = OUT_DIR / "fig2_emm_simple_effects.pdf"
    png_path = OUT_DIR / "fig2_emm_simple_effects_600dpi.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", transparent=True)
    print(f"Saved PDF → {pdf_path}")

    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight")
    print(f"Saved PNG → {png_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
