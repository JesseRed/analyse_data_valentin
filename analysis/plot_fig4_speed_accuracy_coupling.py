#!/usr/bin/env python3
"""Figure 4 – Block-level speed–accuracy coupling by group and day.

Two-panel scatter + model-based regression lines (Day × errorRate interaction),
one panel per group, shared axes.

Output:
  analysis/outputs/04_figures/fig4_speed_accuracy_coupling.pdf
  analysis/outputs/04_figures/fig4_speed_accuracy_coupling_600dpi.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
BLOCK_CSV = ROOT / "analysis" / "outputs" / "02_metrics" / "block_level_metrics.csv"
OUT_DIR = ROOT / "analysis" / "outputs" / "04_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib settings (consistent with Fig 1–3)
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

COLOURS = {1: "#0072B2", 2: "#D55E00"}   # Day 1 = blue, Day 2 = orange
SCATTER_ALPHA = 0.12
SCATTER_SIZE = 12
LINE_LW = 2.2
CI_ALPHA = 0.17
N_GRID = 200
MIN_HITS = 4   # minimum hits per block to include


# ---------------------------------------------------------------------------
# Load & aggregate to one row per PID × day × BlockNumber
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Collapse sequences/conditions: average RT and error rate across rows
    # that share the same PID × day × BlockNumber
    agg = (
        df.groupby(["PID", "Group", "day", "BlockNumber"], dropna=False)
        .agg(
            meanRT_hit_ms=("meanRT_hit_ms", "mean"),
            errorRate=("errorRate", "mean"),
            nHits=("nHits", "sum"),
        )
        .reset_index()
    )
    # Keep only blocks with enough hits for a stable RT estimate
    agg = agg[agg["nHits"] >= MIN_HITS].copy()
    agg = agg[agg["meanRT_hit_ms"].notna()].copy()
    return agg


# ---------------------------------------------------------------------------
# Compute shared axis limits
# ---------------------------------------------------------------------------
def compute_limits(data: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    rt_p1 = data["meanRT_hit_ms"].quantile(0.01)
    rt_p99 = data["meanRT_hit_ms"].quantile(0.99)
    pad = (rt_p99 - rt_p1) * 0.05
    ylim = (rt_p1 - pad, rt_p99 + pad)

    er_p99 = data["errorRate"].quantile(0.99)
    xmax = min(0.50, er_p99 + 0.03)
    xmin = -0.01   # tiny left margin so x=0 points are visible
    xlim = (xmin, xmax)
    return xlim, ylim


# ---------------------------------------------------------------------------
# Fit interaction model per group and produce prediction grid + CI
# ---------------------------------------------------------------------------
def fit_and_predict(
    data: pd.DataFrame, group: str, xlim: tuple[float, float]
) -> dict[int, dict[str, np.ndarray]]:
    """Return {day: {'x', 'y_hat', 'ci_lo', 'ci_hi'}} for one group."""
    sub = data[data["Group"] == group].copy()
    sub["day"] = sub["day"].astype(int)

    # OLS with cluster-robust (HC3) SEs
    model = smf.ols(
        "meanRT_hit_ms ~ C(day) + errorRate + C(day):errorRate",
        data=sub,
    ).fit(cov_type="HC3")

    x_grid = np.linspace(xlim[0], xlim[1], N_GRID)
    result: dict[int, dict[str, np.ndarray]] = {}

    for day in (1, 2):
        new_data = pd.DataFrame({"day": [day] * N_GRID, "errorRate": x_grid})
        pred = model.get_prediction(new_data)
        frame = pred.summary_frame(alpha=0.05)
        result[day] = {
            "x": x_grid,
            "y_hat": frame["mean"].to_numpy(),
            "ci_lo": frame["mean_ci_lower"].to_numpy(),
            "ci_hi": frame["mean_ci_upper"].to_numpy(),
        }

    return result


# ---------------------------------------------------------------------------
# Draw one panel
# ---------------------------------------------------------------------------
def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    preds: dict[int, dict[str, np.ndarray]],
    group: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_legend: bool = False,
) -> None:
    sub = data[data["Group"] == group]

    # Scatter – coloured by day; small horizontal jitter to break discrete stripes
    rng = np.random.default_rng(42)
    for day in (1, 2):
        d = sub[sub["day"] == day]
        jitter = rng.uniform(-0.008, 0.008, size=len(d))
        ax.scatter(
            d["errorRate"].to_numpy() + jitter,
            d["meanRT_hit_ms"].to_numpy(),
            c=COLOURS[day],
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            linewidths=0,
            zorder=1,
            rasterized=True,
        )

    # Model lines + CI
    for day in (1, 2):
        p = preds[day]
        colour = COLOURS[day]
        ax.fill_between(
            p["x"], p["ci_lo"], p["ci_hi"],
            color=colour, alpha=CI_ALPHA, linewidth=0, zorder=2,
        )
        ax.plot(
            p["x"], p["y_hat"],
            color=colour, lw=LINE_LW, zorder=3,
            label=f"Day {day}",
        )

    # Annotation: n participants, n blocks
    n_pids = sub["PID"].nunique()
    n_blocks = len(sub)
    ax.text(
        0.03, 0.97,
        f"$n$ participants = {n_pids}\n$n$ blocks = {n_blocks}",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        va="top",
        ha="left",
    )

    # Axes
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    x_ticks = np.arange(0.0, xlim[1] + 0.001, 0.10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{v:.2f}" for v in x_ticks], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("Error rate (per block)", fontsize=11)

    group_labels = {"A": "Group A (VR+SRTT)", "B": "Group B (SRTT-only)"}
    ax.set_title(group_labels[group], fontsize=11, fontweight="bold", loc="left", pad=4)

    if show_legend:
        ax.legend(
            title=None,
            loc="upper right",
            fontsize=10,
            frameon=False,
            handlelength=1.8,
        )


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def make_figure(data: pd.DataFrame) -> plt.Figure:
    xlim, ylim = compute_limits(data)

    # Fit models
    preds_a = fit_and_predict(data, "A", xlim)
    preds_b = fit_and_predict(data, "B", xlim)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(7.09, 3.3),
        sharey=True,
        sharex=True,
    )
    fig.subplots_adjust(wspace=0.12, left=0.10, right=0.98, top=0.91, bottom=0.18)

    draw_panel(ax_a, data, preds_a, "A", xlim, ylim, show_legend=False)
    draw_panel(ax_b, data, preds_b, "B", xlim, ylim, show_legend=True)

    ax_a.set_ylabel("Reaction time (ms)", fontsize=11)
    ax_b.set_ylabel("")   # shared via sharey

    # Panel letters
    for ax, letter in [(ax_a, "A"), (ax_b, "B")]:
        ax.text(
            -0.08, 1.06, letter,
            transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left",
        )

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading block-level data…")
    data = load_data(BLOCK_CSV)
    print(f"  {len(data)} blocks after nHits ≥ {MIN_HITS} filter")
    print(f"  errorRate range: {data['errorRate'].min():.3f} – {data['errorRate'].max():.3f}")
    print(f"  RT range: {data['meanRT_hit_ms'].min():.0f} – {data['meanRT_hit_ms'].max():.0f} ms")

    print("Fitting models…")
    fig = make_figure(data)

    pdf_path = OUT_DIR / "fig4_speed_accuracy_coupling.pdf"
    png_path = OUT_DIR / "fig4_speed_accuracy_coupling_600dpi.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", transparent=True)
    print(f"Saved PDF → {pdf_path}")

    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight")
    print(f"Saved PNG → {png_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
