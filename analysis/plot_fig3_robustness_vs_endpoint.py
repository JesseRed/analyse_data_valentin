#!/usr/bin/env python3
"""Figure 3 – Robustness of the dynamic DoD effect vs endpoint-style contrasts.

Panel A: Forest plot – DoD (% RT change) from primary model + 5 sensitivity checks.
Panel B: Endpoint-style between-group contrasts for day-2 sequence learning
         and retention, with permutation p-values.

Output:
  analysis/outputs/04_figures/fig3_robustness_vs_endpoint.pdf
  analysis/outputs/04_figures/fig3_robustness_vs_endpoint_600dpi.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "outputs"
OUT_FIGS = OUT / "04_figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib settings (consistent with Fig 1/2)
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

PRIMARY_COLOUR = "#0072B2"
GREY = "#666666"
LIGHT_GREY = "#999999"
REF_COLOUR = "#555555"


# ---------------------------------------------------------------------------
# Helper: convert log-scale DoD to percent + backtransformed CI
# ---------------------------------------------------------------------------
def log_to_pct(log_est: float, se: float, z: float = 1.96) -> tuple[float, float, float]:
    """Return (est_pct, ci_low_pct, ci_high_pct) from log-scale estimate and SE."""
    est_pct = 100.0 * (np.exp(log_est) - 1.0)
    ci_lo_pct = 100.0 * (np.exp(log_est - z * se) - 1.0)
    ci_hi_pct = 100.0 * (np.exp(log_est + z * se) - 1.0)
    return est_pct, ci_lo_pct, ci_hi_pct


# ---------------------------------------------------------------------------
# Load Panel A data – DoD sensitivity checks
# ---------------------------------------------------------------------------
def build_panel_a(out: Path) -> pd.DataFrame:
    """
    Collect DoD estimates from primary model and pre-specified sensitivity checks.

    Sign convention in CSVs:  positive coef = Group B has larger day-to-day
    structured-advantage growth than Group A.
    We flip the sign so that negative = Group A grows more (the expected direction).
    """
    rows: list[dict] = []

    # 1. Primary model (supp_time_trend_DoD_linear)
    dod_csv = out / "05_additional_analyses" / "supp_time_trend_DoD_linear.csv"
    d = pd.read_csv(dod_csv).iloc[0]
    est, lo, hi = log_to_pct(float(d["log_DoD"]), float(d["SE"]))
    rows.append(
        dict(label="Primary mixed model (LMM)", est=est, low=lo, high=hi, is_primary=True)
    )

    # 2–6. Random-effects / dependence sensitivity checks
    # All share sign convention: coef = B − A direction → flip sign
    re_csv = out / "07_additional_260219" / "a_random_effects_sensitivity.csv"
    re = pd.read_csv(re_csv)
    a5 = re[re["model"] == "A5_max_sensible"].iloc[0]
    est, lo, hi = log_to_pct(-float(a5["dod_log"]), float(a5["dod_se"]))
    rows.append(
        dict(label="RE sensitivity (A5 max sensible)", est=est, low=lo, high=hi, is_primary=False)
    )

    b5_csv = out / "07_additional_260219" / "b5_session_random_intercept.csv"
    b5 = pd.read_csv(b5_csv).iloc[0]
    est, lo, hi = log_to_pct(-float(b5["coef"]), float(b5["se"]))
    rows.append(
        dict(label="Session-level random intercept (B5)", est=est, low=lo, high=hi, is_primary=False)
    )

    b2_csv = out / "07_additional_260219" / "b2_cluster_robust_ols.csv"
    b2 = pd.read_csv(b2_csv).iloc[0]
    est, lo, hi = log_to_pct(-float(b2["coef"]), float(b2["se"]))
    rows.append(
        dict(label="Cluster-robust inference (B2)", est=est, low=lo, high=hi, is_primary=False)
    )

    b3_csv = out / "07_additional_260219" / "b3_aggregation_sensitivity.csv"
    b3 = pd.read_csv(b3_csv)
    chunk5 = b3[b3["analysis"] == "B3_chunk_5"].iloc[0]
    est, lo, hi = log_to_pct(-float(chunk5["coef"]), float(chunk5["se"]))
    rows.append(
        dict(label="Aggregation (5-block chunks, B3)", est=est, low=lo, high=hi, is_primary=False)
    )

    b4_csv = out / "07_additional_260219" / "b4_gee_ar1.csv"
    b4 = pd.read_csv(b4_csv).iloc[0]
    est, lo, hi = log_to_pct(-float(b4["coef"]), float(b4["se"]))
    rows.append(
        dict(label="AR(1) sensitivity / GEE (B4)", est=est, low=lo, high=hi, is_primary=False)
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Load Panel B data – endpoint contrasts + permutation p
# ---------------------------------------------------------------------------
def build_panel_b(out: Path) -> pd.DataFrame:
    """
    Day-2 sequence learning and retention: observed B−A difference in ms
    with two-sample t-test 95% CI and permutation p-values.
    """
    perm = pd.read_csv(out / "05_additional_analyses" / "permutation_tests.csv")
    pday = pd.read_csv(out / "02_metrics" / "participant_day_metrics.csv")
    ret = pd.read_csv(out / "02_metrics" / "participant_retention_metrics.csv")

    rows: list[dict] = []

    # Day-2 SeqLearning (B − A)
    d2 = pday[pday["day"] == 2]
    ga = d2[d2["Group"] == "A"]["SeqLearning_Index_all"].dropna()
    gb = d2[d2["Group"] == "B"]["SeqLearning_Index_all"].dropna()
    obs_diff = gb.mean() - ga.mean()
    t_res = stats.ttest_ind(gb, ga)
    df_t = len(ga) + len(gb) - 2
    se_diff = (gb.mean() - ga.mean()) / t_res.statistic if t_res.statistic != 0 else np.nan
    ci_margin = stats.t.ppf(0.975, df_t) * se_diff
    p_perm = float(perm.loc[perm["outcome"] == "SeqLearning_Index_all_day2", "perm_p_two_sided"].iloc[0])
    rows.append(
        dict(
            label="Day 2 seq. learning",
            est=obs_diff,
            low=obs_diff - ci_margin,
            high=obs_diff + ci_margin,
            p_perm=p_perm,
        )
    )

    # Retention Sequence (B − A)
    ra = ret[ret["Group"] == "A"]["Retention_Sequence"].dropna()
    rb = ret[ret["Group"] == "B"]["Retention_Sequence"].dropna()
    obs_diff_r = rb.mean() - ra.mean()
    t_res_r = stats.ttest_ind(rb, ra)
    df_t_r = len(ra) + len(rb) - 2
    se_diff_r = (rb.mean() - ra.mean()) / t_res_r.statistic if t_res_r.statistic != 0 else np.nan
    ci_margin_r = stats.t.ppf(0.975, df_t_r) * se_diff_r
    p_perm_r = float(perm.loc[perm["outcome"] == "Retention_Sequence", "perm_p_two_sided"].iloc[0])
    rows.append(
        dict(
            label="Retention seq. learning",
            est=obs_diff_r,
            low=obs_diff_r - ci_margin_r,
            high=obs_diff_r + ci_margin_r,
            p_perm=p_perm_r,
        )
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Draw figure
# ---------------------------------------------------------------------------
def make_figure(pa: pd.DataFrame, pb: pd.DataFrame) -> plt.Figure:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(7.09, 3.9),
        gridspec_kw={"width_ratios": [7, 3]},
    )
    fig.subplots_adjust(wspace=0.55, left=0.03, right=0.95, top=0.93, bottom=0.14)

    # ------------------------------------------------------------------
    # Panel A – Forest plot for DoD
    # ------------------------------------------------------------------
    n_a = len(pa)
    y_positions = list(range(n_a - 1, -1, -1))   # top-to-bottom

    x_vals_a = list(pa["est"]) + list(pa["low"]) + list(pa["high"])
    x_min_a = min(x_vals_a) - abs(min(x_vals_a)) * 0.15
    x_max_a = max(x_vals_a) + abs(max(x_vals_a)) * 0.20
    x_min_a = min(x_min_a, -0.5)   # ensure 0 line is clearly inside

    ax_a.axvline(0, color=REF_COLOUR, lw=0.8, alpha=0.6, zorder=1)

    for i, (_, row) in enumerate(pa.iterrows()):
        y = y_positions[i]
        colour = PRIMARY_COLOUR if row["is_primary"] else GREY
        ms = 6.5 if row["is_primary"] else 5.5
        lw = 2.0 if row["is_primary"] else 1.5
        ax_a.errorbar(
            row["est"], y,
            xerr=[[row["est"] - row["low"]], [row["high"] - row["est"]]],
            fmt="o",
            color=colour,
            markersize=ms,
            elinewidth=lw,
            capsize=3,
            capthick=lw,
            zorder=3,
        )
        # Value label: above the dot, centred on the estimate (data coords)
        label_text = f"{row['est']:.1f}%"
        p_text = "  p=0.002" if row["is_primary"] else ""
        ax_a.text(
            row["est"], y + 0.30,
            label_text + p_text,
            fontsize=8.5,
            color=colour,
            va="bottom",
            ha="center",
        )

    # y-axis row labels
    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels(list(pa["label"]), fontsize=9)
    ax_a.set_ylim(-0.7, n_a - 0.3)

    ax_a.set_xlim(x_min_a, x_max_a)
    # x ticks in clean steps
    x_tick_step = 3
    x_ticks = np.arange(
        int(np.floor(x_min_a / x_tick_step)) * x_tick_step,
        int(np.ceil(x_max_a / x_tick_step)) * x_tick_step + x_tick_step,
        x_tick_step,
    )
    ax_a.set_xticks(x_ticks)
    ax_a.set_xticklabels([f"{v:+.0f}%" for v in x_ticks], fontsize=9)
    ax_a.set_xlabel("Difference-in-differences (DoD), % RT change", fontsize=11)
    ax_a.tick_params(axis="y", length=0)

    # Subtle separator between primary and sensitivity rows
    ax_a.axhline(n_a - 1.5, color="#cccccc", lw=0.6, ls="--", zorder=0)

    ax_a.text(
        -0.02, 1.05, "A",
        transform=ax_a.transAxes,
        fontsize=11, fontweight="bold", va="top", ha="left",
    )
    ax_a.set_title("Dynamic DoD effect – primary model and sensitivity checks",
                   fontsize=10, loc="left", pad=5,
                   x=0.07)   # shift right to clear the "A" panel letter

    # ------------------------------------------------------------------
    # Panel B – Endpoint contrasts
    # ------------------------------------------------------------------
    n_b = len(pb)
    y_pos_b = list(range(n_b - 1, -1, -1))

    x_vals_b = list(pb["est"]) + list(pb["low"]) + list(pb["high"])
    x_range_b = max(abs(v) for v in x_vals_b) * 1.5
    x_min_b = -x_range_b
    x_max_b = x_range_b

    ax_b.axvline(0, color=REF_COLOUR, lw=0.8, alpha=0.6, zorder=1)

    LABEL_OFFSET = 0.16   # data-units above the marker
    PPERM_OFFSET = 0.16   # data-units below the marker

    for i, (_, row) in enumerate(pb.iterrows()):
        y = y_pos_b[i]
        ax_b.errorbar(
            row["est"], y,
            xerr=[[row["est"] - row["low"]], [row["high"] - row["est"]]],
            fmt="s",
            color=GREY,
            markersize=5.5,
            elinewidth=1.5,
            capsize=3,
            capthick=1.5,
            zorder=3,
        )
        # Row label: fixed offset ABOVE the marker, left-aligned
        ax_b.text(
            x_min_b + (x_max_b - x_min_b) * 0.03,
            y + LABEL_OFFSET,
            row["label"],
            fontsize=9,
            color="#333333",
            va="bottom",
            ha="left",
        )
        # p_perm: fixed offset BELOW the marker, left-aligned
        ax_b.text(
            x_min_b + (x_max_b - x_min_b) * 0.03,
            y - PPERM_OFFSET,
            f"p_perm={row['p_perm']:.2f}",
            fontsize=8.5,
            color=LIGHT_GREY,
            va="top",
            ha="left",
        )

    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels([])
    ax_b.set_ylim(-0.7, n_b - 0.3)

    ax_b.set_xlim(x_min_b, x_max_b)
    ax_b.set_xlabel("Between-group difference\n(B−A), ms", fontsize=11)
    ax_b.tick_params(axis="both", labelsize=9)
    ax_b.tick_params(axis="y", length=0)

    ax_b.text(
        -0.10, 1.05, "B",
        transform=ax_b.transAxes,
        fontsize=11, fontweight="bold", va="top", ha="left",
    )
    ax_b.set_title("Endpoint-style contrasts", fontsize=10, loc="left", pad=5)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Building Panel A data…")
    pa = build_panel_a(OUT)
    print(pa[["label", "est", "low", "high"]].to_string(index=False))

    print("\nBuilding Panel B data…")
    pb = build_panel_b(OUT)
    print(pb[["label", "est", "low", "high", "p_perm"]].to_string(index=False))

    print("\nRendering figure…")
    fig = make_figure(pa, pb)

    pdf_path = OUT_FIGS / "fig3_robustness_vs_endpoint.pdf"
    png_path = OUT_FIGS / "fig3_robustness_vs_endpoint_600dpi.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", transparent=True)
    print(f"Saved PDF → {pdf_path}")

    fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight")
    print(f"Saved PNG → {png_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
