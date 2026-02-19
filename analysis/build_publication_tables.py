#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "outputs" / "06_publication_tables"
MODELS_DIR = ROOT / "analysis" / "outputs" / "03_models"
ADD_DIR = ROOT / "analysis" / "outputs" / "05_additional_analyses"
METRICS_DIR = ROOT / "analysis" / "outputs" / "02_metrics"
QC_DIR = ROOT / "analysis" / "outputs" / "01_ingest_and_qc"

META_PATH = ROOT / "Datensatz_Round_2.csv"
INVENTORY_PATH = QC_DIR / "inventory_pid_day_files.csv"

try:
    from analysis.analysis_config import EXCLUDED_PIDS
except ModuleNotFoundError:  # pragma: no cover
    try:
        from analysis_config import EXCLUDED_PIDS  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        EXCLUDED_PIDS = []


def fmt_num(x, digits=3):
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def fmt_p(x):
    if pd.isna(x):
        return "NA"
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def fmt_mean_sd(x: pd.Series, digits: int = 1) -> str:
    x = pd.to_numeric(x, errors="coerce")
    m = float(x.mean())
    s = float(x.std(ddof=1))
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def fmt_n_pct(n: int, denom: int) -> str:
    if denom <= 0:
        return "NA"
    return f"{n} ({100*n/denom:.1f}%)"


def _safe_import_scipy_stats():
    try:  # pragma: no cover
        import scipy.stats as st  # type: ignore

        return st
    except Exception:  # pragma: no cover
        return None


def build_table1_demographics() -> pd.DataFrame:
    """
    Baseline demographics for the included two-day sample.

    Inclusion here mirrors the main analysis cohort:
    - Exclude PIDs in EXCLUDED_PIDS
    - Keep participants with both day files according to the inventory
    """
    if not META_PATH.exists() or not INVENTORY_PATH.exists():
        return pd.DataFrame()

    meta = pd.read_csv(META_PATH, sep=";")
    meta["PID"] = pd.to_numeric(meta["PID"], errors="coerce").astype("Int64")
    meta = meta.loc[~meta["PID"].isin(EXCLUDED_PIDS)].copy()

    inv = pd.read_csv(INVENTORY_PATH)
    both = inv.groupby("PID")["has_both_days"].max()
    included_pids = set(both[both].index.astype(int).tolist())
    meta = meta.loc[meta["PID"].astype(int).isin(included_pids)].copy()

    meta["Group"] = meta["Group"].astype(str)

    # Continuous vars
    meta["Age"] = pd.to_numeric(meta.get("Age"), errors="coerce")
    meta["Weight"] = pd.to_numeric(meta.get("Weight"), errors="coerce")
    meta["Biomag_height"] = pd.to_numeric(meta.get("Biomag_height"), errors="coerce")
    meta["BMI"] = meta["Weight"] / (meta["Biomag_height"] / 100.0) ** 2

    # Education (collapsed)
    if "Biomag_Bildungsgrad" in meta.columns:
        m = {
            "8-10. Klasse": "School",
            "Abitur": "School",
            "Ausbildung": "Apprenticeship",
            "Studium": "University degree",
        }
        meta["Education_cat"] = meta["Biomag_Bildungsgrad"].map(m).fillna("Missing/Other")
    else:
        meta["Education_cat"] = "NA"

    # Split groups after derived columns are created.
    A = meta.loc[meta["Group"] == "A"].copy()
    B = meta.loc[meta["Group"] == "B"].copy()
    nA, nB, nT = len(A), len(B), len(meta)

    st = _safe_import_scipy_stats()
    p_age = (
        fmt_p(float(st.ttest_ind(A["Age"], B["Age"], equal_var=False, nan_policy="omit").pvalue))
        if st is not None
        else "NA"
    )
    p_bmi = (
        fmt_p(float(st.ttest_ind(A["BMI"], B["BMI"], equal_var=False, nan_policy="omit").pvalue))
        if st is not None
        else "NA"
    )

    # Sex (Gender column: expected values Female/Male)
    sex_p = "NA"
    if "Gender" in meta.columns and st is not None:
        tab = pd.crosstab(meta["Group"], meta["Gender"])
        if set(tab.columns) >= {"Female", "Male"} and set(tab.index) >= {"A", "B"}:
            sex_p = fmt_p(
                float(
                    st.fisher_exact(
                        [
                            [int(tab.loc["A", "Female"]), int(tab.loc["A", "Male"])],
                            [int(tab.loc["B", "Female"]), int(tab.loc["B", "Male"])],
                        ]
                    )[1]
                )
            )

    edu_p = "NA"
    if "Education_cat" in meta.columns and st is not None:
        edu_tab = pd.crosstab(meta["Group"], meta["Education_cat"])
        try:
            edu_p = fmt_p(float(st.chi2_contingency(edu_tab)[1]))
        except Exception:  # pragma: no cover
            edu_p = "NA"

    def _count(group_df: pd.DataFrame, col: str, value: str) -> int:
        return int((group_df[col] == value).sum())

    # Build rows (p shown on the first row of each block)
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "Characteristic": "Age, years",
            f"Group A (n={nA})": fmt_mean_sd(A["Age"]),
            f"Group B (n={nB})": fmt_mean_sd(B["Age"]),
            f"Total (n={nT})": fmt_mean_sd(meta["Age"]),
            "p value": p_age,
        }
    )
    rows.append(
        {
            "Characteristic": "Age range",
            f"Group A (n={nA})": f"{int(pd.to_numeric(A['Age'], errors='coerce').min())}–{int(pd.to_numeric(A['Age'], errors='coerce').max())}",
            f"Group B (n={nB})": f"{int(pd.to_numeric(B['Age'], errors='coerce').min())}–{int(pd.to_numeric(B['Age'], errors='coerce').max())}",
            f"Total (n={nT})": f"{int(pd.to_numeric(meta['Age'], errors='coerce').min())}–{int(pd.to_numeric(meta['Age'], errors='coerce').max())}",
            "p value": "",
        }
    )

    if "Gender" in meta.columns:
        rows.append(
            {
                "Characteristic": "Sex (Female)",
                f"Group A (n={nA})": fmt_n_pct(_count(A, "Gender", "Female"), nA),
                f"Group B (n={nB})": fmt_n_pct(_count(B, "Gender", "Female"), nB),
                f"Total (n={nT})": fmt_n_pct(_count(meta, "Gender", "Female"), nT),
                "p value": sex_p,
            }
        )
        rows.append(
            {
                "Characteristic": "Sex (Male)",
                f"Group A (n={nA})": fmt_n_pct(_count(A, "Gender", "Male"), nA),
                f"Group B (n={nB})": fmt_n_pct(_count(B, "Gender", "Male"), nB),
                f"Total (n={nT})": fmt_n_pct(_count(meta, "Gender", "Male"), nT),
                "p value": "",
            }
        )

    if "Education_cat" in meta.columns:
        rows.append(
            {
                "Characteristic": "Education",
                f"Group A (n={nA})": "",
                f"Group B (n={nB})": "",
                f"Total (n={nT})": "",
                "p value": edu_p,
            }
        )
        for cat in ["School", "Apprenticeship", "University degree", "Missing/Other"]:
            if cat not in set(meta["Education_cat"].unique()):
                continue
            rows.append(
                {
                    "Characteristic": f"  – {cat}",
                    f"Group A (n={nA})": fmt_n_pct(_count(A, "Education_cat", cat), nA),
                    f"Group B (n={nB})": fmt_n_pct(_count(B, "Education_cat", cat), nB),
                    f"Total (n={nT})": fmt_n_pct(_count(meta, "Education_cat", cat), nT),
                    "p value": "",
                }
            )

    rows.append(
        {
            "Characteristic": "Body mass index, kg/m²",
            f"Group A (n={nA})": fmt_mean_sd(A["BMI"]),
            f"Group B (n={nB})": fmt_mean_sd(B["BMI"]),
            f"Total (n={nT})": fmt_mean_sd(meta["BMI"]),
            "p value": p_bmi,
        }
    )

    return pd.DataFrame(rows)


def build_table2_primary():
    g = pd.read_csv(MODELS_DIR / "glm_group_effects.csv")
    rows = []
    for outcome in sorted(g["outcome"].unique()):
        sub = g[g["outcome"] == outcome].copy()
        main = sub[sub["term"] == "C(Group)[T.B]"]
        inter = sub[sub["term"] == "C(Group)[T.B]:C(day)[T.2]"]
        m = main.iloc[0] if len(main) else None
        i = inter.iloc[0] if len(inter) else None
        rows.append(
            {
                "Outcome": outcome,
                "Group effect B vs A (coef)": np.nan if m is None else float(m["Coef."]),
                "Group effect 95% CI": "NA"
                if m is None
                else f"[{fmt_num(float(m['[0.025']))}, {fmt_num(float(m['0.975]']))}]",
                "Group effect p": "NA" if m is None else fmt_p(float(m["P>|z|"])),
                "Group×Day interaction (coef)": np.nan if i is None else float(i["Coef."]),
                "Group×Day 95% CI": "NA"
                if i is None
                else f"[{fmt_num(float(i['[0.025']))}, {fmt_num(float(i['0.975]']))}]",
                "Group×Day p": "NA" if i is None else fmt_p(float(i["P>|z|"])),
            }
        )
    t2 = pd.DataFrame(rows)
    return t2


def build_table3_mixed_simple_effects():
    c = pd.read_csv(ADD_DIR / "planned_contrasts_simple_effects.csv")
    keep = c[c["contrast"].isin(["structured_minus_random", "day2_minus_day1", "B_minus_A"])].copy()
    keep["Effect"] = keep["contrast"]
    keep["Group"] = keep["Group"].fillna("-")
    keep["day"] = keep["day"].fillna("-")
    keep["condition"] = keep["condition"].fillna("-")
    keep["ratio_fmt"] = keep.apply(
        lambda r: f"{fmt_num(r['ratio'])} [{fmt_num(np.exp(r['ci_low']))}, {fmt_num(np.exp(r['ci_high']))}]",
        axis=1,
    )
    keep["pct_fmt"] = keep.apply(
        lambda r: f"{fmt_num(100*r['pct_change'],2)}% [{fmt_num(100*r['pct_ci_low'],2)}%, {fmt_num(100*r['pct_ci_high'],2)}%]",
        axis=1,
    )
    out = keep[
        ["Effect", "Group", "day", "condition", "log_diff", "se", "ratio_fmt", "pct_fmt"]
    ].rename(columns={"day": "Day", "condition": "Condition", "log_diff": "log-diff", "se": "SE"})
    return out


def build_table_sensitivity():
    s = pd.read_csv(ADD_DIR / "missingness_sensitivity.csv")
    s["Seq Group p"] = s["seq_group_p"].map(fmt_p)
    s["Retention Group p"] = s["ret_group_p"].map(fmt_p)
    return s[
        [
            "scenario",
            "n_participant_day",
            "n_retention",
            "long_session_excluded",
            "seq_group_coef",
            "Seq Group p",
            "ret_group_coef",
            "Retention Group p",
        ]
    ]


def build_table_permutation():
    p = pd.read_csv(ADD_DIR / "permutation_tests.csv")
    p["Permutation p (two-sided)"] = p["perm_p_two_sided"].map(fmt_p)
    return p[["outcome", "observed_B_minus_A", "Permutation p (two-sided)", "n_perm"]]


def build_table_stratification():
    st = pd.read_csv(ADD_DIR / "clinical_stratification_group_effects.csv")
    st["B vs A (95% CI)"] = st.apply(
        lambda r: f"{fmt_num(r['coef_B_vs_A'])} [{fmt_num(r['ci_low'])}, {fmt_num(r['ci_high'])}]",
        axis=1,
    )
    st["p"] = st["p_value"].map(fmt_p)
    return st[["stratum", "n", "B vs A (95% CI)", "p"]]


def build_table_time_trend_sx():
    lin = pd.read_csv(ADD_DIR / "supp_time_trend_contrasts_linear.csv")
    flex = pd.read_csv(ADD_DIR / "supp_time_trend_contrasts_flexible.csv")
    dod_lin = pd.read_csv(ADD_DIR / "supp_time_trend_DoD_linear.csv")
    dod_flex = pd.read_csv(ADD_DIR / "supp_time_trend_DoD_flexible.csv")
    fit = pd.read_csv(ADD_DIR / "supp_time_trend_fit_comparison.csv")

    # Focus on primary planned contrasts.
    keep_cols = ["contrast", "Group", "Day", "Condition", "log_diff", "SE", "p", "pct_change"]
    lin = lin[keep_cols].copy()
    flex = flex[keep_cols].copy()
    merged = lin.merge(
        flex,
        on=["contrast", "Group", "Day", "Condition"],
        suffixes=("_linear", "_flex"),
    )
    merged["delta_pct_points"] = 100 * (merged["pct_change_flex"] - merged["pct_change_linear"])
    merged["materially_unchanged_2pp"] = merged["delta_pct_points"].abs() < 2.0

    dod = pd.DataFrame(
        [
            {
                "model": "linear",
                "log_DoD": float(dod_lin.loc[0, "log_DoD"]),
                "SE": float(dod_lin.loc[0, "SE"]),
                "p": float(dod_lin.loc[0, "p"]),
                "pct_change": float(dod_lin.loc[0, "pct_change"]),
            },
            {
                "model": "flexible_bs_df4",
                "log_DoD": float(dod_flex.loc[0, "log_DoD"]),
                "SE": float(dod_flex.loc[0, "SE"]),
                "p": float(dod_flex.loc[0, "p"]),
                "pct_change": float(dod_flex.loc[0, "pct_change"]),
            },
        ]
    )
    return merged, dod, fit


def write_markdown_tables(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    sens: pd.DataFrame,
    perm: pd.DataFrame,
    strat: pd.DataFrame,
    sx_contrasts: pd.DataFrame,
    sx_dod: pd.DataFrame,
    sx_fit: pd.DataFrame,
):
    md_path = ROOT / "analysis" / "tables.md"
    lines = []
    if len(table1):
        lines.append("## Table 1. Baseline demographic characteristics (included sample)\n")
        lines.append(table1.to_markdown(index=False))
        lines.append("")
    lines.append("## Table 2. Primary group effects (adjusted GLM, HC3)\n")
    lines.append(table2.to_markdown(index=False))
    lines.append("\n\n## Table 3. Mixed-model simple effects (ratio scale)\n")
    lines.append(table3.to_markdown(index=False))
    lines.append("\n\n## Supplementary Table S1. Missingness / data-quality sensitivity\n")
    lines.append(sens.to_markdown(index=False))
    lines.append("\n\n## Supplementary Table S2. Permutation tests\n")
    lines.append(perm.to_markdown(index=False))
    lines.append("\n\n## Supplementary Table S3. Clinical stratification\n")
    lines.append(strat.to_markdown(index=False))
    lines.append("\n\n## Supplementary Table S4. Time-trend robustness (linear vs flexible)\n")
    lines.append("Planned contrasts (linear vs flexible block-wise trend), including material change criterion (|delta| < 2 percentage points).\n")
    lines.append(sx_contrasts.to_markdown(index=False))
    lines.append("\n\n### Supplementary Table S4b. Difference-in-Differences (DoD) comparison\n")
    lines.append(sx_dod.to_markdown(index=False))
    lines.append("\n\n### Supplementary Table S4c. Model fit comparison (secondary)\n")
    lines.append(sx_fit.to_markdown(index=False))
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table1 = build_table1_demographics()
    table2 = build_table2_primary()
    table3 = build_table3_mixed_simple_effects()
    sens = build_table_sensitivity()
    perm = build_table_permutation()
    strat = build_table_stratification()
    sx_contrasts, sx_dod, sx_fit = build_table_time_trend_sx()

    if len(table1):
        table1.to_csv(OUT_DIR / "table1_demographics.csv", index=False)
    table2.to_csv(OUT_DIR / "table2_primary_glm.csv", index=False)
    table3.to_csv(OUT_DIR / "table3_mixed_simple_effects.csv", index=False)
    sens.to_csv(OUT_DIR / "supp_table_s1_sensitivity.csv", index=False)
    perm.to_csv(OUT_DIR / "supp_table_s2_permutation.csv", index=False)
    strat.to_csv(OUT_DIR / "supp_table_s3_stratification.csv", index=False)
    sx_contrasts.to_csv(OUT_DIR / "supp_table_s4_time_trend_contrasts.csv", index=False)
    sx_dod.to_csv(OUT_DIR / "supp_table_s4_time_trend_DoD.csv", index=False)
    sx_fit.to_csv(OUT_DIR / "supp_table_s4_time_trend_fit.csv", index=False)
    write_markdown_tables(table1, table2, table3, sens, perm, strat, sx_contrasts, sx_dod, sx_fit)

    print("Wrote publication tables to:")
    print(OUT_DIR)
    print(ROOT / "analysis" / "tables.md")


if __name__ == "__main__":
    main()

