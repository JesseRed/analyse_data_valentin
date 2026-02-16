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


def write_markdown_tables(
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    sens: pd.DataFrame,
    perm: pd.DataFrame,
    strat: pd.DataFrame,
):
    md_path = ROOT / "analysis" / "tables.md"
    lines = []
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
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table2 = build_table2_primary()
    table3 = build_table3_mixed_simple_effects()
    sens = build_table_sensitivity()
    perm = build_table_permutation()
    strat = build_table_stratification()

    table2.to_csv(OUT_DIR / "table2_primary_glm.csv", index=False)
    table3.to_csv(OUT_DIR / "table3_mixed_simple_effects.csv", index=False)
    sens.to_csv(OUT_DIR / "supp_table_s1_sensitivity.csv", index=False)
    perm.to_csv(OUT_DIR / "supp_table_s2_permutation.csv", index=False)
    strat.to_csv(OUT_DIR / "supp_table_s3_stratification.csv", index=False)
    write_markdown_tables(table2, table3, sens, perm, strat)

    print("Wrote publication tables to:")
    print(OUT_DIR)
    print(ROOT / "analysis" / "tables.md")


if __name__ == "__main__":
    main()

