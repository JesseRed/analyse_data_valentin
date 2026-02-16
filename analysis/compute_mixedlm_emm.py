#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
BLOCK_PATH = ROOT / "analysis" / "outputs" / "02_metrics" / "block_level_metrics.csv"
META_PATH = ROOT / "Datensatz_Round_2.csv"
OUT_DIR = ROOT / "analysis" / "outputs" / "03_models"


FORMULA = (
    "logRT ~ C(Group) * C(day) * C(condition) + BlockNumber"
    " + Age + fuglmayrshort_sum + MoCa_sum"
)


def normalize_meta() -> pd.DataFrame:
    meta = pd.read_csv(META_PATH, sep=";", na_values=["", "null", "N.A.", "NA"], low_memory=False)
    meta["PID"] = pd.to_numeric(meta["PID"], errors="coerce")
    meta = meta.dropna(subset=["PID"]).copy()
    meta["PID"] = meta["PID"].astype(int)

    # Align with pipeline conventions.
    if "Age" in meta.columns:
        meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    else:
        meta["Age"] = pd.to_numeric(meta.get("Biomag_Untersuchung"), errors="coerce")

    for col in ["fuglmayrshort_sum", "MoCa_sum"]:
        meta[col] = pd.to_numeric(meta.get(col), errors="coerce")

    keep = ["PID", "Group", "Age", "fuglmayrshort_sum", "MoCa_sum"]
    return meta[keep].copy()


def load_block_cov() -> pd.DataFrame:
    block = pd.read_csv(BLOCK_PATH)
    meta = normalize_meta()
    d = block.merge(meta, on=["PID", "Group"], how="left")
    d = d[d["meanRT_hit_ms"].notna()].copy()
    d["logRT"] = np.log(d["meanRT_hit_ms"])
    d["day"] = d["day"].astype(int)
    d["condition"] = d["condition"].astype(str)
    d["Group"] = d["Group"].astype(str)

    # Match pipeline mixed-model filtering.
    d = d.dropna(
        subset=[
            "PID",
            "Group",
            "condition",
            "day",
            "BlockNumber",
            "Age",
            "fuglmayrshort_sum",
            "MoCa_sum",
        ]
    )
    return d


def fit_mixedlm(d: pd.DataFrame):
    model = smf.mixedlm(FORMULA, data=d, groups=d["PID"], re_formula="1")
    fit = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
    return fit


def build_emm_grid(d: pd.DataFrame) -> pd.DataFrame:
    # Use means of covariates and a representative mid-task block.
    cov_means = {
        "Age": float(d["Age"].mean()),
        "fuglmayrshort_sum": float(d["fuglmayrshort_sum"].mean()),
        "MoCa_sum": float(d["MoCa_sum"].mean()),
    }
    block_number = float(d["BlockNumber"].mean())

    grid = []
    for group in ["A", "B"]:
        for day in [1, 2]:
            for condition in ["random", "structured"]:
                grid.append(
                    {
                        "Group": group,
                        "day": day,
                        "condition": condition,
                        "BlockNumber": block_number,
                        **cov_means,
                    }
                )
    return pd.DataFrame(grid)


def fe_cov(fit) -> pd.DataFrame:
    cov = fit.cov_params()
    fe_names = list(fit.fe_params.index)
    if isinstance(cov, pd.DataFrame):
        return cov.loc[fe_names, fe_names]
    return pd.DataFrame(cov[: len(fe_names), : len(fe_names)], index=fe_names, columns=fe_names)


def design_matrix(fit, new_data: pd.DataFrame) -> pd.DataFrame:
    info = fit.model.data.design_info
    X = patsy.build_design_matrices([info], new_data, return_type="dataframe")[0]
    # Ensure column order matches fixed effect params.
    X = X.loc[:, fit.fe_params.index]
    return X


def emm_table(fit, d: pd.DataFrame) -> pd.DataFrame:
    grid = build_emm_grid(d)
    X = design_matrix(fit, grid)
    beta = fit.fe_params
    cov = fe_cov(fit)

    mu = X.to_numpy() @ beta.to_numpy()
    se = np.sqrt(np.einsum("ij,jk,ik->i", X.to_numpy(), cov.to_numpy(), X.to_numpy()))

    out = grid.copy()
    out["logRT_hat"] = mu
    out["logRT_se"] = se
    out["logRT_ci_low"] = mu - 1.96 * se
    out["logRT_ci_high"] = mu + 1.96 * se
    out["RT_hat_ms"] = np.exp(out["logRT_hat"])
    out["RT_ci_low_ms"] = np.exp(out["logRT_ci_low"])
    out["RT_ci_high_ms"] = np.exp(out["logRT_ci_high"])
    return out


def contrast(fit, row_a: pd.Series, row_b: pd.Series) -> tuple[float, float]:
    X = design_matrix(fit, pd.DataFrame([row_a.to_dict(), row_b.to_dict()]))
    L = (X.iloc[0] - X.iloc[1]).to_numpy()
    beta = fit.fe_params.to_numpy()
    cov = fe_cov(fit).to_numpy()
    est = float(L @ beta)
    se = float(np.sqrt(L @ cov @ L))
    return est, se


def build_contrasts(fit, emm: pd.DataFrame) -> pd.DataFrame:
    # Helper to get a single row.
    def get(group: str, day: int, condition: str) -> pd.Series:
        row = emm[(emm["Group"] == group) & (emm["day"] == day) & (emm["condition"] == condition)]
        if len(row) != 1:
            raise RuntimeError(f"Expected single EMM row for {group},{day},{condition}; got {len(row)}")
        return row.iloc[0]

    contrasts = []

    # 1) structured - random within each group/day (log difference => ratio).
    for group in ["A", "B"]:
        for day in [1, 2]:
            a = get(group, day, "structured")
            b = get(group, day, "random")
            est, se = contrast(fit, a, b)
            contrasts.append(
                {
                    "contrast": "structured_minus_random",
                    "Group": group,
                    "day": day,
                    "condition": None,
                    "log_diff": est,
                    "se": se,
                }
            )

    # 2) day2 - day1 within each group/condition.
    for group in ["A", "B"]:
        for condition in ["random", "structured"]:
            a = get(group, 2, condition)
            b = get(group, 1, condition)
            est, se = contrast(fit, a, b)
            contrasts.append(
                {
                    "contrast": "day2_minus_day1",
                    "Group": group,
                    "day": None,
                    "condition": condition,
                    "log_diff": est,
                    "se": se,
                }
            )

    # 3) groupB - groupA within each day/condition.
    for day in [1, 2]:
        for condition in ["random", "structured"]:
            a = get("B", day, condition)
            b = get("A", day, condition)
            est, se = contrast(fit, a, b)
            contrasts.append(
                {
                    "contrast": "B_minus_A",
                    "Group": None,
                    "day": day,
                    "condition": condition,
                    "log_diff": est,
                    "se": se,
                }
            )

    # 4) Difference-in-differences: (structured-random) day2-day1 difference between groups.
    #    Equivalent to 3-way interaction on log scale.
    #    Compute: [ (S-R)_B_day2 - (S-R)_B_day1 ] - [ (S-R)_A_day2 - (S-R)_A_day1 ].
    s_r = {}
    for group in ["A", "B"]:
        for day in [1, 2]:
            s = get(group, day, "structured")
            r = get(group, day, "random")
            s_r[(group, day)] = (s, r)

    # Build as a single linear contrast using the four rows.
    # L = (B2S - B2R) - (B1S - B1R) - (A2S - A2R) + (A1S - A1R)
    rows = [
        ("B2S", s_r[("B", 2)][0], +1),
        ("B2R", s_r[("B", 2)][1], -1),
        ("B1S", s_r[("B", 1)][0], -1),
        ("B1R", s_r[("B", 1)][1], +1),
        ("A2S", s_r[("A", 2)][0], -1),
        ("A2R", s_r[("A", 2)][1], +1),
        ("A1S", s_r[("A", 1)][0], +1),
        ("A1R", s_r[("A", 1)][1], -1),
    ]
    X = design_matrix(fit, pd.DataFrame([r.to_dict() for _, r, _ in rows]))
    L = np.zeros(X.shape[1])
    for i, (_, _, w) in enumerate(rows):
        L += w * X.iloc[i].to_numpy()
    beta = fit.fe_params.to_numpy()
    cov = fe_cov(fit).to_numpy()
    est = float(L @ beta)
    se = float(np.sqrt(L @ cov @ L))
    contrasts.append(
        {
            "contrast": "diff_in_diff_(structured-random)_change_day2-day1_between_groups",
            "Group": None,
            "day": None,
            "condition": None,
            "log_diff": est,
            "se": se,
        }
    )

    out = pd.DataFrame(contrasts)
    out["ci_low"] = out["log_diff"] - 1.96 * out["se"]
    out["ci_high"] = out["log_diff"] + 1.96 * out["se"]
    out["ratio"] = np.exp(out["log_diff"])
    out["ratio_ci_low"] = np.exp(out["ci_low"])
    out["ratio_ci_high"] = np.exp(out["ci_high"])
    out["pct_change"] = out["ratio"] - 1.0
    out["pct_ci_low"] = out["ratio_ci_low"] - 1.0
    out["pct_ci_high"] = out["ratio_ci_high"] - 1.0
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = load_block_cov()
    fit = fit_mixedlm(d)
    emm = emm_table(fit, d)
    con = build_contrasts(fit, emm)

    emm.to_csv(OUT_DIR / "mixedlm_marginal_means.csv", index=False)
    con.to_csv(OUT_DIR / "mixedlm_contrasts.csv", index=False)
    with open(OUT_DIR / "mixedlm_emm_note.txt", "w", encoding="utf-8") as f:
        f.write("EMMs computed from fixed effects at covariate means and BlockNumber mean.\n")
        f.write(f"Formula: {FORMULA}\n")
        f.write(str(fit.summary()))

    print("Wrote:", OUT_DIR / "mixedlm_marginal_means.csv")
    print("Wrote:", OUT_DIR / "mixedlm_contrasts.csv")


if __name__ == "__main__":
    main()

