#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy.stats import chi2, norm

try:
    from analysis.analysis_config import EXCLUDED_PIDS, PRIMARY_COVARS
except ModuleNotFoundError:
    from analysis_config import EXCLUDED_PIDS, PRIMARY_COVARS


ROOT = Path(__file__).resolve().parents[1]
BLOCK_PATH = ROOT / "analysis" / "outputs" / "02_metrics" / "block_level_metrics.csv"
META_PATH = ROOT / "Datensatz_Round_2.csv"
OUT_DIR = ROOT / "analysis" / "outputs" / "05_additional_analyses"

PRIMARY_FORMULA = (
    "logRT ~ C(Group) * C(day) * C(condition) + BlockNumber"
    " + " + " + ".join(PRIMARY_COVARS)
)
FLEX_FORMULA = (
    "logRT ~ C(Group) * C(day) * C(condition) + bs(BlockNumber, df=4)"
    " + " + " + ".join(PRIMARY_COVARS)
)


def load_data() -> pd.DataFrame:
    block = pd.read_csv(BLOCK_PATH)
    block = block.loc[~block["PID"].isin(EXCLUDED_PIDS)].copy()
    meta = pd.read_csv(META_PATH, sep=";", na_values=["", "null", "N.A.", "NA"], low_memory=False)
    meta["PID"] = pd.to_numeric(meta["PID"], errors="coerce")
    meta = meta.dropna(subset=["PID"]).copy()
    meta["PID"] = meta["PID"].astype(int)
    meta = meta.loc[~meta["PID"].isin(EXCLUDED_PIDS)].copy()
    if "Age" in meta.columns:
        meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    else:
        meta["Age"] = pd.to_numeric(meta.get("Biomag_Untersuchung"), errors="coerce")
    for col in PRIMARY_COVARS:
        if col == "Age":
            continue
        meta[col] = pd.to_numeric(meta.get(col), errors="coerce")
    meta = meta[["PID", "Group", *PRIMARY_COVARS]]

    d = block.merge(meta, on=["PID", "Group"], how="left")
    d = d[d["meanRT_hit_ms"].notna()].copy()
    d["logRT"] = np.log(d["meanRT_hit_ms"])
    d["day"] = d["day"].astype(int)
    d = d.dropna(
        subset=[
            "PID",
            "Group",
            "condition",
            "day",
            "BlockNumber",
            *PRIMARY_COVARS,
            "logRT",
        ]
    ).copy()
    return d


def fe_cov(fit):
    cov = fit.cov_params()
    fe_names = list(fit.fe_params.index)
    if isinstance(cov, pd.DataFrame):
        return cov.loc[fe_names, fe_names]
    return pd.DataFrame(cov[: len(fe_names), : len(fe_names)], index=fe_names, columns=fe_names)


def design_matrix(fit, new_data: pd.DataFrame):
    info = fit.model.data.design_info
    X = patsy.build_design_matrices([info], new_data, return_type="dataframe")[0]
    return X.loc[:, fit.fe_params.index]


def model_emm_grid(d: pd.DataFrame) -> pd.DataFrame:
    cov_means = {c: float(d[c].mean()) for c in PRIMARY_COVARS}
    block_mean = float(d["BlockNumber"].mean())
    rows = []
    for g in ["A", "B"]:
        for day in [1, 2]:
            for cond in ["random", "structured"]:
                rows.append(
                    {
                        "Group": g,
                        "day": day,
                        "condition": cond,
                        "BlockNumber": block_mean,
                        **cov_means,
                    }
                )
    return pd.DataFrame(rows)


def contrast_from_rows(fit, row_a: pd.Series, row_b: pd.Series):
    X = design_matrix(fit, pd.DataFrame([row_a.to_dict(), row_b.to_dict()]))
    L = (X.iloc[0] - X.iloc[1]).to_numpy()
    beta = fit.fe_params.to_numpy()
    cov = fe_cov(fit).to_numpy()
    est = float(L @ beta)
    se = float(np.sqrt(L @ cov @ L))
    z = est / se if se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return est, se, z, p


def get_cell(emm: pd.DataFrame, group: str, day: int, condition: str) -> pd.Series:
    row = emm[(emm["Group"] == group) & (emm["day"] == day) & (emm["condition"] == condition)]
    if len(row) != 1:
        raise RuntimeError(f"Expected one cell for {group}-{day}-{condition}, got {len(row)}")
    return row.iloc[0]


def compute_contrasts(fit, d: pd.DataFrame, model_name: str):
    emm = model_emm_grid(d)
    X = design_matrix(fit, emm)
    beta = fit.fe_params.to_numpy()
    cov = fe_cov(fit).to_numpy()

    mu = X.to_numpy() @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", X.to_numpy(), cov, X.to_numpy()))
    emm = emm.copy()
    emm["logRT_hat"] = mu
    emm["logRT_se"] = se
    emm["logRT_ci_low"] = mu - 1.96 * se
    emm["logRT_ci_high"] = mu + 1.96 * se
    emm["RT_hat_ms"] = np.exp(mu)
    emm["RT_ci_low_ms"] = np.exp(emm["logRT_ci_low"])
    emm["RT_ci_high_ms"] = np.exp(emm["logRT_ci_high"])

    rows = []
    # structured-random within group x day
    for g in ["A", "B"]:
        for day in [1, 2]:
            a = get_cell(emm, g, day, "structured")
            b = get_cell(emm, g, day, "random")
            est, se_, z, p = contrast_from_rows(fit, a, b)
            rows.append({"contrast": "structured_minus_random", "Group": g, "Day": day, "Condition": np.nan, "log_diff": est, "SE": se_, "z": z, "p": p})

    # day2-day1 within group x condition
    for g in ["A", "B"]:
        for c in ["random", "structured"]:
            a = get_cell(emm, g, 2, c)
            b = get_cell(emm, g, 1, c)
            est, se_, z, p = contrast_from_rows(fit, a, b)
            rows.append({"contrast": "day2_minus_day1", "Group": g, "Day": np.nan, "Condition": c, "log_diff": est, "SE": se_, "z": z, "p": p})

    # B-A within day x condition
    for day in [1, 2]:
        for c in ["random", "structured"]:
            a = get_cell(emm, "B", day, c)
            b = get_cell(emm, "A", day, c)
            est, se_, z, p = contrast_from_rows(fit, a, b)
            rows.append({"contrast": "B_minus_A", "Group": np.nan, "Day": day, "Condition": c, "log_diff": est, "SE": se_, "z": z, "p": p})

    con = pd.DataFrame(rows)
    con["ci_low"] = con["log_diff"] - 1.96 * con["SE"]
    con["ci_high"] = con["log_diff"] + 1.96 * con["SE"]
    con["ratio"] = np.exp(con["log_diff"])
    con["ratio_ci_low"] = np.exp(con["ci_low"])
    con["ratio_ci_high"] = np.exp(con["ci_high"])
    con["pct_change"] = con["ratio"] - 1.0
    con["pct_ci_low"] = con["ratio_ci_low"] - 1.0
    con["pct_ci_high"] = con["ratio_ci_high"] - 1.0
    con["model"] = model_name

    # DoD per requested definition:
    # DoD = [(S-R)A,d2 - (S-R)A,d1] - [(S-R)B,d2 - (S-R)B,d1]
    A_d2 = con[(con["contrast"] == "structured_minus_random") & (con["Group"] == "A") & (con["Day"] == 2)].iloc[0]
    A_d1 = con[(con["contrast"] == "structured_minus_random") & (con["Group"] == "A") & (con["Day"] == 1)].iloc[0]
    B_d2 = con[(con["contrast"] == "structured_minus_random") & (con["Group"] == "B") & (con["Day"] == 2)].iloc[0]
    B_d1 = con[(con["contrast"] == "structured_minus_random") & (con["Group"] == "B") & (con["Day"] == 1)].iloc[0]

    # Compute as one linear contrast for correct SE:
    cells = [
        (get_cell(emm, "A", 2, "structured"), +1),
        (get_cell(emm, "A", 2, "random"), -1),
        (get_cell(emm, "A", 1, "structured"), -1),
        (get_cell(emm, "A", 1, "random"), +1),
        (get_cell(emm, "B", 2, "structured"), -1),
        (get_cell(emm, "B", 2, "random"), +1),
        (get_cell(emm, "B", 1, "structured"), +1),
        (get_cell(emm, "B", 1, "random"), -1),
    ]
    Xd = design_matrix(fit, pd.DataFrame([r.to_dict() for r, _ in cells]))
    L = np.zeros(Xd.shape[1])
    for i, (_, w) in enumerate(cells):
        L += w * Xd.iloc[i].to_numpy()
    beta = fit.fe_params.to_numpy()
    cov = fe_cov(fit).to_numpy()
    dod_est = float(L @ beta)
    dod_se = float(np.sqrt(L @ cov @ L))
    dod_z = dod_est / dod_se if dod_se > 0 else np.nan
    dod_p = 2 * (1 - norm.cdf(abs(dod_z))) if np.isfinite(dod_z) else np.nan

    dod = pd.DataFrame(
        [
            {
                "model": model_name,
                "definition": "[(S-R)_A,d2-(S-R)_A,d1]-[(S-R)_B,d2-(S-R)_B,d1]",
                "log_DoD": dod_est,
                "SE": dod_se,
                "z": dod_z,
                "p": dod_p,
                "ci_low": dod_est - 1.96 * dod_se,
                "ci_high": dod_est + 1.96 * dod_se,
                "ratio": np.exp(dod_est),
                "ratio_ci_low": np.exp(dod_est - 1.96 * dod_se),
                "ratio_ci_high": np.exp(dod_est + 1.96 * dod_se),
                "pct_change": np.exp(dod_est) - 1.0,
                "pct_ci_low": np.exp(dod_est - 1.96 * dod_se) - 1.0,
                "pct_ci_high": np.exp(dod_est + 1.96 * dod_se) - 1.0,
                "A_day2_minus_day1_structured_advantage": float((A_d2["log_diff"] - A_d1["log_diff"])),
                "B_day2_minus_day1_structured_advantage": float((B_d2["log_diff"] - B_d1["log_diff"])),
            }
        ]
    )
    return emm, con, dod


def fit_and_export(d: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m_linear = smf.mixedlm(PRIMARY_FORMULA, data=d, groups=d["PID"], re_formula="1").fit(reml=False, method="lbfgs", maxiter=200, disp=False)
    m_flex = smf.mixedlm(FLEX_FORMULA, data=d, groups=d["PID"], re_formula="1").fit(reml=False, method="lbfgs", maxiter=200, disp=False)

    emm_lin, con_lin, dod_lin = compute_contrasts(m_linear, d, "linear")
    emm_flex, con_flex, dod_flex = compute_contrasts(m_flex, d, "flexible_bs_df4")

    con_lin.to_csv(OUT_DIR / "supp_time_trend_contrasts_linear.csv", index=False)
    con_flex.to_csv(OUT_DIR / "supp_time_trend_contrasts_flexible.csv", index=False)
    dod_lin.to_csv(OUT_DIR / "supp_time_trend_DoD_linear.csv", index=False)
    dod_flex.to_csv(OUT_DIR / "supp_time_trend_DoD_flexible.csv", index=False)

    llr = 2 * (m_flex.llf - m_linear.llf)
    df_diff = (m_flex.df_modelwc - m_linear.df_modelwc)
    p_llr = 1 - chi2.cdf(llr, max(int(df_diff), 1))
    fit_comp = pd.DataFrame(
        [
            {
                "model": "linear",
                "formula": PRIMARY_FORMULA,
                "AIC": m_linear.aic,
                "BIC": m_linear.bic,
                "logLik": m_linear.llf,
                "df_modelwc": m_linear.df_modelwc,
                "n_obs": m_linear.nobs,
            },
            {
                "model": "flexible_bs_df4",
                "formula": FLEX_FORMULA,
                "AIC": m_flex.aic,
                "BIC": m_flex.bic,
                "logLik": m_flex.llf,
                "df_modelwc": m_flex.df_modelwc,
                "n_obs": m_flex.nobs,
            },
            {
                "model": "comparison",
                "formula": "flexible_vs_linear",
                "AIC": np.nan,
                "BIC": np.nan,
                "logLik": np.nan,
                "df_modelwc": df_diff,
                "n_obs": m_linear.nobs,
                "LLR": llr,
                "LLR_p": p_llr,
            },
        ]
    )
    fit_comp.to_csv(OUT_DIR / "supp_time_trend_fit_comparison.csv", index=False)

    # Stability summary (materially unchanged criterion)
    s_lin = con_lin[con_lin["contrast"] == "structured_minus_random"].copy()
    s_flex = con_flex[con_flex["contrast"] == "structured_minus_random"].copy()
    merged = s_lin.merge(s_flex, on=["contrast", "Group", "Day", "Condition"], suffixes=("_linear", "_flex"))
    merged["delta_pct_points"] = 100 * (merged["pct_change_flex"] - merged["pct_change_linear"])
    merged["materially_unchanged_2pp"] = merged["delta_pct_points"].abs() < 2.0

    dod_compare = dod_lin[["log_DoD", "SE", "p", "pct_change"]].add_suffix("_linear").join(
        dod_flex[["log_DoD", "SE", "p", "pct_change"]].add_suffix("_flex")
    )
    dod_compare["DoD_same_direction"] = np.sign(dod_compare["log_DoD_linear"]) == np.sign(dod_compare["log_DoD_flex"])
    dod_compare["DoD_both_p_lt_0_05"] = (dod_compare["p_linear"] < 0.05) & (dod_compare["p_flex"] < 0.05)

    merged.to_csv(OUT_DIR / "supp_time_trend_stability_structured_advantage.csv", index=False)
    dod_compare.to_csv(OUT_DIR / "supp_time_trend_stability_DoD.csv", index=False)


def main():
    d = load_data()
    fit_and_export(d)
    print("Time-trend robustness outputs written.")


if __name__ == "__main__":
    main()

