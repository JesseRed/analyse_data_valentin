#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from scipy.stats import chi2
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.run_srtt_analysis import (
    aggregate_metrics,
    load_metadata,
    merge_covariates,
    run_glm_models,
)
OUT_DIR = ROOT / "analysis" / "outputs" / "05_additional_analyses"
MODELS_DIR = ROOT / "analysis" / "outputs" / "03_models"
QC_DIR = ROOT / "analysis" / "outputs" / "01_ingest_and_qc"
METRICS_DIR = ROOT / "analysis" / "outputs" / "02_metrics"


def ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_core():
    trials = pd.read_csv(QC_DIR / "trial_level_all.csv")
    block = pd.read_csv(METRICS_DIR / "block_level_metrics.csv")
    pday = pd.read_csv(METRICS_DIR / "participant_day_metrics.csv")
    pret = pd.read_csv(METRICS_DIR / "participant_retention_metrics.csv")
    return trials, block, pday, pret


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


def emm_and_contrasts_from_mixed(d: pd.DataFrame):
    formula = (
        "logRT ~ C(Group) * C(day) * C(condition) + BlockNumber"
        " + Age + AES_sum + MoCa_sum + NIHSS + TSS"
    )
    model = smf.mixedlm(formula, data=d, groups=d["PID"], re_formula="1")
    fit = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)

    cov_means = {
        "Age": float(d["Age"].mean()),
        "AES_sum": float(d["AES_sum"].mean()),
        "MoCa_sum": float(d["MoCa_sum"].mean()),
        "NIHSS": float(d["NIHSS"].mean()),
        "TSS": float(d["TSS"].mean()),
    }
    block_number = float(d["BlockNumber"].mean())
    grid = []
    for group in ["A", "B"]:
        for day in [1, 2]:
            for condition in ["random", "structured"]:
                grid.append(
                    {"Group": group, "day": day, "condition": condition, "BlockNumber": block_number, **cov_means}
                )
    grid = pd.DataFrame(grid)
    X = design_matrix(fit, grid)
    beta = fit.fe_params
    cov = fe_cov(fit)
    mu = X.to_numpy() @ beta.to_numpy()
    se = np.sqrt(np.einsum("ij,jk,ik->i", X.to_numpy(), cov.to_numpy(), X.to_numpy()))
    emm = grid.copy()
    emm["logRT_hat"] = mu
    emm["RT_hat_ms"] = np.exp(mu)
    emm["logRT_se"] = se
    emm["logRT_ci_low"] = mu - 1.96 * se
    emm["logRT_ci_high"] = mu + 1.96 * se
    emm["RT_ci_low_ms"] = np.exp(emm["logRT_ci_low"])
    emm["RT_ci_high_ms"] = np.exp(emm["logRT_ci_high"])

    def get(group, day, condition):
        row = emm[(emm["Group"] == group) & (emm["day"] == day) & (emm["condition"] == condition)]
        return row.iloc[0]

    def contrast(a, b):
        X2 = design_matrix(fit, pd.DataFrame([a.to_dict(), b.to_dict()]))
        L = (X2.iloc[0] - X2.iloc[1]).to_numpy()
        est = float(L @ fit.fe_params.to_numpy())
        se_ = float(np.sqrt(L @ fe_cov(fit).to_numpy() @ L))
        return est, se_

    rows = []
    for g in ["A", "B"]:
        for day in [1, 2]:
            est, se_ = contrast(get(g, day, "structured"), get(g, day, "random"))
            rows.append({"contrast": "structured_minus_random", "Group": g, "day": day, "condition": np.nan, "log_diff": est, "se": se_})
    for g in ["A", "B"]:
        for c in ["random", "structured"]:
            est, se_ = contrast(get(g, 2, c), get(g, 1, c))
            rows.append({"contrast": "day2_minus_day1", "Group": g, "day": np.nan, "condition": c, "log_diff": est, "se": se_})
    for day in [1, 2]:
        for c in ["random", "structured"]:
            est, se_ = contrast(get("B", day, c), get("A", day, c))
            rows.append({"contrast": "B_minus_A", "Group": np.nan, "day": day, "condition": c, "log_diff": est, "se": se_})
    con = pd.DataFrame(rows)
    con["ci_low"] = con["log_diff"] - 1.96 * con["se"]
    con["ci_high"] = con["log_diff"] + 1.96 * con["se"]
    con["ratio"] = np.exp(con["log_diff"])
    con["pct_change"] = con["ratio"] - 1.0
    con["pct_ci_low"] = np.exp(con["ci_low"]) - 1.0
    con["pct_ci_high"] = np.exp(con["ci_high"]) - 1.0
    return fit, emm, con


def nonlinear_learning_curves(d: pd.DataFrame):
    d = d.copy()
    d["logRT"] = np.log(d["meanRT_hit_ms"])
    d = d.dropna(subset=["logRT", "BlockNumber", "Group", "day", "condition", "Age", "AES_sum", "MoCa_sum", "NIHSS", "TSS"])

    linear_formula = (
        "logRT ~ C(Group) * C(day) * C(condition) + BlockNumber"
        " + Age + AES_sum + MoCa_sum + NIHSS + TSS"
    )
    spline_formula = (
        "logRT ~ C(Group) * C(day) * C(condition)"
        " + bs(BlockNumber, df=4)"
        " + C(condition):bs(BlockNumber, df=4)"
        " + Age + AES_sum + MoCa_sum + NIHSS + TSS"
    )
    m_lin = smf.ols(linear_formula, data=d).fit()
    m_spl = smf.ols(spline_formula, data=d).fit()

    # Approximate LR-like test on OLS likelihoods.
    llr = 2 * (m_spl.llf - m_lin.llf)
    df_diff = int(m_spl.df_model - m_lin.df_model)
    p_llr = 1 - chi2.cdf(llr, df=max(df_diff, 1))

    model_cmp = pd.DataFrame(
        [
            {
                "model": "linear",
                "AIC": m_lin.aic,
                "BIC": m_lin.bic,
                "R2": m_lin.rsquared,
                "adj_R2": m_lin.rsquared_adj,
                "llf": m_lin.llf,
                "df_model": m_lin.df_model,
            },
            {
                "model": "spline_df4",
                "AIC": m_spl.aic,
                "BIC": m_spl.bic,
                "R2": m_spl.rsquared,
                "adj_R2": m_spl.rsquared_adj,
                "llf": m_spl.llf,
                "df_model": m_spl.df_model,
            },
        ]
    )
    test_row = pd.DataFrame([{"llr": llr, "df_diff": df_diff, "p_value": p_llr}])
    return model_cmp, test_row, m_lin, m_spl


def blue_green_separate(d: pd.DataFrame):
    d = d.copy()
    d["logRT"] = np.log(d["meanRT_hit_ms"])
    d = d[d["sequence"].isin(["blue", "green", "yellow"])].dropna(
        subset=["logRT", "Group", "day", "sequence", "BlockNumber", "Age", "AES_sum", "MoCa_sum", "NIHSS", "TSS"]
    )
    formula = (
        "logRT ~ C(Group) * C(day) * C(sequence) + BlockNumber"
        " + Age + AES_sum + MoCa_sum + NIHSS + TSS"
    )
    model = smf.mixedlm(formula, data=d, groups=d["PID"], re_formula="1")
    fit = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)

    # EMM-style grid.
    cov_means = {
        "Age": float(d["Age"].mean()),
        "AES_sum": float(d["AES_sum"].mean()),
        "MoCa_sum": float(d["MoCa_sum"].mean()),
        "NIHSS": float(d["NIHSS"].mean()),
        "TSS": float(d["TSS"].mean()),
        "BlockNumber": float(d["BlockNumber"].mean()),
    }
    grid = []
    for g in ["A", "B"]:
        for day in [1, 2]:
            for seq in ["yellow", "blue", "green"]:
                grid.append({"Group": g, "day": day, "sequence": seq, **cov_means})
    grid = pd.DataFrame(grid)
    X = design_matrix(fit, grid)
    mu = X.to_numpy() @ fit.fe_params.to_numpy()
    se = np.sqrt(np.einsum("ij,jk,ik->i", X.to_numpy(), fe_cov(fit).to_numpy(), X.to_numpy()))
    emm = grid.copy()
    emm["logRT_hat"] = mu
    emm["RT_hat_ms"] = np.exp(mu)
    emm["se"] = se

    def get(group, day, sequence):
        return emm[(emm["Group"] == group) & (emm["day"] == day) & (emm["sequence"] == sequence)].iloc[0]

    # blue-yellow and green-yellow within each group/day
    rows = []
    for g in ["A", "B"]:
        for day in [1, 2]:
            for seq in ["blue", "green"]:
                a = get(g, day, seq)
                b = get(g, day, "yellow")
                X2 = design_matrix(fit, pd.DataFrame([a.to_dict(), b.to_dict()]))
                L = (X2.iloc[0] - X2.iloc[1]).to_numpy()
                est = float(L @ fit.fe_params.to_numpy())
                se_ = float(np.sqrt(L @ fe_cov(fit).to_numpy() @ L))
                rows.append({"contrast": f"{seq}_minus_yellow", "Group": g, "day": day, "log_diff": est, "se": se_})
    con = pd.DataFrame(rows)
    con["ci_low"] = con["log_diff"] - 1.96 * con["se"]
    con["ci_high"] = con["log_diff"] + 1.96 * con["se"]
    con["ratio"] = np.exp(con["log_diff"])
    con["pct_change"] = con["ratio"] - 1.0
    return fit, emm, con


def speed_accuracy_tradeoff(block_cov: pd.DataFrame, trials: pd.DataFrame):
    # Block-level RT model with error-rate moderation.
    b = block_cov.copy()
    b["logRT"] = np.log(b["meanRT_hit_ms"])
    b = b.dropna(subset=["logRT", "errorRate", "Group", "day", "condition", "BlockNumber", "Age", "AES_sum", "MoCa_sum", "NIHSS", "TSS"])
    m1 = smf.ols(
        "logRT ~ C(Group)*C(day)*C(condition) + errorRate + C(Group):errorRate + BlockNumber + Age + AES_sum + MoCa_sum + NIHSS + TSS",
        data=b,
    ).fit(cov_type="HC3")

    # Trial-level logistic model for hit probability with zRT predictor.
    t = trials.copy()
    t = t[(t["RT_ms"] > 0) & (t["RT_ms"] < 5000)].copy()
    t["zRT"] = t.groupby("PID")["RT_ms"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
    t["zRT"] = t["zRT"].clip(-3, 3)
    t = t.dropna(subset=["isHit", "zRT", "Group", "day", "condition"])
    glm_hit = smf.glm(
        "isHit ~ zRT + C(Group) + C(day) + C(condition) + C(Group):zRT + C(day):zRT + C(condition):zRT",
        data=t,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": t["PID"]})
    return m1, glm_hit


def robust_penalized_models(pday: pd.DataFrame):
    d = pday[pday["day"] == 2].copy()
    outcome = "SeqLearning_Index_all"
    predictors = [
        "C(Group)",
        "AES_sum",
        "Age",
        "Gender_num",
        "Depression_num",
        "SportsActivity_num",
        "fuglmayrshort_sum",
        "EQ5D_health_status",
        "GDS_sum",
        "MoCa_sum",
        "MORE_sum",
        "TSS",
        "NIHSS",
    ]
    formula = outcome + " ~ " + " + ".join(predictors)
    y, X = dmatrices(formula, d, return_type="dataframe")
    y = np.asarray(y).ravel()

    # Robust linear model (Huber).
    rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()

    # Penalized models.
    ridge = Pipeline(
        [("scaler", StandardScaler(with_mean=False)), ("model", RidgeCV(alphas=np.logspace(-3, 3, 31), cv=5))]
    )
    ridge.fit(X, y)
    ridge_coef = pd.Series(ridge.named_steps["model"].coef_, index=X.columns)

    lasso = Pipeline(
        [("scaler", StandardScaler(with_mean=False)), ("model", LassoCV(cv=5, random_state=42, max_iter=20000))]
    )
    lasso.fit(X, y)
    lasso_coef = pd.Series(lasso.named_steps["model"].coef_, index=X.columns)

    robust_tbl = pd.DataFrame({"term": X.columns, "coef_rlm": rlm.params})
    penal_tbl = pd.DataFrame({"term": X.columns, "coef_ridge": ridge_coef, "coef_lasso": lasso_coef})
    return robust_tbl, penal_tbl, rlm


def sensitivity_missingness(trials: pd.DataFrame):
    meta = load_metadata()
    file_qc = pd.read_csv(QC_DIR / "file_qc_flags.csv")
    long_sess = file_qc.sort_values("n_blocks", ascending=False).head(1)[["PID", "day"]]
    long_pid = int(long_sess.iloc[0]["PID"])
    long_day = int(long_sess.iloc[0]["day"])

    scenarios = {
        "baseline": trials.copy(),
        "truncate_119": trials[trials["BlockNumber"] <= 119].copy(),
        "truncate_120": trials[trials["BlockNumber"] <= 120].copy(),
        "exclude_long_session": trials[~((trials["PID"] == long_pid) & (trials["day"] == long_day))].copy(),
    }
    rows = []
    for name, t in scenarios.items():
        block, pday, ret = aggregate_metrics(t)
        pday_cov = merge_covariates(pday, meta)
        ret_cov = merge_covariates(ret, meta)
        glm_index, glm_models = run_glm_models(pday_cov, ret_cov)
        # pull target outcomes only
        target = {"SeqLearning_Index_all": None, "Retention_Sequence": None}
        for m in glm_models:
            if m.outcome in target:
                coef = m.coef_table
                r = coef[coef["term"] == "C(Group)[T.B]"]
                if len(r):
                    target[m.outcome] = r.iloc[0]
        rows.append(
            {
                "scenario": name,
                "n_participant_day": len(pday_cov),
                "n_retention": len(ret_cov),
                "long_session_excluded": f"{long_pid}_day{long_day}" if name == "exclude_long_session" else "",
                "seq_group_coef": np.nan if target["SeqLearning_Index_all"] is None else float(target["SeqLearning_Index_all"]["Coef."]),
                "seq_group_p": np.nan if target["SeqLearning_Index_all"] is None else float(target["SeqLearning_Index_all"]["P>|z|"]),
                "ret_group_coef": np.nan if target["Retention_Sequence"] is None else float(target["Retention_Sequence"]["Coef."]),
                "ret_group_p": np.nan if target["Retention_Sequence"] is None else float(target["Retention_Sequence"]["P>|z|"]),
            }
        )
    return pd.DataFrame(rows)


def predictor_interaction_screening(pday: pd.DataFrame):
    d = pday[pday["day"] == 2].copy()
    candidates = ["AES_sum", "MoCa_sum", "GDS_sum", "NIHSS", "TSS", "MORE_sum"]
    rows = []
    for var in candidates:
        formula = f"SeqLearning_Index_all ~ C(Group) + {var} + C(Group):{var} + Age + Gender_num + fuglmayrshort_sum"
        m = smf.ols(formula, data=d.dropna(subset=["SeqLearning_Index_all", "Group", var, "Age", "Gender_num", "fuglmayrshort_sum"])).fit(
            cov_type="HC3"
        )
        term = f"C(Group)[T.B]:{var}"
        if term in m.params.index:
            rows.append(
                {
                    "variable": var,
                    "interaction_coef": float(m.params[term]),
                    "interaction_p": float(m.pvalues[term]),
                    "n": int(m.nobs),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["interaction_p_fdr"] = multipletests(out["interaction_p"], method="fdr_bh")[1]
    return out


def clinically_meaningful_stratification(pday: pd.DataFrame):
    d = pday[pday["day"] == 2].copy()
    strata_defs = {
        "MoCa_low": d["MoCa_sum"] < 24,
        "MoCa_high": d["MoCa_sum"] >= 24,
        "GDS_high": d["GDS_sum"] >= 6,
        "GDS_low": d["GDS_sum"] < 6,
        "Fugl_low": d["fuglmayrshort_sum"] <= 10,
        "Fugl_high": d["fuglmayrshort_sum"] > 10,
    }
    rows = []
    for name, mask in strata_defs.items():
        sub = d[mask].dropna(subset=["SeqLearning_Index_all", "Group", "Age", "Gender_num"])
        if sub["Group"].nunique() < 2 or len(sub) < 12:
            continue
        m = smf.ols("SeqLearning_Index_all ~ C(Group) + Age + Gender_num", data=sub).fit(cov_type="HC3")
        if "C(Group)[T.B]" in m.params.index:
            rows.append(
                {
                    "stratum": name,
                    "n": int(m.nobs),
                    "coef_B_vs_A": float(m.params["C(Group)[T.B]"]),
                    "p_value": float(m.pvalues["C(Group)[T.B]"]),
                    "ci_low": float(m.conf_int().loc["C(Group)[T.B]", 0]),
                    "ci_high": float(m.conf_int().loc["C(Group)[T.B]", 1]),
                }
            )
    return pd.DataFrame(rows)


def permutation_tests(pday: pd.DataFrame, pret: pd.DataFrame, n_perm: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)

    # SeqLearning day 2
    d2 = pday[pday["day"] == 2].dropna(subset=["SeqLearning_Index_all", "Group"]).copy()
    obs = d2[d2["Group"] == "B"]["SeqLearning_Index_all"].mean() - d2[d2["Group"] == "A"]["SeqLearning_Index_all"].mean()
    vals = d2["SeqLearning_Index_all"].to_numpy()
    groups = d2["Group"].to_numpy()
    nB = np.sum(groups == "B")
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(vals))
        b = vals[idx[:nB]]
        a = vals[idx[nB:]]
        perm_stats[i] = np.mean(b) - np.mean(a)
    p_seq = (np.sum(np.abs(perm_stats) >= abs(obs)) + 1) / (n_perm + 1)

    # Retention sequence
    r = pret.dropna(subset=["Retention_Sequence", "Group"]).copy()
    obs_r = r[r["Group"] == "B"]["Retention_Sequence"].mean() - r[r["Group"] == "A"]["Retention_Sequence"].mean()
    vals_r = r["Retention_Sequence"].to_numpy()
    groups_r = r["Group"].to_numpy()
    nB_r = np.sum(groups_r == "B")
    perm_stats_r = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(vals_r))
        b = vals_r[idx[:nB_r]]
        a = vals_r[idx[nB_r:]]
        perm_stats_r[i] = np.mean(b) - np.mean(a)
    p_ret = (np.sum(np.abs(perm_stats_r) >= abs(obs_r)) + 1) / (n_perm + 1)

    return pd.DataFrame(
        [
            {"outcome": "SeqLearning_Index_all_day2", "observed_B_minus_A": obs, "perm_p_two_sided": p_seq, "n_perm": n_perm},
            {"outcome": "Retention_Sequence", "observed_B_minus_A": obs_r, "perm_p_two_sided": p_ret, "n_perm": n_perm},
        ]
    )


def main():
    ensure_out()
    trials, block, pday, pret = load_core()
    meta = load_metadata()
    block_cov = merge_covariates(block, meta)
    block_cov = block_cov.dropna(subset=["meanRT_hit_ms", "Group", "day", "condition"])
    block_cov["day"] = block_cov["day"].astype(int)
    block_cov["logRT"] = np.log(block_cov["meanRT_hit_ms"])

    # 1) Planned contrasts from mixed model
    fit_mixed, emm, con = emm_and_contrasts_from_mixed(block_cov.dropna(subset=["Age", "AES_sum", "MoCa_sum", "NIHSS", "TSS", "BlockNumber"]))
    emm.to_csv(OUT_DIR / "planned_contrasts_emm.csv", index=False)
    con.to_csv(OUT_DIR / "planned_contrasts_simple_effects.csv", index=False)
    with open(OUT_DIR / "planned_contrasts_mixedlm_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(fit_mixed.summary()))

    # 2) Non-linear learning curves
    cmp_tbl, llr_tbl, m_lin, m_spl = nonlinear_learning_curves(block_cov)
    cmp_tbl.to_csv(OUT_DIR / "nonlinear_model_comparison.csv", index=False)
    llr_tbl.to_csv(OUT_DIR / "nonlinear_llr_test.csv", index=False)
    with open(OUT_DIR / "nonlinear_linear_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(m_lin.summary()))
    with open(OUT_DIR / "nonlinear_spline_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(m_spl.summary()))

    # 3) Blue vs green separately
    fit_seq, emm_seq, con_seq = blue_green_separate(block_cov)
    emm_seq.to_csv(OUT_DIR / "blue_green_emm.csv", index=False)
    con_seq.to_csv(OUT_DIR / "blue_green_contrasts.csv", index=False)
    with open(OUT_DIR / "blue_green_mixedlm_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(fit_seq.summary()))

    # 4) Speed-accuracy
    m_tradeoff, gee_hit = speed_accuracy_tradeoff(block_cov, trials)
    with open(OUT_DIR / "speed_accuracy_tradeoff_rt_model.txt", "w", encoding="utf-8") as f:
        f.write(str(m_tradeoff.summary()))
    with open(OUT_DIR / "speed_accuracy_tradeoff_hit_gee.txt", "w", encoding="utf-8") as f:
        f.write(str(gee_hit.summary()))
    pd.DataFrame(
        [
            {
                "term_errorRate": "errorRate",
                "coef": m_tradeoff.params.get("errorRate", np.nan),
                "p": m_tradeoff.pvalues.get("errorRate", np.nan),
            },
            {
                "term_errorRate": "C(Group)[T.B]:errorRate",
                "coef": m_tradeoff.params.get("C(Group)[T.B]:errorRate", np.nan),
                "p": m_tradeoff.pvalues.get("C(Group)[T.B]:errorRate", np.nan),
            },
        ]
    ).to_csv(OUT_DIR / "speed_accuracy_tradeoff_key_terms.csv", index=False)
    pd.DataFrame(
        [
            {"term": "zRT", "coef": gee_hit.params.get("zRT", np.nan), "p": gee_hit.pvalues.get("zRT", np.nan)},
            {
                "term": "C(Group)[T.B]:zRT",
                "coef": gee_hit.params.get("C(Group)[T.B]:zRT", np.nan),
                "p": gee_hit.pvalues.get("C(Group)[T.B]:zRT", np.nan),
            },
            {
                "term": "C(day)[T.2]:zRT",
                "coef": gee_hit.params.get("C(day)[T.2]:zRT", np.nan),
                "p": gee_hit.pvalues.get("C(day)[T.2]:zRT", np.nan),
            },
            {
                "term": "C(condition)[T.structured]:zRT",
                "coef": gee_hit.params.get("C(condition)[T.structured]:zRT", np.nan),
                "p": gee_hit.pvalues.get("C(condition)[T.structured]:zRT", np.nan),
            },
        ]
    ).to_csv(OUT_DIR / "speed_accuracy_hit_model_key_terms.csv", index=False)

    # 5) Robust + penalized
    robust_tbl, penal_tbl, rlm = robust_penalized_models(pday)
    robust_tbl.to_csv(OUT_DIR / "robust_regression_coefficients.csv", index=False)
    penal_tbl.to_csv(OUT_DIR / "penalized_regression_coefficients.csv", index=False)
    with open(OUT_DIR / "robust_regression_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(rlm.summary()))

    # 6) Missingness/data-quality sensitivity
    sens = sensitivity_missingness(trials)
    sens.to_csv(OUT_DIR / "missingness_sensitivity.csv", index=False)

    # 7) Predictor interaction screening
    inter = predictor_interaction_screening(pday)
    inter.to_csv(OUT_DIR / "predictor_interaction_screening.csv", index=False)

    # 8) Clinical stratification
    strat = clinically_meaningful_stratification(pday)
    strat.to_csv(OUT_DIR / "clinical_stratification_group_effects.csv", index=False)

    # 9) Permutation tests
    perm = permutation_tests(pday, pret)
    perm.to_csv(OUT_DIR / "permutation_tests.csv", index=False)

    manifest = {
        "planned_contrasts_rows": int(len(con)),
        "nonlinear_models_compared": 2,
        "blue_green_contrasts_rows": int(len(con_seq)),
        "interaction_screen_rows": int(len(inter)),
        "strat_rows": int(len(strat)),
        "perm_tests_rows": int(len(perm)),
    }
    with open(OUT_DIR / "manifest_additional.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

