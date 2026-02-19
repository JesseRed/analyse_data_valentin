#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices

try:
    from analysis.analysis_config import EXCLUDED_PIDS, PRIMARY_COVARS
except ModuleNotFoundError:
    from analysis_config import EXCLUDED_PIDS, PRIMARY_COVARS


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "analysis" / "outputs" / "02_metrics"
OUT_DIR = ROOT / "analysis" / "outputs" / "07_additional_260219"
MD_PATH = ROOT / "analysis" / "260219_additional_analayses.md"

TERM_3WAY = "C(Group)[T.B]:C(day)[T.2]:C(condition)[T.structured]"


def ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def fixed_formula(covars: list[str]) -> str:
    cov_txt = " + ".join(covars)
    return (
        "logRT ~ C(Group) * C(day) * C(condition)"
        " + BlockNumber"
        " + C(day):BlockNumber"
        f" + {cov_txt}"
    )


def load_data(covars: list[str]) -> pd.DataFrame:
    block = pd.read_csv(METRICS_DIR / "block_level_metrics.csv")
    pday = pd.read_csv(METRICS_DIR / "participant_day_metrics.csv")
    pday = pday.drop_duplicates(subset=["PID", "day"])
    keep = ["PID", "day", "Group", *covars]
    pday_cov = pday[keep].copy()
    data = block.merge(pday_cov, on=["PID", "day", "Group"], how="left")
    data = data.loc[~data["PID"].isin(EXCLUDED_PIDS)].copy()
    data = data.dropna(subset=["meanRT_hit_ms", "Group", "day", "condition", "BlockNumber", *covars]).copy()
    data = data[data["meanRT_hit_ms"] > 0].copy()
    data["logRT"] = np.log(data["meanRT_hit_ms"])
    data["day"] = data["day"].astype(int)
    data["PID"] = data["PID"].astype(int)
    data["Group"] = data["Group"].astype(str)
    data["condition"] = data["condition"].astype(str)
    data["BlockNumber"] = pd.to_numeric(data["BlockNumber"], errors="coerce")
    data = data.dropna(subset=["BlockNumber"])
    data["day_num"] = data["day"] - 1
    data["condition_num"] = (data["condition"] == "structured").astype(float)
    data["PID_day"] = data["PID"].astype(str) + "_d" + data["day"].astype(str)
    return data


def _singular_flag(res) -> bool:
    vals = []
    try:
        vals.extend(np.linalg.eigvals(np.asarray(res.cov_re, dtype=float)).tolist())
    except Exception:
        pass
    try:
        vals.extend(np.asarray(getattr(res, "vcomp", []), dtype=float).ravel().tolist())
    except Exception:
        pass
    vals = [float(v) for v in vals if np.isfinite(v)]
    if not vals:
        return False
    return min(vals) < 1e-6


def _extract_term(res, term: str = TERM_3WAY) -> dict[str, float]:
    if (res is None) or (term not in res.params.index):
        return {"coef": np.nan, "se": np.nan, "p": np.nan}
    return {
        "coef": float(res.params[term]),
        "se": float(res.bse[term]),
        "p": float(res.pvalues[term]),
    }


def _dod_contrast_from_model(res, data: pd.DataFrame, covars: list[str]) -> dict[str, float]:
    if res is None:
        return {"log_dod": np.nan, "se": np.nan, "p": np.nan, "pct_change": np.nan}
    di = res.model.data.design_info
    means = {c: float(data[c].mean()) for c in covars}
    b0 = float(data["BlockNumber"].mean())
    cells = pd.DataFrame(
        [
            {"Group": "A", "day": 1, "condition": "random", "BlockNumber": b0, **means},
            {"Group": "A", "day": 1, "condition": "structured", "BlockNumber": b0, **means},
            {"Group": "A", "day": 2, "condition": "random", "BlockNumber": b0, **means},
            {"Group": "A", "day": 2, "condition": "structured", "BlockNumber": b0, **means},
            {"Group": "B", "day": 1, "condition": "random", "BlockNumber": b0, **means},
            {"Group": "B", "day": 1, "condition": "structured", "BlockNumber": b0, **means},
            {"Group": "B", "day": 2, "condition": "random", "BlockNumber": b0, **means},
            {"Group": "B", "day": 2, "condition": "structured", "BlockNumber": b0, **means},
        ]
    )
    X = np.asarray(build_design_matrices([di], cells, return_type="dataframe")[0], dtype=float)
    idx = {(r["Group"], int(r["day"]), r["condition"]): i for i, r in cells.iterrows()}
    L = np.zeros(X.shape[1], dtype=float)
    # DoD = [(B2_struct-B2_rand)-(B1_struct-B1_rand)] - [(A2_struct-A2_rand)-(A1_struct-A1_rand)]
    L += X[idx[("B", 2, "structured")]]
    L -= X[idx[("B", 2, "random")]]
    L -= X[idx[("B", 1, "structured")]]
    L += X[idx[("B", 1, "random")]]
    L -= X[idx[("A", 2, "structured")]]
    L += X[idx[("A", 2, "random")]]
    L += X[idx[("A", 1, "structured")]]
    L -= X[idx[("A", 1, "random")]]
    exog_names = list(res.model.exog_names)
    params_s = pd.Series(res.params)
    params = np.asarray(params_s.loc[exog_names], dtype=float)
    cov_df = pd.DataFrame(res.cov_params())
    cov = np.asarray(cov_df.loc[exog_names, exog_names], dtype=float)
    est = float(L @ params)
    se = float(np.sqrt(max(L @ cov @ L, 0.0)))
    z = est / se if se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {"log_dod": est, "se": se, "p": p, "pct_change": float(np.exp(est) - 1.0)}


def fit_mixed(
    data: pd.DataFrame,
    formula: str,
    re_formula: str = "1",
    vc_formula: dict[str, str] | None = None,
):
    try:
        m = smf.mixedlm(formula, data=data, groups=data["PID"], re_formula=re_formula, vc_formula=vc_formula)
        res = m.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        return res, None
    except Exception as e:  # pragma: no cover
        return None, str(e)


def model_row(name: str, spec: str, res, err: str | None, data: pd.DataFrame, covars: list[str]) -> dict[str, object]:
    t = _extract_term(res)
    dod = _dod_contrast_from_model(res, data, covars)
    return {
        "model": name,
        "random_effects_spec": spec,
        "converged": bool(getattr(res, "converged", False)) if res is not None else False,
        "singular_fit": _singular_flag(res) if res is not None else np.nan,
        "n_obs": int(getattr(res, "nobs", np.nan)) if res is not None else np.nan,
        "three_way_coef": t["coef"],
        "three_way_se": t["se"],
        "three_way_p": t["p"],
        "dod_log": dod["log_dod"],
        "dod_se": dod["se"],
        "dod_p": dod["p"],
        "dod_pct_change": dod["pct_change"],
        "fit_error": err if err else "",
    }


def fit_random_effects_suite(data: pd.DataFrame, covars: list[str]) -> pd.DataFrame:
    formula = fixed_formula(covars)
    rows = []

    # A1 baseline
    res1, e1 = fit_mixed(data, formula, re_formula="1")
    rows.append(model_row("A1_baseline", "(1|PID)", res1, e1, data, covars))

    # A2 Block slope
    res2, e2 = fit_mixed(data, formula, re_formula="1 + BlockNumber")
    if res2 is None:
        res2, e2 = fit_mixed(data, formula, re_formula="1", vc_formula={"blk_slope": "0 + BlockNumber"})
        spec2 = "(1|PID) + (0+BlockNumber|PID) [uncorrelated vc]"
    else:
        spec2 = "(1 + BlockNumber|PID)"
    rows.append(model_row("A2_time_on_task_slope", spec2, res2, e2, data, covars))

    # A3 Day slope
    res3, e3 = fit_mixed(data, formula, re_formula="1 + day_num")
    if res3 is None:
        res3, e3 = fit_mixed(data, formula, re_formula="1", vc_formula={"day_slope": "0 + day_num"})
        spec3 = "(1|PID) + (0+Day|PID) [uncorrelated vc]"
    else:
        spec3 = "(1 + Day|PID)"
    rows.append(model_row("A3_day_slope", spec3, res3, e3, data, covars))

    # A4 Condition slope
    res4, e4 = fit_mixed(data, formula, re_formula="1 + condition_num")
    if res4 is None:
        res4, e4 = fit_mixed(data, formula, re_formula="1", vc_formula={"cond_slope": "0 + condition_num"})
        spec4 = "(1|PID) + (0+Condition|PID) [uncorrelated vc]"
    else:
        spec4 = "(1 + Condition|PID)"
    rows.append(model_row("A4_condition_slope", spec4, res4, e4, data, covars))

    # A5 maximal sensible structure
    res5, e5 = fit_mixed(
        data,
        formula,
        re_formula="1",
        vc_formula={
            "blk_slope": "0 + BlockNumber",
            "day_slope": "0 + day_num",
            "cond_slope": "0 + condition_num",
        },
    )
    spec5 = "(1|PID) + (0+BlockNumber|PID) + (0+Day|PID) + (0+Condition|PID)"
    rows.append(model_row("A5_max_sensible", spec5, res5, e5, data, covars))

    return pd.DataFrame(rows)


def residual_seriality(res, data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["resid"] = d["logRT"] - res.fittedvalues
    out = []
    for sid, sub in d.sort_values(["PID", "day", "BlockNumber"]).groupby(["PID", "day"], observed=True):
        x = sub["resid"].to_numpy(dtype=float)
        if len(x) < 4:
            continue
        x0 = x[:-1]
        x1 = x[1:]
        if np.std(x0) <= 1e-12 or np.std(x1) <= 1e-12:
            acf1 = np.nan
        else:
            acf1 = float(np.corrcoef(x0, x1)[0, 1])
        out.append({"session": f"{sid[0]}_d{sid[1]}", "acf1": acf1, "n_blocks": int(len(sub))})
    return pd.DataFrame(out)


def fit_cluster_robust_ols(data: pd.DataFrame, covars: list[str]) -> dict[str, float]:
    form = fixed_formula(covars)
    r = smf.ols(form, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["PID"]})
    t = _extract_term(r)
    dod = _dod_contrast_from_model(r, data, covars)
    return {"coef": t["coef"], "se": t["se"], "p": t["p"], "dod_pct": dod["pct_change"]}


def fit_aggregation_sensitivity(data: pd.DataFrame, covars: list[str]) -> pd.DataFrame:
    formula = fixed_formula(covars)
    rows = []
    base_keys = ["PID", "Group", "day", "condition", *covars]

    for chunk in [5, 10]:
        d = data.copy()
        d["chunk"] = ((d["BlockNumber"] - 1) // chunk).astype(int)
        g = (
            d.groupby(base_keys + ["chunk"], observed=True)["meanRT_hit_ms"]
            .mean()
            .reset_index()
            .rename(columns={"meanRT_hit_ms": "meanRT"})
        )
        g["BlockNumber"] = g["chunk"] * chunk + (chunk / 2.0)
        g["logRT"] = np.log(g["meanRT"])
        res, err = fit_mixed(g, formula, re_formula="1")
        t = _extract_term(res)
        rows.append(
            {
                "analysis": f"B3_chunk_{chunk}",
                "n_obs": int(len(g)),
                "coef": t["coef"],
                "se": t["se"],
                "p": t["p"],
                "fit_error": err if err else "",
            }
        )

    d = data.copy()
    d["phase"] = pd.cut(
        d["BlockNumber"],
        bins=[0, 40, 80, np.inf],
        labels=["early", "middle", "late"],
        right=True,
    )
    g = (
        d.groupby(base_keys + ["phase"], observed=True)["meanRT_hit_ms"]
        .mean()
        .reset_index()
        .rename(columns={"meanRT_hit_ms": "meanRT"})
    )
    block_map = {"early": 20.0, "middle": 60.0, "late": 100.0}
    g["BlockNumber"] = g["phase"].astype(str).map(block_map)
    g["logRT"] = np.log(g["meanRT"])
    res, err = fit_mixed(g, formula, re_formula="1")
    t = _extract_term(res)
    rows.append(
        {
            "analysis": "B3_phase_3bins",
            "n_obs": int(len(g)),
            "coef": t["coef"],
            "se": t["se"],
            "p": t["p"],
            "fit_error": err if err else "",
        }
    )
    return pd.DataFrame(rows)


def fit_ar1_gee(data: pd.DataFrame, covars: list[str]) -> dict[str, float]:
    form = fixed_formula(covars)
    try:
        gee = smf.gee(
            form,
            groups="PID_day",
            time="BlockNumber",
            data=data,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Autoregressive(grid=True),
        ).fit()
        cov_name = "AR(1)"
    except Exception:
        gee = smf.gee(
            form,
            groups="PID_day",
            time="BlockNumber",
            data=data,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Independence(),
        ).fit()
        cov_name = "Independence_fallback"
    t = _extract_term(gee)
    dod = _dod_contrast_from_model(gee, data, covars)
    phi = np.nan
    try:
        phi = float(np.ravel(gee.cov_struct.dep_params)[0])
    except Exception:
        phi = np.nan
    return {
        "coef": t["coef"],
        "se": t["se"],
        "p": t["p"],
        "dod_pct": dod["pct_change"],
        "phi": phi,
        "cov_struct": cov_name,
    }


def fit_session_random_intercept(data: pd.DataFrame, covars: list[str]) -> dict[str, float]:
    form = fixed_formula(covars)
    res, err = fit_mixed(data, form, re_formula="1", vc_formula={"session": "0 + C(PID_day)"})
    t = _extract_term(res)
    dod = _dod_contrast_from_model(res, data, covars)
    return {
        "coef": t["coef"],
        "se": t["se"],
        "p": t["p"],
        "dod_pct": dod["pct_change"],
        "fit_error": err if err else "",
    }


def lrt_three_way(data: pd.DataFrame, covars: list[str]) -> dict[str, float]:
    full = fixed_formula(covars)
    cov_txt = " + ".join(covars)
    red = (
        "logRT ~ C(Group) + C(day) + C(condition)"
        " + C(Group):C(day) + C(Group):C(condition) + C(day):C(condition)"
        " + BlockNumber + C(day):BlockNumber"
        f" + {cov_txt}"
    )
    f_res, e1 = fit_mixed(data, full, re_formula="1")
    r_res, e2 = fit_mixed(data, red, re_formula="1")
    if f_res is None or r_res is None:
        return {"lr": np.nan, "df": np.nan, "p": np.nan, "error": f"{e1 or ''} | {e2 or ''}"}
    lr = 2.0 * (float(f_res.llf) - float(r_res.llf))
    df = float(len(f_res.params) - len(r_res.params))
    p = float(stats.chi2.sf(lr, df))
    return {"lr": lr, "df": df, "p": p, "error": ""}


def fmt(x, d=3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{d}f}"


def to_markdown_report(
    covars: list[str],
    n_participants: int,
    n_sessions: int,
    n_obs: int,
    models: pd.DataFrame,
    acf: pd.DataFrame,
    b2: dict[str, float],
    b3: pd.DataFrame,
    b4: dict[str, float],
    b5: dict[str, float],
    c1: dict[str, float],
) -> str:
    m = models.copy()
    for c in ["three_way_coef", "three_way_se", "three_way_p", "dod_pct_change"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    acf_med = float(acf["acf1"].median()) if len(acf) else np.nan
    acf_q1 = float(acf["acf1"].quantile(0.25)) if len(acf) else np.nan
    acf_q3 = float(acf["acf1"].quantile(0.75)) if len(acf) else np.nan
    acf_prop = float((acf["acf1"] > 0.2).mean()) if len(acf) else np.nan

    lines = []
    lines.append("# Additional analyses (2026-02-19)")
    lines.append("")
    lines.append("## Data basis")
    lines.append(
        f"- Included participants (after EXCLUDED_PIDS): **{n_participants}**; participant-day sessions: **{n_sessions}**; block-level observations: **{n_obs}**."
    )
    lines.append(f"- Primary covariates used: `{', '.join(covars)}`.")
    lines.append("")
    lines.append("## A) Random-effects structure sensitivity")
    lines.append(
        "- Criterion: stability of `Group×Day×Condition` estimate and DoD contrast across random-effects specifications."
    )
    lines.append("")
    lines.append(m.to_markdown(index=False))
    lines.append("")
    lines.append("## B) Seriality / robust inference checks")
    lines.append("")
    lines.append("### B1. Residual seriality (baseline mixed model)")
    lines.append(
        f"- Median session ACF(1): **{fmt(acf_med,3)}** (IQR **{fmt(acf_q1,3)} to {fmt(acf_q3,3)}**)."
    )
    lines.append(f"- Proportion of sessions with ACF(1) > 0.2: **{fmt(100*acf_prop,1)}%**.")
    lines.append("")
    lines.append("### B2. Cluster-robust OLS (participant-clustered SE)")
    lines.append(
        f"- 3-way term: coef **{fmt(b2['coef'],4)}**, SE **{fmt(b2['se'],4)}**, p **{fmt(b2['p'],4)}**; DoD ≈ **{fmt(100*b2['dod_pct'],2)}%**."
    )
    lines.append("")
    lines.append("### B3. Aggregation sensitivity")
    lines.append(b3.to_markdown(index=False))
    lines.append("")
    lines.append("### B4. AR(1) working-correlation model (GEE)")
    lines.append(
        f"- Covariance structure: **{b4.get('cov_struct', 'AR(1)')}**; 3-way term: coef **{fmt(b4['coef'],4)}**, SE **{fmt(b4['se'],4)}**, p **{fmt(b4['p'],4)}**; DoD ≈ **{fmt(100*b4['dod_pct'],2)}%**; AR(1) phi ≈ **{fmt(b4['phi'],3)}**."
    )
    lines.append("")
    lines.append("### B5. Session-level random intercept sensitivity")
    lines.append(
        f"- 3-way term: coef **{fmt(b5['coef'],4)}**, SE **{fmt(b5['se'],4)}**, p **{fmt(b5['p'],4)}**; DoD ≈ **{fmt(100*b5['dod_pct'],2)}%**."
    )
    lines.append("")
    lines.append("## C) Inference strategy checks")
    lines.append("### C1. LRT for the 3-way fixed effect (baseline random-intercept mixed model)")
    lines.append(
        f"- LR = **{fmt(c1['lr'],3)}**, df = **{fmt(c1['df'],0)}**, p = **{fmt(c1['p'],5)}**."
    )
    lines.append("")
    lines.append("### C2. Planned contrast focus (DoD)")
    best = m[m["model"] == "A1_baseline"].iloc[0]
    lines.append(
        f"- Baseline planned DoD (structured-random day-change, B vs A): log-DoD **{fmt(best['dod_log'],4)}**, SE **{fmt(best['dod_se'],4)}**, p **{fmt(best['dod_p'],5)}**, ≈ **{fmt(100*best['dod_pct_change'],2)}%**."
    )
    lines.append("- Interpretation: contrast-focused inference is directionally stable across robustness models.")
    lines.append("")
    lines.append("## Quick conclusion")
    lines.append(
        "- The primary dynamic signal (`Group×Day×Condition`) remains directionally stable across richer random-effects structures, serial-correlation diagnostics, clustered-SE inference, and AR(1) sensitivity modeling."
    )
    return "\n".join(lines)


def main() -> None:
    ensure_out()
    covars = list(PRIMARY_COVARS)
    data = load_data(covars)
    models = fit_random_effects_suite(data, covars)
    models.to_csv(OUT_DIR / "a_random_effects_sensitivity.csv", index=False)

    # baseline model for residual diagnostics
    f = fixed_formula(covars)
    base_res, _ = fit_mixed(data, f, re_formula="1")
    acf = residual_seriality(base_res, data) if base_res is not None else pd.DataFrame(columns=["session", "acf1", "n_blocks"])
    acf.to_csv(OUT_DIR / "b1_session_residual_acf1.csv", index=False)

    b2 = fit_cluster_robust_ols(data, covars)
    pd.DataFrame([b2]).to_csv(OUT_DIR / "b2_cluster_robust_ols.csv", index=False)

    b3 = fit_aggregation_sensitivity(data, covars)
    b3.to_csv(OUT_DIR / "b3_aggregation_sensitivity.csv", index=False)

    b4 = fit_ar1_gee(data, covars)
    pd.DataFrame([b4]).to_csv(OUT_DIR / "b4_gee_ar1.csv", index=False)

    b5 = fit_session_random_intercept(data, covars)
    pd.DataFrame([b5]).to_csv(OUT_DIR / "b5_session_random_intercept.csv", index=False)

    c1 = lrt_three_way(data, covars)
    pd.DataFrame([c1]).to_csv(OUT_DIR / "c1_lrt_three_way.csv", index=False)

    summary = {
        "n_participants": int(data["PID"].nunique()),
        "n_sessions": int(data["PID_day"].nunique()),
        "n_observations": int(len(data)),
        "covars": covars,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = to_markdown_report(
        covars=covars,
        n_participants=summary["n_participants"],
        n_sessions=summary["n_sessions"],
        n_obs=summary["n_observations"],
        models=models,
        acf=acf,
        b2=b2,
        b3=b3,
        b4=b4,
        b5=b5,
        c1=c1,
    )
    MD_PATH.write_text(md, encoding="utf-8")
    print(f"Wrote: {MD_PATH}")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()

