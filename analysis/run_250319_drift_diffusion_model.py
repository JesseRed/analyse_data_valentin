#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from patsy import build_design_matrices

try:
    from analysis.analysis_config import EXCLUDED_PIDS, PRIMARY_COVARS
except ModuleNotFoundError:
    from analysis_config import EXCLUDED_PIDS, PRIMARY_COVARS


ROOT = Path(__file__).resolve().parents[1]
TRIAL_PATH = ROOT / "analysis" / "outputs" / "01_ingest_and_qc" / "trial_level_all.csv"
PDAY_PATH = ROOT / "analysis" / "outputs" / "02_metrics" / "participant_day_metrics.csv"
OUT_DIR = ROOT / "analysis" / "outputs" / "08_ddm_250319"
MD_PATH = ROOT / "analysis" / "250319_drift_diffusion_model.md"

TERM_3WAY = "C(Group)[T.B]:C(day)[T.2]:C(condition)[T.structured]"
SCALE_S = 0.1  # EZ-DDM conventional scaling constant


def ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def ez_diffusion(pc: float, vrt: float, mrt: float, s: float = SCALE_S) -> tuple[float, float, float]:
    """
    Wagenmakers et al. EZ-diffusion approximation.
    Returns (v, a, Ter) = (drift rate, boundary separation, non-decision time).
    """
    pc = float(np.clip(pc, 1e-4, 1 - 1e-4))
    if vrt <= 0 or mrt <= 0:
        return np.nan, np.nan, np.nan

    L = np.log(pc / (1 - pc))
    x = (L * (pc**2 * L - pc * L + pc - 0.5)) / vrt
    if x <= 0 or not np.isfinite(x):
        return np.nan, np.nan, np.nan

    v = np.sign(pc - 0.5) * s * (x ** 0.25)
    if abs(v) < 1e-12 or not np.isfinite(v):
        return np.nan, np.nan, np.nan

    a = (s**2 * L) / v
    y = (-v * a) / (s**2)
    mdt = (a / (2 * v)) * ((1 - np.exp(y)) / (1 + np.exp(y)))
    ter = mrt - mdt
    if not np.isfinite(a) or not np.isfinite(ter):
        return np.nan, np.nan, np.nan
    return float(v), float(a), float(ter)


def load_trial_data() -> pd.DataFrame:
    t = pd.read_csv(TRIAL_PATH)
    t = t.loc[~t["PID"].isin(EXCLUDED_PIDS)].copy()
    t["isHit"] = pd.to_numeric(t["isHit"], errors="coerce")
    t["RT_s"] = pd.to_numeric(t["RT_s"], errors="coerce")
    t["RT_ms"] = pd.to_numeric(t["RT_ms"], errors="coerce")

    # Keep valid trials for RT+accuracy modeling, including errors.
    t = t[
        (~t["flag_parse_or_reset"])
        & (~t["flag_sequence_invalid"])
        & (t["keep_hard"])
        & (t["RT_ms"] > 0)
        & np.isfinite(t["RT_ms"])
        & t["condition"].isin(["random", "structured"])
    ].copy()
    t["day"] = t["day"].astype(int)
    t["PID"] = t["PID"].astype(int)
    t["Group"] = t["Group"].astype(str)
    t["condition"] = t["condition"].astype(str)
    return t


def compute_ddm_cells(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grp = t.groupby(["PID", "Group", "day", "condition"], observed=True)
    for (pid, group, day, cond), sub in grp:
        n_trials = int(len(sub))
        n_correct = int((sub["isHit"] == 1).sum())
        pc = n_correct / n_trials if n_trials > 0 else np.nan
        rt_correct = sub.loc[sub["isHit"] == 1, "RT_s"].dropna().to_numpy()
        mrt = float(np.mean(rt_correct)) if len(rt_correct) else np.nan
        vrt = float(np.var(rt_correct, ddof=1)) if len(rt_correct) >= 2 else np.nan

        valid = (
            np.isfinite(pc)
            and np.isfinite(mrt)
            and np.isfinite(vrt)
            and (n_trials >= 20)
            and (n_correct >= 8)
            and (0.55 <= pc <= 0.99)
        )
        if valid:
            v, a, ter = ez_diffusion(pc=pc, vrt=vrt, mrt=mrt, s=SCALE_S)
        else:
            v, a, ter = (np.nan, np.nan, np.nan)

        rows.append(
            {
                "PID": pid,
                "Group": group,
                "day": day,
                "condition": cond,
                "n_trials": n_trials,
                "n_correct": n_correct,
                "pcorrect": pc,
                "mrt_s_correct": mrt,
                "vrt_s2_correct": vrt,
                "v_drift": v,
                "a_boundary": a,
                "ter_nondecision_s": ter,
                "ddm_valid": bool(np.isfinite(v) and np.isfinite(a) and np.isfinite(ter) and (ter > 0)),
            }
        )

    d = pd.DataFrame(rows)
    d = d[d["ddm_valid"]].copy()
    d["condition"] = pd.Categorical(d["condition"], categories=["random", "structured"], ordered=True)
    return d


def add_covariates(d: pd.DataFrame) -> pd.DataFrame:
    pday = pd.read_csv(PDAY_PATH)
    pday = pday.drop_duplicates(subset=["PID", "day"])[["PID", "day", *PRIMARY_COVARS]]
    merged = d.merge(pday, on=["PID", "day"], how="left")
    merged = merged.dropna(subset=list(PRIMARY_COVARS)).copy()
    return merged


def fit_param_model(d: pd.DataFrame, outcome: str):
    f = (
        f"{outcome} ~ C(Group) * C(day) * C(condition)"
        + " + "
        + " + ".join(PRIMARY_COVARS)
    )
    m = smf.mixedlm(f, data=d, groups=d["PID"], re_formula="1")
    res = m.fit(reml=False, method="lbfgs", maxiter=300, disp=False)
    return res, f


def dod_from_model(res, d: pd.DataFrame) -> dict[str, float]:
    di = res.model.data.design_info
    means = {c: float(d[c].mean()) for c in PRIMARY_COVARS}
    cells = pd.DataFrame(
        [
            {"Group": "A", "day": 1, "condition": "random", **means},
            {"Group": "A", "day": 1, "condition": "structured", **means},
            {"Group": "A", "day": 2, "condition": "random", **means},
            {"Group": "A", "day": 2, "condition": "structured", **means},
            {"Group": "B", "day": 1, "condition": "random", **means},
            {"Group": "B", "day": 1, "condition": "structured", **means},
            {"Group": "B", "day": 2, "condition": "random", **means},
            {"Group": "B", "day": 2, "condition": "structured", **means},
        ]
    )
    X = np.asarray(build_design_matrices([di], cells, return_type="dataframe")[0], dtype=float)
    exog_names = list(res.model.exog_names)
    params = np.asarray(pd.Series(res.params).loc[exog_names], dtype=float)
    cov = np.asarray(pd.DataFrame(res.cov_params()).loc[exog_names, exog_names], dtype=float)
    idx = {(r["Group"], int(r["day"]), str(r["condition"])): i for i, r in cells.iterrows()}
    L = np.zeros(X.shape[1], dtype=float)
    L += X[idx[("B", 2, "structured")]]
    L -= X[idx[("B", 2, "random")]]
    L -= X[idx[("B", 1, "structured")]]
    L += X[idx[("B", 1, "random")]]
    L -= X[idx[("A", 2, "structured")]]
    L += X[idx[("A", 2, "random")]]
    L += X[idx[("A", 1, "structured")]]
    L -= X[idx[("A", 1, "random")]]
    est = float(L @ params)
    se = float(np.sqrt(max(L @ cov @ L, 0.0)))
    z = est / se if se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {"dod_est": est, "dod_se": se, "dod_p": p}


def fmt(x: float, d: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{d}f}"


def main() -> None:
    ensure_out()
    t = load_trial_data()
    ddm = compute_ddm_cells(t)
    ddm = add_covariates(ddm)

    ddm.to_csv(OUT_DIR / "ddm_cell_estimates.csv", index=False)

    summary_rows = []
    lines = []
    lines.append("# Trial-level speed-accuracy via diffusion modeling (EZ-DDM)")
    lines.append("")
    lines.append("## Method")
    lines.append(
        "- We used an EZ-diffusion approximation (Wagenmakers et al.) on participant × day × condition cells, based on trial-level accuracy and correct-trial RT moments."
    )
    lines.append(
        "- Parameters: drift rate (`v_drift`), boundary separation (`a_boundary`, caution), and non-decision time (`ter_nondecision_s`)."
    )
    lines.append(
        "- Inclusion for valid EZ-DDM cells: `n_trials >= 20`, `n_correct >= 8`, `0.55 <= p(correct) <= 0.99`, finite RT moments, and positive `Ter`."
    )
    lines.append(
        f"- Mixed models: `parameter ~ Group × Day × Condition + {' + '.join(PRIMARY_COVARS)} + (1|PID)`."
    )
    lines.append("")
    lines.append("## Data coverage")
    lines.append(f"- Participants with valid DDM cells: **{ddm['PID'].nunique()}**")
    lines.append(f"- Valid cells (PID×day×condition): **{len(ddm)}**")
    lines.append("")

    for param, label in [
        ("v_drift", "Drift rate (information processing)"),
        ("a_boundary", "Boundary separation (caution)"),
        ("ter_nondecision_s", "Non-decision time"),
    ]:
        try:
            res, formula = fit_param_model(ddm, param)
            term_coef = float(res.params.get(TERM_3WAY, np.nan))
            term_se = float(res.bse.get(TERM_3WAY, np.nan))
            term_p = float(res.pvalues.get(TERM_3WAY, np.nan))
            dod = dod_from_model(res, ddm)
            summary_rows.append(
                {
                    "parameter": param,
                    "label": label,
                    "n_obs": int(res.nobs),
                    "three_way_coef": term_coef,
                    "three_way_se": term_se,
                    "three_way_p": term_p,
                    "dod_est": dod["dod_est"],
                    "dod_se": dod["dod_se"],
                    "dod_p": dod["dod_p"],
                    "formula": formula,
                }
            )
        except Exception as e:
            summary_rows.append(
                {
                    "parameter": param,
                    "label": label,
                    "n_obs": np.nan,
                    "three_way_coef": np.nan,
                    "three_way_se": np.nan,
                    "three_way_p": np.nan,
                    "dod_est": np.nan,
                    "dod_se": np.nan,
                    "dod_p": np.nan,
                    "formula": "ERROR",
                    "error": str(e),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "ddm_mixed_model_summary.csv", index=False)

    lines.append("## Model results")
    show = summary[
        ["label", "n_obs", "three_way_coef", "three_way_se", "three_way_p", "dod_est", "dod_se", "dod_p"]
    ].copy()
    show = show.rename(
        columns={
            "label": "Parameter",
            "n_obs": "n_obs",
            "three_way_coef": "Group×Day×Condition coef",
            "three_way_se": "SE",
            "three_way_p": "p",
            "dod_est": "DoD estimate",
            "dod_se": "DoD SE",
            "dod_p": "DoD p",
        }
    )
    lines.append(show.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation (VR-related mechanism)")

    def _row(p: str) -> pd.Series:
        return summary.loc[summary["parameter"] == p].iloc[0]

    rv = _row("v_drift")
    ra = _row("a_boundary")
    rt = _row("ter_nondecision_s")

    lines.append(
        f"- **Drift (`v`)**: 3-way p = **{fmt(rv['three_way_p'],4)}**; DoD p = **{fmt(rv['dod_p'],4)}**."
    )
    lines.append(
        f"- **Caution (`a`)**: 3-way p = **{fmt(ra['three_way_p'],4)}**; DoD p = **{fmt(ra['dod_p'],4)}**."
    )
    lines.append(
        f"- **Non-decision (`Ter`)**: 3-way p = **{fmt(rt['three_way_p'],4)}**; DoD p = **{fmt(rt['dod_p'],4)}**."
    )
    lines.append(
        "- Practical reading: evidence for a VR-related change in information processing is strongest if `v_drift` terms are significant; evidence for strategy/caution differences is strongest if `a_boundary` terms are significant."
    )
    lines.append("")
    lines.append("## Output files")
    lines.append(f"- `{OUT_DIR / 'ddm_cell_estimates.csv'}`")
    lines.append(f"- `{OUT_DIR / 'ddm_mixed_model_summary.csv'}`")

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "excluded_pids": EXCLUDED_PIDS,
                "primary_covars": PRIMARY_COVARS,
                "n_trials_clean": int(len(t)),
                "n_ddm_cells": int(len(ddm)),
                "n_participants_ddm": int(ddm["PID"].nunique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {MD_PATH}")
    print(f"Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()

