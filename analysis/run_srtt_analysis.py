#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import mstats
from statsmodels.regression.mixed_linear_model import MixedLMResults

try:
    # When imported as `analysis.run_srtt_analysis` (namespace package).
    from analysis.analysis_config import EXCLUDED_PIDS, PRIMARY_COVARS
except ModuleNotFoundError:
    # When executed as a script: `python analysis/run_srtt_analysis.py`.
    from analysis_config import EXCLUDED_PIDS, PRIMARY_COVARS


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
META_PATH = ROOT / "Datensatz_Round_2.csv"
OUT_DIR = ROOT / "analysis" / "outputs"
DIRS = {
    "inventory": OUT_DIR / "01_ingest_and_qc",
    "metrics": OUT_DIR / "02_metrics",
    "models": OUT_DIR / "03_models",
    "figures": OUT_DIR / "04_figures",
}

EXPECTED_SRT_COLUMNS = [
    "BlockNumber",
    "EventNumber",
    "Time Since Block start",
    "isHit",
    "target",
    "pressed",
    "sequence",
]
EXPECTED_SEQUENCES = {"blue", "green", "yellow"}

RT_MIN_MS = 150.0
RT_MAX_MS = 3000.0
MAD_SCALE = 1.4826
MAD_MULTIPLIER = 3.0

# Primary, publication-focused minimal adjustment set (configured in analysis_config).
NUMERIC_COVARS = list(PRIMARY_COVARS)


@dataclass
class ModelResult:
    outcome: str
    n: int
    formula: str
    summary_text: str
    coef_table: pd.DataFrame


def ensure_dirs() -> None:
    for directory in DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)


def parse_pid_to_folder(pid: int) -> str:
    return f"{pid:02d}"


def normalize_decimal_comma(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def parse_sports_activity(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    mapping = {
        "no": 0.0,
        "1h_week": 1.0,
        "1-3h_week": 2.0,
        ">3h_week": 3.0,
    }
    return mapping.get(text, np.nan)


def parse_depression(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in {"no", "0", "false"}:
        return 0.0
    if text.startswith("yes"):
        return 1.0
    return np.nan


def parse_gender(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text == "female":
        return 0.0
    if text == "male":
        return 1.0
    return np.nan


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(
        META_PATH,
        sep=";",
        na_values=["", "null", "N.A.", "NA"],
        low_memory=False,
    )
    meta["PID"] = pd.to_numeric(meta["PID"], errors="coerce").astype("Int64")
    meta = meta.dropna(subset=["PID"]).copy()
    meta["PID"] = meta["PID"].astype(int)
    meta = meta.loc[~meta["PID"].isin(EXCLUDED_PIDS)].copy()

    if "Age" in meta.columns:
        meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    else:
        meta["Age"] = pd.to_numeric(meta.get("Biomag_Untersuchung"), errors="coerce")

    if "Gender" not in meta.columns:
        meta["Gender"] = meta.get("Biomag_gender")
    if "Depression" not in meta.columns:
        meta["Depression"] = meta.get("Biomag_History of depression")
    if "SportsActivity" not in meta.columns:
        meta["SportsActivity"] = meta.get("Biomag_Sports activity")

    meta["Gender_num"] = meta["Gender"].map(parse_gender)
    meta["Depression_num"] = meta["Depression"].map(parse_depression)
    meta["SportsActivity_num"] = meta["SportsActivity"].map(parse_sports_activity)

    numeric_candidates = [
        "AES_sum",
        "fuglmayrshort_sum",
        "EQ5D_health_status",
        "GDS_sum",
        "MoCa_sum",
        "MORE_sum",
        "TSS",
        "NIHSS",
    ]
    for col in numeric_candidates:
        meta[col] = pd.to_numeric(meta.get(col), errors="coerce")

    keep_cols = [
        "PID",
        "Group",
        "AES_sum",
        "Age",
        "Gender",
        "Gender_num",
        "Depression",
        "Depression_num",
        "SportsActivity",
        "SportsActivity_num",
        "fuglmayrshort_sum",
        "EQ5D_health_status",
        "GDS_sum",
        "MoCa_sum",
        "MORE_sum",
        "TSS",
        "NIHSS",
    ]
    return meta[keep_cols].copy()


def file_rows(path: Path) -> int:
    try:
        return pd.read_csv(path, sep=";", usecols=[0]).shape[0]
    except Exception:
        return -1


def build_inventory(meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, dict[int, Path]]]:
    records: list[dict[str, object]] = []
    selected_files: dict[int, dict[int, Path]] = {}

    for _, row in meta.iterrows():
        pid = int(row["PID"])
        folder = DATA_DIR / parse_pid_to_folder(pid)
        day_files = {}
        for day in (1, 2):
            # Some exports include an extra token before day number (e.g., *_FRA_2_fertig.csv).
            patterns = [f"*_{day}_fertig.csv", f"*_FRA_{day}_fertig.csv"]
            matches: list[Path] = []
            for pat in patterns:
                matches.extend(folder.glob(pat))
            files = sorted({p.resolve() for p in matches})
            chosen = files[0] if files else None
            day_files[day] = chosen
            records.append(
                {
                    "PID": pid,
                    "PID2": parse_pid_to_folder(pid),
                    "Group": row["Group"],
                    "day": day,
                    "folder_exists": folder.exists(),
                    "n_candidate_files": len(files),
                    "selected_file": str(chosen) if chosen else None,
                    "selected_file_rows": file_rows(chosen) if chosen else 0,
                }
            )

        selected_files[pid] = {d: p for d, p in day_files.items() if p is not None}

    inventory = pd.DataFrame(records)
    inventory["has_both_days"] = inventory.groupby("PID")["selected_file"].transform(
        lambda s: s.notna().sum() == 2
    )
    return inventory, selected_files


def load_all_trials(meta: pd.DataFrame, selected_files: dict[int, dict[int, Path]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for _, mrow in meta.iterrows():
        pid = int(mrow["PID"])
        group = mrow["Group"]
        files = selected_files.get(pid, {})
        for day in (1, 2):
            path = files.get(day)
            if path is None:
                continue
            raw = pd.read_csv(path, sep=";", na_values=["", "null", "N.A.", "NA"], low_memory=False)
            if any(col not in raw.columns for col in EXPECTED_SRT_COLUMNS):
                missing = [c for c in EXPECTED_SRT_COLUMNS if c not in raw.columns]
                raise ValueError(f"Missing expected columns in {path}: {missing}")

            d = pd.DataFrame(
                {
                    "PID": pid,
                    "Group": group,
                    "day": day,
                    "source_file": str(path),
                    "BlockNumber": pd.to_numeric(raw["BlockNumber"], errors="coerce"),
                    "EventNumber": pd.to_numeric(raw["EventNumber"], errors="coerce"),
                    "TimeSinceBlockStart_raw": raw["Time Since Block start"],
                    "TimeSinceBlockStart_s": normalize_decimal_comma(raw["Time Since Block start"]),
                    "isHit": pd.to_numeric(raw["isHit"], errors="coerce"),
                    "target": pd.to_numeric(raw["target"], errors="coerce"),
                    "pressed": pd.to_numeric(raw["pressed"], errors="coerce"),
                    "sequence": raw["sequence"].astype(str).str.strip().str.lower(),
                }
            )
            rows.append(d)

    if not rows:
        raise RuntimeError("No SRT files found.")
    trials = pd.concat(rows, ignore_index=True)
    trials = trials.sort_values(["PID", "day", "BlockNumber", "EventNumber"]).reset_index(drop=True)
    trials["condition"] = np.where(trials["sequence"].isin(["blue", "green"]), "structured", "random")
    return trials


def compute_file_qc(trials: pd.DataFrame) -> pd.DataFrame:
    qc_records: list[dict[str, object]] = []
    for (pid, day, source), g in trials.groupby(["PID", "day", "source_file"], dropna=False):
        g = g.sort_values(["BlockNumber", "EventNumber"]).copy()
        non_monotonic_time = (
            g.groupby("BlockNumber")["TimeSinceBlockStart_s"].diff().fillna(g["TimeSinceBlockStart_s"]) <= 0
        ).sum()
        duplicate_block_event = g.duplicated(subset=["BlockNumber", "EventNumber"]).sum()
        invalid_sequence = (~g["sequence"].isin(EXPECTED_SEQUENCES)).sum()
        invalid_is_hit = (~g["isHit"].isin([0, 1])).sum()
        invalid_target = (~g["target"].isin([1, 2, 3, 4])).sum()
        invalid_pressed = (~g["pressed"].isin([1, 2, 3, 4])).sum()

        qc_records.append(
            {
                "PID": pid,
                "day": day,
                "source_file": source,
                "n_rows": len(g),
                "n_blocks": g["BlockNumber"].nunique(dropna=True),
                "n_events": g["EventNumber"].nunique(dropna=True),
                "min_block": g["BlockNumber"].min(),
                "max_block": g["BlockNumber"].max(),
                "duplicate_block_event": int(duplicate_block_event),
                "non_monotonic_time": int(non_monotonic_time),
                "invalid_sequence": int(invalid_sequence),
                "invalid_is_hit": int(invalid_is_hit),
                "invalid_target": int(invalid_target),
                "invalid_pressed": int(invalid_pressed),
                "is_complete_960": len(g) == 960,
                "is_complete_120_blocks": g["BlockNumber"].nunique(dropna=True) == 120,
            }
        )
    qc = pd.DataFrame(qc_records)
    modal_rows = int(qc["n_rows"].mode().iloc[0]) if not qc.empty else 0
    modal_blocks = int(qc["n_blocks"].mode().iloc[0]) if not qc.empty else 0
    qc["is_complete_modal_rows"] = qc["n_rows"] == modal_rows
    qc["is_complete_modal_blocks"] = qc["n_blocks"] == modal_blocks
    qc["modal_expected_rows"] = modal_rows
    qc["modal_expected_blocks"] = modal_blocks
    return qc


def add_rt_and_outlier_flags(trials: pd.DataFrame) -> pd.DataFrame:
    d = trials.sort_values(["PID", "day", "BlockNumber", "EventNumber"]).copy()
    first_or_diff = d.groupby(["PID", "day", "BlockNumber"])["TimeSinceBlockStart_s"].diff()
    d["RT_s"] = first_or_diff.fillna(d["TimeSinceBlockStart_s"])
    d["RT_ms"] = d["RT_s"] * 1000.0

    d["flag_rt_nonpositive"] = d["RT_ms"].isna() | (d["RT_ms"] <= 0)
    d["flag_rt_hard_outlier"] = (d["RT_ms"] < RT_MIN_MS) | (d["RT_ms"] > RT_MAX_MS)
    d["flag_sequence_invalid"] = ~d["sequence"].isin(EXPECTED_SEQUENCES)
    d["flag_parse_or_reset"] = (
        d.groupby(["PID", "day", "BlockNumber"])["TimeSinceBlockStart_s"].diff().fillna(1.0) <= 0
    )

    # MAD trimming on hit-only, plausible RT trials.
    d["flag_mad_outlier"] = False
    mad_base = d[(d["isHit"] == 1) & (~d["flag_rt_nonpositive"]) & (~d["flag_rt_hard_outlier"])].copy()
    grouped = mad_base.groupby(["PID", "day", "condition"])
    for keys, g in grouped:
        idx = g.index
        med = g["RT_ms"].median()
        mad = np.median(np.abs(g["RT_ms"] - med))
        if np.isnan(mad) or mad == 0:
            continue
        robust_sigma = MAD_SCALE * mad
        outlier_mask = np.abs(g["RT_ms"] - med) > (MAD_MULTIPLIER * robust_sigma)
        d.loc[idx, "flag_mad_outlier"] = outlier_mask.values

    # Winsorized RT for sensitivity.
    d["RT_ms_winsor"] = d["RT_ms"]
    valid = d[(d["isHit"] == 1) & (~d["flag_rt_nonpositive"])].copy()
    for keys, g in valid.groupby(["PID", "day", "condition"]):
        idx = g.index
        wins = mstats.winsorize(g["RT_ms"].to_numpy(), limits=[0.01, 0.01])
        d.loc[idx, "RT_ms_winsor"] = np.asarray(wins)

    d["keep_hard"] = ~(d["flag_rt_nonpositive"] | d["flag_rt_hard_outlier"] | d["flag_parse_or_reset"])
    d["keep_mad"] = d["keep_hard"] & (~d["flag_mad_outlier"])
    return d


def aggregate_metrics(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Block-level from MAD-cleaned trials.
    keep_rt = (trials["isHit"] == 1) & trials["keep_mad"]
    rt_hits = trials[keep_rt].copy()

    block_rt = (
        rt_hits.groupby(["PID", "Group", "day", "BlockNumber", "sequence", "condition"], dropna=False)["RT_ms"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "meanRT_hit_ms",
                "median": "medianRT_hit_ms",
                "std": "sdRT_hit_ms",
                "count": "nHits",
            }
        )
    )
    block_all = (
        trials.groupby(["PID", "Group", "day", "BlockNumber", "sequence", "condition"], dropna=False)
        .agg(nTrials=("isHit", "size"), errorRate=("isHit", lambda x: float((1 - x).mean())))
        .reset_index()
    )
    block_level = block_all.merge(
        block_rt, on=["PID", "Group", "day", "BlockNumber", "sequence", "condition"], how="left"
    )

    # Participant-day metrics.
    participant_day_records: list[dict[str, object]] = []
    for (pid, day), g_day in block_level.groupby(["PID", "day"]):
        group = g_day["Group"].iloc[0]
        rec: dict[str, object] = {"PID": pid, "Group": group, "day": day}
        # General learning from block-averaged RT (all conditions).
        by_block = (
            g_day.groupby("BlockNumber", dropna=False)["meanRT_hit_ms"].mean().dropna().sort_index().reset_index()
        )
        rec["rt_mean_day"] = by_block["meanRT_hit_ms"].mean() if not by_block.empty else np.nan
        rec["rt_early_day"] = by_block[by_block["BlockNumber"] <= 20]["meanRT_hit_ms"].mean()
        rec["rt_late_day"] = by_block[by_block["BlockNumber"] >= 101]["meanRT_hit_ms"].mean()
        rec["RT_Delta_All"] = rec["rt_early_day"] - rec["rt_late_day"]
        if len(by_block) >= 5:
            slope = np.polyfit(by_block["BlockNumber"], by_block["meanRT_hit_ms"], 1)[0]
            rec["RT_Slope_All"] = slope
        else:
            rec["RT_Slope_All"] = np.nan

        # Sequence learning (structured vs random).
        cond_mean = g_day.groupby("condition")["meanRT_hit_ms"].mean()
        rec["meanRT_structured"] = cond_mean.get("structured", np.nan)
        rec["meanRT_random"] = cond_mean.get("random", np.nan)
        rec["SeqLearning_Index_all"] = rec["meanRT_random"] - rec["meanRT_structured"]

        late = g_day[g_day["BlockNumber"] >= 81]
        cond_late = late.groupby("condition")["meanRT_hit_ms"].mean()
        rec["SeqLearning_Index_late"] = cond_late.get("random", np.nan) - cond_late.get("structured", np.nan)

        early = g_day[g_day["BlockNumber"] <= 20]
        cond_early = early.groupby("condition")["meanRT_hit_ms"].mean()
        rec["SeqLearning_Index_early"] = cond_early.get("random", np.nan) - cond_early.get("structured", np.nan)

        # Slope difference by condition.
        slope_by_cond: dict[str, float] = {}
        for cond, g_cond in g_day.groupby("condition"):
            x = g_cond["BlockNumber"].to_numpy()
            y = g_cond["meanRT_hit_ms"].to_numpy()
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() >= 5:
                slope_by_cond[cond] = np.polyfit(x[valid], y[valid], 1)[0]
            else:
                slope_by_cond[cond] = np.nan
        rec["Slope_structured"] = slope_by_cond.get("structured", np.nan)
        rec["Slope_random"] = slope_by_cond.get("random", np.nan)
        rec["SeqLearning_SlopeDiff"] = rec["Slope_random"] - rec["Slope_structured"]

        # Accuracy and IES.
        trial_subset = trials[(trials["PID"] == pid) & (trials["day"] == day)]
        rec["ErrorRate_All"] = float((1 - trial_subset["isHit"]).mean())
        for cond in ["structured", "random"]:
            t = trial_subset[trial_subset["condition"] == cond]
            rec[f"ErrorRate_{cond}"] = float((1 - t["isHit"]).mean()) if len(t) else np.nan
        if rec["ErrorRate_All"] < 1:
            rec["IES"] = np.nanmedian(
                rt_hits[(rt_hits["PID"] == pid) & (rt_hits["day"] == day)]["RT_ms"].to_numpy()
            ) / (1 - rec["ErrorRate_All"])
        else:
            rec["IES"] = np.nan

        participant_day_records.append(rec)

    participant_day = pd.DataFrame(participant_day_records)

    # Retention metrics (participant-level across days).
    p = participant_day.copy()
    d1 = p[p["day"] == 1].set_index("PID")
    d2 = p[p["day"] == 2].set_index("PID")
    common = d1.index.intersection(d2.index)
    retention = pd.DataFrame({"PID": common})
    retention["Group"] = d1.loc[common, "Group"].to_numpy()
    retention["Retention_General"] = d2.loc[common, "rt_early_day"].to_numpy() - d1.loc[common, "rt_late_day"].to_numpy()
    retention["Retention_Sequence"] = (
        d2.loc[common, "SeqLearning_Index_early"].to_numpy() - d1.loc[common, "SeqLearning_Index_late"].to_numpy()
    )
    return block_level, participant_day, retention


def merge_covariates(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    return df.merge(meta, on=["PID", "Group"], how="left")


def _fit_ols(data: pd.DataFrame, formula: str, outcome: str) -> ModelResult | None:
    vars_in_formula = [v for v in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula) if v != "C"]
    vars_in_formula = sorted(set(vars_in_formula))
    if outcome not in data.columns:
        return None
    use_cols = [c for c in vars_in_formula if c in data.columns]
    work = data[use_cols].copy()
    work = work.dropna()
    if len(work) < 15:
        return None
    model = smf.ols(formula, data=work).fit(cov_type="HC3")
    coef = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    return ModelResult(
        outcome=outcome,
        n=len(work),
        formula=formula,
        summary_text=str(model.summary()),
        coef_table=coef,
    )


def run_glm_models(participant_day_cov: pd.DataFrame, retention_cov: pd.DataFrame) -> tuple[pd.DataFrame, list[ModelResult]]:
    results: list[ModelResult] = []

    # Day-specific outcomes (long format with Group*day).
    long_outcomes = [
        "SeqLearning_Index_all",
        "RT_Slope_All",
        "RT_Delta_All",
        "ErrorRate_All",
        "IES",
    ]
    covars = " + ".join(NUMERIC_COVARS)
    for outcome in long_outcomes:
        formula = f"{outcome} ~ C(Group) * C(day) + {covars}"
        res = _fit_ols(participant_day_cov, formula, outcome)
        if res:
            results.append(res)

    # Retention outcomes.
    for outcome in ["Retention_General", "Retention_Sequence"]:
        formula = f"{outcome} ~ C(Group) + {covars}"
        res = _fit_ols(retention_cov, formula, outcome)
        if res:
            results.append(res)

    rows = []
    for res in results:
        rows.append({"outcome": res.outcome, "n": res.n, "formula": res.formula})
    return pd.DataFrame(rows), results


def run_mixed_model(block_cov: pd.DataFrame) -> tuple[pd.DataFrame, MixedLMResults | None]:
    d = block_cov.copy()
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
            *NUMERIC_COVARS,
        ]
    )
    if len(d) < 300:
        return pd.DataFrame(), None

    covars = " + ".join(NUMERIC_COVARS)
    formula = f"logRT ~ C(Group) * C(day) * C(condition) + BlockNumber + {covars}"
    model = smf.mixedlm(formula, data=d, groups=d["PID"], re_formula="1")
    fit = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
    coef_table = fit.summary().tables[1]
    if isinstance(coef_table, pd.DataFrame):
        coef = coef_table.reset_index().rename(columns={"index": "term"})
    else:
        coef = pd.DataFrame(coef_table.data[1:], columns=coef_table.data[0])
    return coef, fit


def save_model_outputs(glm_index: pd.DataFrame, glm_models: list[ModelResult], mixed_coef: pd.DataFrame, mixed_fit) -> None:
    models_dir = DIRS["models"]
    glm_index.to_csv(models_dir / "glm_model_index.csv", index=False)
    coef_rows = []
    for model in glm_models:
        model.coef_table.to_csv(models_dir / f"glm_coef_{model.outcome}.csv", index=False)
        with open(models_dir / f"glm_summary_{model.outcome}.txt", "w", encoding="utf-8") as f:
            f.write(model.summary_text)
        focus = model.coef_table[model.coef_table["term"].str.contains("C\\(Group\\)", na=False)].copy()
        if not focus.empty:
            focus["outcome"] = model.outcome
            coef_rows.append(focus)
    if coef_rows:
        pd.concat(coef_rows, ignore_index=True).to_csv(models_dir / "glm_group_effects.csv", index=False)

    if mixed_fit is not None:
        mixed_coef.to_csv(models_dir / "mixedlm_coef.csv", index=False)
        with open(models_dir / "mixedlm_summary.txt", "w", encoding="utf-8") as f:
            f.write(str(mixed_fit.summary()))


def make_figures(
    trials: pd.DataFrame,
    participant_day: pd.DataFrame,
    retention: pd.DataFrame,
    block_level: pd.DataFrame,
) -> None:
    fig_dir = DIRS["figures"]
    sns.set_theme(style="whitegrid")

    # QC RT distribution pre/post.
    plt.figure(figsize=(10, 5))
    sns.histplot(trials["RT_ms"], bins=100, color="#888888", stat="density", label="raw", alpha=0.5)
    sns.histplot(
        trials.loc[(trials["isHit"] == 1) & trials["keep_mad"], "RT_ms"],
        bins=100,
        color="#2f7ed8",
        stat="density",
        label="clean_hit_mad",
        alpha=0.5,
    )
    plt.xlim(0, 4000)
    plt.xlabel("RT (ms)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "qc_rt_distribution_before_after.png", dpi=200)
    plt.close()

    # Trimmed trial share.
    trim = (
        trials[trials["isHit"] == 1]
        .groupby(["PID", "Group"])
        .agg(total_hits=("isHit", "size"), trimmed=("keep_mad", lambda x: int((~x).sum())))
        .reset_index()
    )
    trim["trimmed_share"] = trim["trimmed"] / trim["total_hits"]
    plt.figure(figsize=(10, 5))
    sns.barplot(data=trim.sort_values("trimmed_share", ascending=False), x="PID", y="trimmed_share", hue="Group")
    plt.xticks(rotation=90)
    plt.ylabel("Trimmed Hit Trial Share")
    plt.tight_layout()
    plt.savefig(fig_dir / "qc_trimmed_share_per_pid.png", dpi=200)
    plt.close()

    # Block RT trajectory with CI.
    traj = (
        block_level.groupby(["Group", "day", "condition", "BlockNumber"])["meanRT_hit_ms"].mean().reset_index()
    )
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=traj,
        x="BlockNumber",
        y="meanRT_hit_ms",
        hue="condition",
        style="Group",
        units="day",
        estimator=None,
        alpha=0.35,
    )
    agg = (
        block_level.groupby(["Group", "condition", "BlockNumber"])["meanRT_hit_ms"]
        .agg(["mean", "sem"])
        .reset_index()
        .dropna()
    )
    for (grp, cond), g in agg.groupby(["Group", "condition"]):
        plt.plot(g["BlockNumber"], g["mean"], label=f"{grp}-{cond}")
        plt.fill_between(g["BlockNumber"], g["mean"] - 1.96 * g["sem"], g["mean"] + 1.96 * g["sem"], alpha=0.15)
    plt.legend(ncol=2, fontsize=8)
    plt.ylabel("Mean RT hit (ms)")
    plt.tight_layout()
    plt.savefig(fig_dir / "qc_block_trajectory_group_condition_ci.png", dpi=200)
    plt.close()

    # Primary plots.
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=participant_day, x="day", y="SeqLearning_Index_all", hue="Group")
    sns.stripplot(data=participant_day, x="day", y="SeqLearning_Index_all", hue="Group", dodge=True, color="black", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_dir / "primary_seq_learning_by_day_group.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=participant_day, x="day", y="RT_Slope_All", hue="Group")
    sns.stripplot(data=participant_day, x="day", y="RT_Slope_All", hue="Group", dodge=True, color="black", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_dir / "primary_rt_slope_by_day_group.png", dpi=200)
    plt.close()

    ret_long = retention.melt(id_vars=["PID", "Group"], value_vars=["Retention_General", "Retention_Sequence"])
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=ret_long, x="variable", y="value", hue="Group")
    sns.stripplot(data=ret_long, x="variable", y="value", hue="Group", dodge=True, color="black", alpha=0.4)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(fig_dir / "primary_retention_by_group.png", dpi=200)
    plt.close()


def summarise_inventory(inventory: pd.DataFrame) -> dict[str, object]:
    per_pid = inventory.groupby("PID").agg(
        has_day1=("day", lambda x: int((x == 1).any())),
        has_day2=("day", lambda x: int((x == 2).any())),
        both=("has_both_days", "max"),
    )
    return {
        "n_pid_total": int(per_pid.shape[0]),
        "n_with_both_days": int((per_pid["both"] == True).sum()),  # noqa: E712
        "n_missing_any_day": int((per_pid["both"] == False).sum()),  # noqa: E712
        "n_day1_files": int(inventory[inventory["day"] == 1]["selected_file"].notna().sum()),
        "n_day2_files": int(inventory[inventory["day"] == 2]["selected_file"].notna().sum()),
    }


def write_report(
    inventory: pd.DataFrame,
    file_qc: pd.DataFrame,
    trials: pd.DataFrame,
    participant_day: pd.DataFrame,
    retention: pd.DataFrame,
    glm_index: pd.DataFrame,
) -> None:
    report_path = OUT_DIR / "report.md"
    inv = summarise_inventory(inventory)
    qc_counts = {
        "flag_rt_nonpositive": int(trials["flag_rt_nonpositive"].sum()),
        "flag_rt_hard_outlier": int(trials["flag_rt_hard_outlier"].sum()),
        "flag_mad_outlier": int(trials["flag_mad_outlier"].sum()),
        "keep_mad_hit_trials": int(((trials["isHit"] == 1) & trials["keep_mad"]).sum()),
        "all_hit_trials": int((trials["isHit"] == 1).sum()),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SRTT Analysis Report\n\n")
        f.write("## 0) Manual exclusions\n")
        f.write(
            "- Excluded PIDs (edit in `analysis/analysis_config.py`): "
            + ", ".join(str(x) for x in EXCLUDED_PIDS)
            + "\n\n"
        )
        f.write("## 1) Data inventory and mapping\n")
        f.write(f"- Total participants in metadata: **{inv['n_pid_total']}**\n")
        f.write(f"- Participants with both day files: **{inv['n_with_both_days']}**\n")
        f.write(f"- Participants missing any day file: **{inv['n_missing_any_day']}**\n")
        f.write(f"- Day-1 files found: **{inv['n_day1_files']}**\n")
        f.write(f"- Day-2 files found: **{inv['n_day2_files']}**\n\n")

        modal_rows = int(file_qc["modal_expected_rows"].iloc[0]) if len(file_qc) else 0
        modal_blocks = int(file_qc["modal_expected_blocks"].iloc[0]) if len(file_qc) else 0
        f.write("## 2) Quality control\n")
        f.write(
            f"- Sessions with canonical modal length ({modal_rows} rows): "
            f"**{int(file_qc['is_complete_modal_rows'].sum())}** / {len(file_qc)}\n"
        )
        f.write(
            f"- Sessions with canonical modal block count ({modal_blocks} blocks): "
            f"**{int(file_qc['is_complete_modal_blocks'].sum())}** / {len(file_qc)}\n"
        )
        f.write(f"- Sessions with exact 960 rows (PDF expectation): **{int(file_qc['is_complete_960'].sum())}** / {len(file_qc)}\n")
        f.write(f"- Sessions with exact 120 blocks (PDF expectation): **{int(file_qc['is_complete_120_blocks'].sum())}** / {len(file_qc)}\n")
        f.write(f"- Non-positive/invalid RT trials: **{qc_counts['flag_rt_nonpositive']}**\n")
        f.write(f"- Hard RT outliers ({RT_MIN_MS:.0f}-{RT_MAX_MS:.0f} ms): **{qc_counts['flag_rt_hard_outlier']}**\n")
        f.write(f"- MAD outliers: **{qc_counts['flag_mad_outlier']}**\n")
        f.write(
            f"- Kept hit-trials for primary RT analysis: **{qc_counts['keep_mad_hit_trials']} / {qc_counts['all_hit_trials']}**\n\n"
        )

        f.write("## 3) Derived metrics\n")
        f.write(f"- Participant-day rows: **{len(participant_day)}**\n")
        f.write(f"- Retention rows (participants with both days): **{len(retention)}**\n")
        f.write("- Main metrics: `RT_Slope_All`, `RT_Delta_All`, `SeqLearning_Index_all`, `ErrorRate_All`, `IES`\n")
        f.write("- Retention metrics: `Retention_General`, `Retention_Sequence`\n\n")

        f.write("## 4) Models\n")
        if glm_index.empty:
            f.write("- No GLM could be estimated with sufficient complete cases.\n\n")
        else:
            f.write("- Estimated GLM models:\n")
            for _, row in glm_index.iterrows():
                f.write(f"  - `{row['outcome']}` (n={int(row['n'])})\n")
            f.write("\n")
        f.write("- Mixed model output is provided in `analysis/outputs/03_models/mixedlm_summary.txt` if convergence succeeded.\n\n")

        f.write("## 5) Output files\n")
        f.write("- `analysis/outputs/01_ingest_and_qc/*`\n")
        f.write("- `analysis/outputs/02_metrics/*`\n")
        f.write("- `analysis/outputs/03_models/*`\n")
        f.write("- `analysis/outputs/04_figures/*`\n")


def save_data_products(
    inventory: pd.DataFrame,
    file_qc: pd.DataFrame,
    trials: pd.DataFrame,
    block_level: pd.DataFrame,
    participant_day: pd.DataFrame,
    retention: pd.DataFrame,
) -> None:
    inv_dir = DIRS["inventory"]
    metrics_dir = DIRS["metrics"]

    pd.DataFrame({"PID": EXCLUDED_PIDS}).to_csv(inv_dir / "excluded_pids.csv", index=False)
    inventory.to_csv(inv_dir / "inventory_pid_day_files.csv", index=False)
    file_qc.to_csv(inv_dir / "file_qc_flags.csv", index=False)
    trials.to_parquet(inv_dir / "trial_level_all.parquet", index=False)
    trials.to_csv(inv_dir / "trial_level_all.csv", index=False)

    block_level.to_parquet(metrics_dir / "block_level_metrics.parquet", index=False)
    block_level.to_csv(metrics_dir / "block_level_metrics.csv", index=False)
    participant_day.to_csv(metrics_dir / "participant_day_metrics.csv", index=False)
    retention.to_csv(metrics_dir / "participant_retention_metrics.csv", index=False)

    # Sensitivity datasets.
    hard_clean = trials[(trials["isHit"] == 1) & trials["keep_hard"]].copy()
    mad_clean = trials[(trials["isHit"] == 1) & trials["keep_mad"]].copy()
    wins = trials[(trials["isHit"] == 1) & (~trials["flag_rt_nonpositive"])].copy()
    hard_clean.to_csv(metrics_dir / "sensitivity_rt_hits_hard.csv", index=False)
    mad_clean.to_csv(metrics_dir / "sensitivity_rt_hits_mad.csv", index=False)
    wins.to_csv(metrics_dir / "sensitivity_rt_hits_winsor.csv", index=False)


def main() -> None:
    ensure_dirs()
    meta = load_metadata()
    inventory, selected = build_inventory(meta)
    trials = load_all_trials(meta, selected)
    file_qc = compute_file_qc(trials)
    trials = add_rt_and_outlier_flags(trials)

    block_level, participant_day, retention = aggregate_metrics(trials)

    participant_day_cov = merge_covariates(participant_day, meta)
    retention_cov = merge_covariates(retention, meta)
    block_cov = merge_covariates(block_level, meta)

    glm_index, glm_models = run_glm_models(participant_day_cov, retention_cov)
    mixed_coef, mixed_fit = run_mixed_model(block_cov)

    save_data_products(inventory, file_qc, trials, block_level, participant_day_cov, retention_cov)
    save_model_outputs(glm_index, glm_models, mixed_coef, mixed_fit)
    make_figures(trials, participant_day_cov, retention_cov, block_cov)
    write_report(inventory, file_qc, trials, participant_day_cov, retention_cov, glm_index)

    manifest = {
        "excluded_pids": EXCLUDED_PIDS,
        "inventory_rows": int(len(inventory)),
        "trial_rows": int(len(trials)),
        "block_rows": int(len(block_level)),
        "participant_day_rows": int(len(participant_day_cov)),
        "retention_rows": int(len(retention_cov)),
        "glm_models": int(len(glm_models)),
        "mixed_model_success": mixed_fit is not None,
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Analysis complete.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
