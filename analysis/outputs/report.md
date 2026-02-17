# SRTT Analysis Report

## 0) Manual exclusions
- Excluded PIDs (edit in `analysis/analysis_config.py`): 2, 7, 13, 26, 27, 32, 38

## 1) Data inventory and mapping
- Total participants in metadata: **36**
- Participants with both day files: **35**
- Participants missing any day file: **1**
- Day-1 files found: **36**
- Day-2 files found: **35**

## 2) Quality control
- Sessions with canonical modal length (952 rows): **69** / 71
- Sessions with canonical modal block count (119 blocks): **69** / 71
- Sessions with exact 960 rows (PDF expectation): **0** / 71
- Sessions with exact 120 blocks (PDF expectation): **0** / 71
- Non-positive/invalid RT trials: **38**
- Hard RT outliers (150-3000 ms): **10886**
- MAD outliers: **3194**
- Kept hit-trials for primary RT analysis: **51537 / 64509**

## 3) Derived metrics
- Participant-day rows: **71**
- Retention rows (participants with both days): **35**
- Main metrics: `RT_Slope_All`, `RT_Delta_All`, `SeqLearning_Index_all`, `ErrorRate_All`, `IES`
- Retention metrics: `Retention_General`, `Retention_Sequence`

## 4) Models
- Estimated GLM models:
  - `SeqLearning_Index_all` (n=69)
  - `RT_Slope_All` (n=69)
  - `RT_Delta_All` (n=69)
  - `ErrorRate_All` (n=69)
  - `IES` (n=69)
  - `Retention_General` (n=34)
  - `Retention_Sequence` (n=34)

- Mixed model output is provided in `analysis/outputs/03_models/mixedlm_summary.txt` if convergence succeeded.

## 5) Output files
- `analysis/outputs/01_ingest_and_qc/*`
- `analysis/outputs/02_metrics/*`
- `analysis/outputs/03_models/*`
- `analysis/outputs/04_figures/*`
