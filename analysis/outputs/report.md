# SRTT Analysis Report

## 0) Manual exclusions
- Excluded PIDs (edit in `analysis/analysis_config.py`): 2, 7, 13, 26, 27, 32, 38

## 1) Data inventory and mapping
- Total participants in metadata: **36**
- Participants with both day files: **36**
- Participants missing any day file: **0**
- Day-1 files found: **36**
- Day-2 files found: **36**

## 2) Quality control
- Sessions with canonical modal length (952 rows): **69** / 72
- Sessions with canonical modal block count (119 blocks): **69** / 72
- Sessions with exact 960 rows (PDF expectation): **0** / 72
- Sessions with exact 120 blocks (PDF expectation): **0** / 72
- Non-positive/invalid RT trials: **38**
- Hard RT outliers (150-3000 ms): **11276**
- MAD outliers: **3208**
- Kept hit-trials for primary RT analysis: **52055 / 65401**

## 3) Derived metrics
- Participant-day rows: **72**
- Retention rows (participants with both days): **36**
- Main metrics: `RT_Slope_All`, `RT_Delta_All`, `SeqLearning_Index_all`, `ErrorRate_All`, `IES`
- Retention metrics: `Retention_General`, `Retention_Sequence`

## 4) Models
- Estimated GLM models:
  - `SeqLearning_Index_all` (n=72)
  - `RT_Slope_All` (n=72)
  - `RT_Delta_All` (n=72)
  - `ErrorRate_All` (n=72)
  - `IES` (n=72)
  - `Retention_General` (n=36)
  - `Retention_Sequence` (n=36)

- Mixed model output is provided in `analysis/outputs/03_models/mixedlm_summary.txt` if convergence succeeded.

## 5) Output files
- `analysis/outputs/01_ingest_and_qc/*`
- `analysis/outputs/02_metrics/*`
- `analysis/outputs/03_models/*`
- `analysis/outputs/04_figures/*`
