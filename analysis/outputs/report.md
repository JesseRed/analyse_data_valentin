# SRTT Analysis Report

## 1) Data inventory and mapping
- Total participants in metadata: **43**
- Participants with both day files: **34**
- Participants missing any day file: **9**
- Day-1 files found: **36**
- Day-2 files found: **37**

## 2) Quality control
- Sessions with canonical modal length (952 rows): **72** / 73
- Sessions with canonical modal block count (119 blocks): **72** / 73
- Sessions with exact 960 rows (PDF expectation): **0** / 73
- Sessions with exact 120 blocks (PDF expectation): **0** / 73
- Non-positive/invalid RT trials: **38**
- Hard RT outliers (150-3000 ms): **11151**
- MAD outliers: **3360**
- Kept hit-trials for primary RT analysis: **53033 / 66410**

## 3) Derived metrics
- Participant-day rows: **73**
- Retention rows (participants with both days): **34**
- Main metrics: `RT_Slope_All`, `RT_Delta_All`, `SeqLearning_Index_all`, `ErrorRate_All`, `IES`
- Retention metrics: `Retention_General`, `Retention_Sequence`

## 4) Models
- Estimated GLM models:
  - `SeqLearning_Index_all` (n=71)
  - `RT_Slope_All` (n=71)
  - `RT_Delta_All` (n=71)
  - `ErrorRate_All` (n=71)
  - `IES` (n=71)
  - `Retention_General` (n=33)
  - `Retention_Sequence` (n=33)

- Mixed model output is provided in `analysis/outputs/03_models/mixedlm_summary.txt` if convergence succeeded.

## 5) Output files
- `analysis/outputs/01_ingest_and_qc/*`
- `analysis/outputs/02_metrics/*`
- `analysis/outputs/03_models/*`
- `analysis/outputs/04_figures/*`
