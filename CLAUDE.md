# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Post-stroke rehabilitation research project analyzing implicit motor sequence learning from a Serial Reaction Time Task (SRTT) clinical trial. Compares two intervention groups (VR+SRTT vs SRTT-only) across two training days. The key finding is that groups differ in the **dynamic day-to-day evolution** of sequence-specific performance, not in simple endpoint differences.

## Running the Analysis Pipeline

Python 3.11.7 via pyenv. Activate the virtual environment first:

```bash
source .venv/bin/activate
```

The pipeline consists of four sequential scripts:

```bash
python analysis/run_srtt_analysis.py          # 1. Data ingestion, preprocessing, QC, RT reconstruction
python analysis/compute_mixedlm_emm.py        # 2. Linear mixed model + estimated marginal means
python analysis/run_additional_analyses.py     # 3. Sensitivity, nonlinear, speed-accuracy, penalized regression
python analysis/build_publication_tables.py    # 4. Format results into publication markdown tables
```

No test suite, linter, or build system is configured.

## Architecture

### Data Flow

```
data/01–43/ (raw behavioral CSVs, semicolon-separated, German decimal format)
  → run_srtt_analysis.py (parse, reconstruct trial-wise RT from cumulative timestamps, QC trim)
  → outputs/01_ingest_and_qc/, outputs/02_metrics/
  → compute_mixedlm_emm.py (block-level mixed model: logRT ~ Group × Day × Condition + covariates)
  → outputs/03_models/
  → run_additional_analyses.py (nonlinear curves, permutation tests, penalized regression)
  → outputs/04_figures/, outputs/05_additional_analyses/
  → build_publication_tables.py
  → outputs/06_publication_tables/
```

### Key Design Decisions

- **RT reconstruction**: Raw data uses cumulative within-block timestamps; scripts convert these to trial-wise RTs
- **QC trimming**: Hard limits (150–3000 ms) then robust within-subject trimming (median ± 3×MAD)
- **Primary model**: Block-level log-RT with 3-way interaction (Group × Day × Condition), random intercepts per participant, adjusted for Age, Fugl-Meyer, MoCa
- **Output organization**: Each pipeline stage writes to a numbered subdirectory under `analysis/outputs/`

### Secondary Analysis (R)

`analyse_Alex/` contains an independent R-based analysis pipeline (likely lme4/emmeans) for cross-validation of results.

## Data

- 43 participants, 73 participant-day sessions, 53,033 clean trials after preprocessing
- Metadata in `Datensatz_Round_2.csv`
- Raw behavioral logs in `data/<PID>/` folders (semicolon-separated CSVs with German locale)
