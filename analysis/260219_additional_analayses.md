# Additional analyses (2026-02-19)

## Data basis
- Included participants (after EXCLUDED_PIDS): **36**; participant-day sessions: **72**; block-level observations: **8615**.
- Primary covariates used: `Age, fuglmayrshort_sum, MoCa_sum`.

## A) Random-effects structure sensitivity
- Criterion: stability of `Group×Day×Condition` estimate and DoD contrast across random-effects specifications.

| model                 | random_effects_spec                                             | converged   | singular_fit   |   n_obs |   three_way_coef |   three_way_se |   three_way_p |   dod_log |    dod_se |       dod_p |   dod_pct_change | fit_error   |
|:----------------------|:----------------------------------------------------------------|:------------|:---------------|--------:|-----------------:|---------------:|--------------:|----------:|----------:|------------:|-----------------:|:------------|
| A1_baseline           | (1|PID)                                                         | True        | False          |    8615 |        0.0515314 |      0.0166726 |   0.00199629  | 0.0515314 | 0.0166726 | 0.00199629  |        0.0528823 |             |
| A2_time_on_task_slope | (1 + BlockNumber|PID)                                           | True        | True           |    8615 |        0.0508491 |      0.0164746 |   0.00202514  | 0.0508491 | 0.0164746 | 0.00202514  |        0.0521641 |             |
| A3_day_slope          | (1 + Day|PID)                                                   | True        | False          |    8615 |        0.0508214 |      0.0155875 |   0.00111258  | 0.0508214 | 0.0155875 | 0.00111258  |        0.052135  |             |
| A4_condition_slope    | (1 + Condition|PID)                                             | False       | False          |    8615 |        0.0523365 |      0.01655   |   0.00156509  | 0.0523365 | 0.01655   | 0.00156509  |        0.0537303 |             |
| A5_max_sensible       | (1|PID) + (0+BlockNumber|PID) + (0+Day|PID) + (0+Condition|PID) | True        | False          |    8615 |        0.0506894 |      0.0152467 |   0.000885408 | 0.0506894 | 0.0152467 | 0.000885408 |        0.0519961 |             |

## B) Seriality / robust inference checks

### B1. Residual seriality (baseline mixed model)
- Median session ACF(1): **0.178** (IQR **0.033 to 0.266**).
- Proportion of sessions with ACF(1) > 0.2: **45.8%**.

### B2. Cluster-robust OLS (participant-clustered SE)
- 3-way term: coef **0.0488**, SE **0.0191**, p **0.0107**; DoD ≈ **5.00%**.

### B3. Aggregation sensitivity
| analysis       |   n_obs |      coef |        se |        p | fit_error   |
|:---------------|--------:|----------:|----------:|---------:|:------------|
| B3_chunk_5     |    3054 | 0.0452292 | 0.0200687 | 0.024214 |             |
| B3_chunk_10    |    1745 | 0.0361382 | 0.0233381 | 0.121511 |             |
| B3_phase_3bins |     432 | 0.0469896 | 0.0332576 | 0.157685 |             |

### B4. AR(1) working-correlation model (GEE)
- Covariance structure: **AR(1)**; 3-way term: coef **0.0595**, SE **0.0294**, p **0.0431**; DoD ≈ **6.13%**; AR(1) phi ≈ **0.822**.

### B5. Session-level random intercept sensitivity
- 3-way term: coef **0.0508**, SE **0.0156**, p **0.0011**; DoD ≈ **5.21%**.

## C) Inference strategy checks
### C1. LRT for the 3-way fixed effect (baseline random-intercept mixed model)
- LR = **9.548**, df = **1**, p = **0.00200**.

### C2. Planned contrast focus (DoD)
- Baseline planned DoD (structured-random day-change, B vs A): log-DoD **0.0515**, SE **0.0167**, p **0.00200**, ≈ **5.29%**.
- Interpretation: contrast-focused inference is directionally stable across robustness models.

## Quick conclusion
- The primary dynamic signal (`Group×Day×Condition`) remains directionally stable across richer random-effects structures, serial-correlation diagnostics, clustered-SE inference, and AR(1) sensitivity modeling.