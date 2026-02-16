## Results

### Sample, data availability, and preprocessing
Data from **43 participants** were available in the metadata table (`Datensatz_Round_2.csv`). Based on file mapping (`data/<PID2>/*_1_fertig.csv`, `*_2_fertig.csv`), **36 day-1 files** and **37 day-2 files** were found, yielding **73 participant-day sessions**. **34/43 participants** had data on both days (used for retention analyses).

Reaction times (RT) were reconstructed trial-wise from the cumulative time stamp (“Time Since Block start”) by within-block differencing. Analyses focused primarily on correct trials (`isHit==1`). Outlier handling followed a data-driven pipeline with (i) hard plausibility limits (**150–3000 ms**), (ii) robust within-subject trimming using the **median ± 3×MAD** rule (within PID×day×condition), and (iii) a winsorized RT variant for sensitivity outputs. In total, **53,033 / 66,410** correct trials were retained for the primary RT analyses after MAD-trimming (see `analysis/outputs/report.md` and QC figures in `analysis/outputs/04_figures/`).

### Descriptive performance and learning metrics
Learning and performance were summarized per participant-day (n=73) using the pre-specified metrics: general learning (`RT_Slope_All`, `RT_Delta_All`), sequence learning (`SeqLearning_Index_all` = meanRT_random − meanRT_structured), accuracy (`ErrorRate_All`), and the inverse efficiency score (`IES`).

Group counts per day were: **day 1** A=18, B=18; **day 2** A=21, B=16 (see `analysis/outputs/02_metrics/participant_day_metrics.csv`). Key descriptives are summarized below (mean ± SD):

- **SeqLearning_Index_all (ms)**:
  - Day 1: A **52.5 ± 48.0**, B **86.5 ± 56.4** (unadjusted B−A: **+33.9 ms**)
  - Day 2: A **78.6 ± 54.6**, B **69.0 ± 60.9** (unadjusted B−A: **−9.7 ms**)
- **General learning (RT_Slope_All, ms/block; negative = faster over blocks)**:
  - Day 1: A **−1.45 ± 1.16**, B **−1.73 ± 1.26**
  - Day 2: A **−0.71 ± 1.12**, B **−0.72 ± 0.82**
- **General learning (RT_Delta_All = early−late, ms; positive = improvement)**:
  - Day 1: A **162.9 ± 136.7**, B **185.6 ± 105.3**
  - Day 2: A **57.2 ± 106.1**, B **67.1 ± 71.7**
- **Accuracy (ErrorRate_All)**:
  - Day 1: A **0.051 ± 0.040**, B **0.075 ± 0.120**
  - Day 2: A **0.041 ± 0.030**, B **0.045 ± 0.083**

Retention metrics were computed for participants with both days (n=34; A=18, B=16; `analysis/outputs/02_metrics/participant_retention_metrics.csv`):

- **Retention_General (ms)**: A **−35.6 ± 155.1**, B **−4.5 ± 107.0**
- **Retention_Sequence (ms)**: A **−17.4 ± 131.8**, B **−15.2 ± 115.4**

### Primary group comparisons (GLM; participant-day level)
Group differences were tested using heteroscedasticity-robust OLS models (HC3) with the planned covariates (AES_sum, Age, Gender, Depression, SportsActivity, Fugl-Meyer, EQ5D, GDS, MoCa, MORE, TSS, NIHSS). For day-dependent outcomes, models included `Group × day`. Model outputs are stored in `analysis/outputs/03_models/`.

#### Sequence learning (SeqLearning_Index_all)
In the adjusted model (`SeqLearning_Index_all ~ Group*day + covariates`, n=67 complete cases), the main contrast for **Group B vs A** at **day 1** was **+37.4 ms** (95% CI −5.3 to 80.1; **p=0.086**). The **Group×day interaction** was **−50.5 ms** (95% CI −110.1 to 9.1; **p=0.097**), indicating that the between-group difference was smaller (and reversed direction) on day 2 (i.e., estimated B−A at day 2 ≈ 37.4 − 50.5 = **−13.1 ms**). These effects did not reach conventional significance in this sample (see `glm_summary_SeqLearning_Index_all.txt` and `glm_group_effects.csv`).

Covariate associations in this model suggested that higher **AES_sum** was associated with larger SeqLearning_Index_all (coef **+4.68 ms** per AES point, p<0.001), whereas higher **GDS_sum** was associated with smaller SeqLearning_Index_all (coef **−12.43 ms**, p=0.005). The model condition number was high, consistent with potential multicollinearity among covariates; results should be interpreted with appropriate caution.

#### General learning, accuracy, and efficiency
For general learning (`RT_Slope_All`, `RT_Delta_All`), as well as `ErrorRate_All` and `IES`, adjusted Group and Group×day effects were small and statistically non-significant (see `analysis/outputs/03_models/glm_group_effects.csv` and the respective `glm_summary_*.txt` files).

#### Retention (participants with both days)
Retention outcomes were modeled with `Retention ~ Group + covariates` (n=31 complete cases). **Group B vs A** effects were not statistically significant for either **Retention_General** (coef +55.8 ms; p=0.320) or **Retention_Sequence** (coef +53.4 ms; p=0.460). In the retention model for sequence learning, **SportsActivity** (numeric coding) showed a negative association with Retention_Sequence (coef −65.5; p=0.033), while other covariates were not clearly associated (see `glm_summary_Retention_Sequence.txt`).

### Secondary analysis (mixed model; block-level RT)
A linear mixed model on block-level log-RT (`logRT`) was fitted to quantify within-subject effects across **day** and **condition** (structured vs random), with a participant-level random intercept and a reduced covariate set for stability (Age, AES_sum, MoCa_sum, NIHSS, TSS). The model converged (8,506 observations; 38 participants; `analysis/outputs/03_models/mixedlm_summary.txt`).

Across groups, **day 2** showed faster performance than day 1 (coef −0.109 on log scale; p<0.001), and **structured blocks** were faster than random (coef −0.057; p<0.001; corresponding to ~**−5.5%** RT). Importantly, a significant **Group×day×condition** interaction was observed (coef **+0.046**, p=0.005; ~**+4.7%** on the RT scale), indicating that the *change in the structured-vs-random RT difference from day 1 to day 2* differed between groups. This interaction supports the presence of group-dependent sequence-specific dynamics at the block level, even though participant-level GLM contrasts of aggregated SeqLearning_Index_all were only trend-level.

#### Planned contrasts from the mixed model (estimated marginal means)
To make the mixed-model interaction interpretable, we computed estimated marginal means (EMMs) for each `Group × day × condition` combination and the corresponding planned contrasts using the fixed-effect variance–covariance matrix (delta method). EMMs were evaluated at the **mean BlockNumber** and **mean covariate values** of the analyzed block-level dataset. Full tables are provided in:

- `analysis/outputs/03_models/mixedlm_marginal_means.csv`
- `analysis/outputs/03_models/mixedlm_contrasts.csv`

Key contrasts (reported as **RT ratio** and **percent change** for structured vs random; negative values indicate faster RT in structured blocks):

- **Structured vs random within each group/day**:
  - Group A, day 1: ratio **0.944** (95% CI 0.929–0.960), **−5.6%** (−7.1% to −4.0%)
  - Group A, day 2: ratio **0.907** (0.893–0.920), **−9.3%** (−10.7% to −8.0%)
  - Group B, day 1: ratio **0.919** (0.905–0.933), **−8.1%** (−9.5% to −6.7%)
  - Group B, day 2: ratio **0.924** (0.908–0.940), **−7.6%** (−9.2% to −6.0%)

- **Difference-in-differences (3-way simple contrast)**: the change in (structured−random) from day 1 to day 2 differed between groups by a factor of **1.047** (95% CI 1.014–1.082), i.e. **+4.7%** (1.4% to 8.2%) on the RT scale. This quantifies the significant `Group×day×condition` interaction as a group difference in how the structured–random separation evolved from day 1 to day 2.

- **Group differences (B−A) within each day/condition**: cell-wise EMM contrasts were small and imprecise (all 95% CIs included 1.0; see `mixedlm_contrasts.csv`), indicating that the most robust signal was the **differential change over day × condition** rather than a consistent overall B–A shift in a single cell.

### Interpretation with respect to the primary question (Group A vs B)
Taken together:

- On **day 1**, group B tended to show a **larger sequence-learning index** (random−structured) than group A (trend-level in adjusted GLM; consistent direction in unadjusted means).
- On **day 2**, this difference **attenuated or reversed** (negative Group×day interaction trend in GLM; unadjusted mean difference B−A was slightly negative).
- For **general learning** (overall RT improvement across blocks) and **retention** metrics, there was **no strong evidence** for systematic between-group differences in the current sample after adjustment.
- The **block-level mixed model** indicated **group-dependent day-by-condition effects**, suggesting that more fine-grained hierarchical modeling may be more sensitive to group differences than participant-level aggregation alone.

### Additional analyses completed (requested extensions)
All requested follow-up analyses were executed and exported under `analysis/outputs/05_additional_analyses/` (script: `analysis/run_additional_analyses.py`).

1. **Planned contrasts (mixed-model simple effects)**  
   Recomputed and exported in `planned_contrasts_simple_effects.csv` and `planned_contrasts_emm.csv`. Results confirmed robust structured-vs-random effects in each group/day (all CIs below 0 on log scale), while B-vs-A cell contrasts remained imprecise.

2. **Non-linear learning curves**  
   A spline model (`bs(BlockNumber, df=4)`) was compared to a linear block model. The spline model improved fit:
   - AIC: **4169.66** (spline) vs **4175.07** (linear)
   - Likelihood-ratio test: **LLR=19.40**, df=7, **p=0.007**  
   (files: `nonlinear_model_comparison.csv`, `nonlinear_llr_test.csv`).

3. **Blue vs green analyzed separately**  
   In `blue_green_contrasts.csv`, both blue and green were faster than yellow across groups/days (all CIs below 0). Example:
   - Group A day 2: blue−yellow **−10.2%**, green−yellow **−7.6%**
   - Group B day 1: blue−yellow **−7.6%**, green−yellow **−9.1%**  
   This supports sequence-specific facilitation for both repeating sequence types.

4. **Speed–accuracy trade-off analyses**  
   - Block-level RT model with error-rate moderation (`speed_accuracy_tradeoff_key_terms.csv`):
     - `errorRate` coef **+0.186**, **p=0.00061**
     - `GroupB:errorRate` coef **−0.271**, **p=0.00012**  
     indicating that the RT penalty with increasing error rate differs by group.
   - Trial-level hit model (`speed_accuracy_hit_model_key_terms.csv`):
     - `GroupB:zRT` **p=0.0016**
     - `Day2:zRT` **p=0.016**  
     consistent with condition-dependent speed–accuracy coupling.

5. **Robust / penalized regression for stability**  
   - Robust regression (Huber RLM, day-2 sequence-learning outcome): Group effect remained small/non-significant (`C(Group)[T.B]` p=0.746).
   - Ridge/Lasso (`penalized_regression_coefficients.csv`): Lasso shrank all coefficients to ~0 in this sample, indicating weak stable sparse signal under penalization and limited power for many predictors.

6. **Missingness and data-quality sensitivity**  
   Primary group coefficients were stable across:
   - baseline
   - truncation to first 119 blocks
   - truncation to first 120 blocks
   - exclusion of the single long session (`PID 9, day 1`)  
   (`missingness_sensitivity.csv`). For SeqLearning day-wise group effect, estimates stayed around **+38 to +39 ms** with p-values ~0.07–0.09.

7. **Predictor interaction screening (Group×predictor)**  
   Screening of `Group×{AES, MoCa, GDS, NIHSS, TSS, MORE}` for day-2 SeqLearning found no robust interactions after FDR correction (all `p_fdr` ≥ 0.905; `predictor_interaction_screening.csv`).

8. **Clinically meaningful stratification**  
   Stratified models by MoCa/GDS/Fugl-Meyer thresholds did not show significant B-vs-A effects in available strata (all p>0.49; `clinical_stratification_group_effects.csv`). Confidence intervals were wide, suggesting limited subgroup precision.

9. **Permutation tests (confirmatory robustness)**  
   Two-sided permutation tests (10,000 permutations; `permutation_tests.csv`) supported the null for unadjusted group differences:
   - SeqLearning day 2: observed B−A = **−9.66 ms**, **p=0.611**
   - Retention_Sequence: observed B−A = **+2.18 ms**, **p=0.961**

Overall, these extensions support the core conclusion: evidence for group differences is strongest for **how sequence-specific effects evolve over day × condition** (mixed-model interaction structure), while direct B-vs-A differences on aggregated endpoint summaries remain small/uncertain in the current sample.

