## Table 1. Baseline demographic characteristics (included sample)

| Characteristic         | Group A (n=19)   | Group B (n=17)   | Total (n=36)   | p value   |
|:-----------------------|:-----------------|:-----------------|:---------------|:----------|
| Age, years             | 66.3 ± 8.8       | 67.2 ± 8.4       | 66.8 ± 8.5     | 0.751     |
| Age range              | 50–84            | 54–89            | 50–89          |           |
| Sex (Female)           | 2 (10.5%)        | 7 (41.2%)        | 9 (25.0%)      | 0.055     |
| Sex (Male)             | 17 (89.5%)       | 10 (58.8%)       | 27 (75.0%)     |           |
| Education              |                  |                  |                | 0.498     |
| – School               | 6 (31.6%)        | 7 (41.2%)        | 13 (36.1%)     |           |
| – Apprenticeship       | 5 (26.3%)        | 6 (35.3%)        | 11 (30.6%)     |           |
| – University degree    | 8 (42.1%)        | 4 (23.5%)        | 12 (33.3%)     |           |
| Body mass index, kg/m² | 26.8 ± 3.1       | 27.3 ± 5.5       | 27.0 ± 4.3     | 0.742     |
| NIHSS                  | 2.1 ± 1.6        | 2.9 ± 1.8        | 2.5 ± 1.7      | 0.086     |
| NIHSS range            | 0–6              | 0–6              | 0–6            |           |
| S-FMA (upper limb)     | 11.3 ± 1.5       | 10.4 ± 2.2       | 10.8 ± 1.9     | 0.136     |
| S-FMA range            | 6–12             | 5–12             | 5–12           |           |
| MoCA                   | 24.0 ± 4.4       | 23.8 ± 3.2       | 23.9 ± 3.8     | 0.632     |
| MoCA range             | 17–30            | 16–29            | 16–30          |           |

*Values are mean ± SD unless otherwise stated. p values from Mann–Whitney U test (continuous variables) or Fisher's exact test (categorical variables). NIHSS = NIH Stroke Scale; S-FMA = Fugl-Meyer Assessment upper limb short form (max. 12); MoCA = Montreal Cognitive Assessment (max. 30).*

## Table 2. Primary group effects (adjusted GLM, HC3)

| Outcome               |   Group effect B vs A (coef) | Group effect 95% CI   |   Group effect p |   Group×Day interaction (coef) | Group×Day 95% CI    | Group×Day p   |
|:----------------------|-----------------------------:|:----------------------|-----------------:|-------------------------------:|:--------------------|:--------------|
| ErrorRate_All         |                    0.0139408 | [-0.034, 0.061]       |            0.565 |                     -0.0213997 | [-0.090, 0.047]     | 0.542         |
| IES                   |                   59.313     | [-164.317, 282.943]   |            0.603 |                      6.08168   | [-304.967, 317.130] | 0.969         |
| RT_Delta_All          |                   44.0083    | [-55.734, 143.751]    |            0.387 |                    -49.4608    | [-164.883, 65.962]  | 0.401         |
| RT_Slope_All          |                   -0.35715   | [-1.255, 0.540]       |            0.435 |                      0.60301   | [-0.569, 1.775]     | 0.313         |
| Retention_General     |                   81.5545    | [-40.169, 203.278]    |            0.189 |                    nan         | NA                  | NA            |
| Retention_Sequence    |                   32.3117    | [-46.080, 110.703]    |            0.419 |                    nan         | NA                  | NA            |
| SeqLearning_Index_all |                   39.8833    | [5.225, 74.542]       |            0.024 |                    -48.9407    | [-101.125, 3.244]   | 0.066         |


## Table 3. Mixed-model simple effects (ratio scale)

| Effect                  | Group   | Day   | Condition   |   log-diff |         SE | ratio_fmt            | pct_fmt                    |
|:------------------------|:--------|:------|:------------|-----------:|-----------:|:---------------------|:---------------------------|
| structured_minus_random | A       | 1.0   | -           | -0.050138  | 0.00820237 | 0.951 [0.936, 0.967] | -4.89% [-6.41%, -3.35%]    |
| structured_minus_random | A       | 2.0   | -           | -0.0970275 | 0.00817014 | 0.908 [0.893, 0.922] | -9.25% [-10.69%, -7.78%]   |
| structured_minus_random | B       | 1.0   | -           | -0.0789088 | 0.00841948 | 0.924 [0.909, 0.940] | -7.59% [-9.10%, -6.05%]    |
| structured_minus_random | B       | 2.0   | -           | -0.0744485 | 0.00865863 | 0.928 [0.913, 0.944] | -7.17% [-8.74%, -5.59%]    |
| day2_minus_day1         | A       | -     | random      | -0.10027   | 0.0100722  | 0.905 [0.887, 0.923] | -9.54% [-11.31%, -7.74%]   |
| day2_minus_day1         | A       | -     | structured  | -0.14716   | 0.00570973 | 0.863 [0.854, 0.873] | -13.68% [-14.64%, -12.71%] |
| day2_minus_day1         | B       | -     | random      | -0.0979915 | 0.0105058  | 0.907 [0.888, 0.926] | -9.33% [-11.18%, -7.45%]   |
| day2_minus_day1         | B       | -     | structured  | -0.0935312 | 0.00599383 | 0.911 [0.900, 0.921] | -8.93% [-9.99%, -7.85%]    |
| B_minus_A               | -       | 1.0   | random      |  0.0800797 | 0.0998626  | 1.083 [0.891, 1.318] | 8.34% [-10.92%, 31.76%]    |
| B_minus_A               | -       | 1.0   | structured  |  0.0513089 | 0.0995084  | 1.053 [0.866, 1.279] | 5.26% [-13.39%, 27.93%]    |
| B_minus_A               | -       | 2.0   | random      |  0.0823584 | 0.0998741  | 1.086 [0.893, 1.321] | 8.58% [-10.72%, 32.06%]    |
| B_minus_A               | -       | 2.0   | structured  |  0.104937  | 0.0995091  | 1.111 [0.914, 1.350] | 11.06% [-8.62%, 34.98%]    |


## Supplementary Table S1. Missingness / data-quality sensitivity

| scenario             |   n_participant_day |   n_retention | long_session_excluded   |   seq_group_coef |   Seq Group p |   ret_group_coef |   Retention Group p |
|:---------------------|--------------------:|--------------:|:------------------------|-----------------:|--------------:|-----------------:|--------------------:|
| baseline             |                  72 |            36 | nan                     |          39.8833 |         0.024 |          32.3117 |               0.419 |
| truncate_119         |                  72 |            36 | nan                     |          41.1795 |         0.02  |          33.1671 |               0.408 |
| truncate_120         |                  72 |            36 | nan                     |          41.3192 |         0.019 |          32.9588 |               0.411 |
| exclude_long_session |                  71 |            35 | 9_day1                  |          40.8413 |         0.028 |          31.9051 |               0.44  |


## Supplementary Table S2. Permutation tests

| outcome                    |   observed_B_minus_A |   Permutation p (two-sided) |   n_perm |
|:---------------------------|---------------------:|----------------------------:|---------:|
| SeqLearning_Index_all_day2 |             -9.74527 |                       0.61  |    10000 |
| Retention_Sequence         |             16.3939  |                       0.692 |    10000 |


## Supplementary Table S3. Clinical stratification

| stratum    |   n | B vs A (95% CI)           |     p |
|:-----------|----:|:--------------------------|------:|
| MoCa_low   |  13 | 1.547 [-90.082, 93.177]   | 0.974 |
| MoCa_high  |  23 | -19.026 [-66.058, 28.005] | 0.428 |
| GDS_low    |  31 | -7.697 [-56.525, 41.130]  | 0.757 |
| NIHSS_low  |  22 | -23.980 [-83.480, 35.519] | 0.43  |
| NIHSS_high |  14 | -17.140 [-95.372, 61.092] | 0.668 |

*Outcome*: Day-2 sequence learning index (SeqLearning_Index_all, ms; random − structured RT). Effect estimate: Group B − Group A (unadjusted OLS within each stratum). 95% CI and p-value from HC3-robust standard errors. Note: GDS_high stratum (n = 5) was omitted from the table due to insufficient cell size.

*Stratification thresholds* (pre-specified, clinically anchored):

| Stratum | Variable | Threshold | Clinical reference |
|:--------|:---------|:----------|:------------------|
| MoCa_low | Montreal Cognitive Assessment (MoCa_sum) | < 24 | Standard cut-off for mild cognitive impairment |
| MoCa_high | MoCa_sum | ≥ 24 | Normal cognitive screening |
| GDS_low | Geriatric Depression Scale (GDS_sum) | < 6 | Below clinically relevant depressive symptom threshold |
| GDS_high | GDS_sum | ≥ 6 | Clinically relevant depressive symptoms (n = 5; not shown) |
| NIHSS_low | NIH Stroke Scale (NIHSS) | ≤ 2 | Minor neurological deficit |
| NIHSS_high | NIHSS | > 2 | Moderate–severe neurological deficit |


## Supplementary Table S4. Time-trend robustness (linear vs flexible)

Planned contrasts (linear vs flexible block-wise trend), including material change criterion (|delta| < 2 percentage points).

| contrast                | Group   |   Day | Condition   |   log_diff_linear |   SE_linear |    p_linear |   pct_change_linear |   log_diff_flex |    SE_flex |      p_flex |   pct_change_flex |   delta_pct_points | materially_unchanged_2pp   |
|:------------------------|:--------|------:|:------------|------------------:|------------:|------------:|--------------------:|----------------:|-----------:|------------:|------------------:|-------------------:|:---------------------------|
| structured_minus_random | A       |     1 | nan         |        -0.050138  |  0.00820237 | 9.80063e-10 |          -0.0489018 |      -0.0492624 | 0.0082029  | 1.90764e-09 |        -0.0480687 |         0.0833163  | True                       |
| structured_minus_random | A       |     2 | nan         |        -0.0970275 |  0.00817014 | 0           |          -0.092469  |      -0.0968974 | 0.00815459 | 0           |        -0.0923509 |         0.0118093  | True                       |
| structured_minus_random | B       |     1 | nan         |        -0.0789088 |  0.00841948 | 0           |          -0.0758758 |      -0.0788554 | 0.00843105 | 0           |        -0.0758265 |         0.0049355  | True                       |
| structured_minus_random | B       |     2 | nan         |        -0.0744485 |  0.00865863 | 0           |          -0.0717447 |      -0.0744003 | 0.00864207 | 0           |        -0.0716999 |         0.00447459 | True                       |
| day2_minus_day1         | A       |   nan | random      |        -0.10027   |  0.0100722  | 0           |          -0.0954071 |      -0.0998728 | 0.0100627  | 0           |        -0.0950474 |         0.0359612  | True                       |
| day2_minus_day1         | A       |   nan | structured  |        -0.14716   |  0.00570973 | 0           |          -0.136844  |      -0.147508  | 0.00570088 | 0           |        -0.137144  |        -0.030036   | True                       |
| day2_minus_day1         | B       |   nan | random      |        -0.0979915 |  0.0105058  | 0           |          -0.0933434 |      -0.0981152 | 0.0105177  | 0           |        -0.0934556 |        -0.0112142  | True                       |
| day2_minus_day1         | B       |   nan | structured  |        -0.0935312 |  0.00599383 | 0           |          -0.0892904 |      -0.0936601 | 0.00599665 | 0           |        -0.0894078 |        -0.0117381  | True                       |
| B_minus_A               | nan     |     1 | random      |         0.0800797 |  0.0998626  | 0.422611    |           0.0833735 |       0.0806271 | 0.0998684  | 0.419475    |         0.0839666 |         0.0593154  | True                       |
| B_minus_A               | nan     |     1 | structured  |         0.0513089 |  0.0995084  | 0.606117    |           0.052648  |       0.051034  | 0.0995144  | 0.60807     |         0.0523587 |        -0.0289286  | True                       |
| B_minus_A               | nan     |     2 | random      |         0.0823584 |  0.0998741  | 0.409586    |           0.0858449 |       0.0823846 | 0.0998784  | 0.409457    |         0.0858734 |         0.00284489 | True                       |
| B_minus_A               | nan     |     2 | structured  |         0.104937  |  0.0995091  | 0.291631    |           0.110641  |       0.104882  | 0.0995149  | 0.291915    |         0.110579  |        -0.00618772 | True                       |


### Supplementary Table S4b. Difference-in-Differences (DoD) comparison

| model           |    log_DoD |        SE |          p |   pct_change |
|:----------------|-----------:|----------:|-----------:|-------------:|
| linear          | -0.0513499 | 0.0167268 | 0.00214121 |   -0.0500538 |
| flexible_bs_df4 | -0.0520902 | 0.0166971 | 0.00181022 |   -0.0507568 |


### Supplementary Table S5. Speed–accuracy coupling

**Note on text claim**: The manuscript statement "In both groups, higher error rates were associated with shorter reaction times" is **incorrect**. Group A shows a *positive* coupling (higher error rate → slower RT), while Group B shows a *negative* coupling (higher error rate → faster RT, i.e., classic speed–accuracy tradeoff). These findings are based on the existing analyses in `speed_accuracy_tradeoff_key_terms.csv` and `speed_accuracy_hit_model_key_terms.csv`.

#### S5a. Block-level OLS model: RT–error coupling (logRT ~ C(Group)×C(day)×C(condition) + errorRate + C(Group):errorRate + BlockNumber)

| Term | Coef (log-RT) | 95% CI | p | % RT per unit error rate |
|:-----|:-------------:|:------:|:-:|:------------------------:|
| errorRate (Group A baseline) | 0.196 | [0.076, 0.317] | 0.001 | +21.7% [+7.9%, +37.3%] |
| C(Group)[T.B]:errorRate (Group B vs A interaction) | −0.539 | [−0.687, −0.391] | <0.001 | −41.7% [−49.7%, −32.5%] |
| **Derived: Group A effective slope** | **0.196** | [0.076, 0.317] | **0.001** | **+21.7%** |
| **Derived: Group B effective slope** | **−0.344** | – | – | **−29.1%** |

Interpretation: Group A blocks with higher error rates are *slower* (consistent/disengagement pattern). Group B blocks with higher error rates are *faster* (classic speed–accuracy tradeoff). The between-group difference in coupling direction is highly significant (p < 0.001).

#### S5b. Trial-level GEE logistic model: hit probability ~ z-scored RT (isHit ~ zRT + C(Group):zRT + C(day):zRT + C(condition):zRT)

| Term | Coef (log-odds) | p | Interpretation |
|:-----|:-----------:|:--:|:--------------|
| zRT (Group A, Day 1 baseline) | 0.129 | 0.117 | Non-significant in Group A |
| C(Group)[T.B]:zRT | 0.303 | 0.006 | Group B: slower trials → more accurate (stronger tradeoff) |
| C(day)[T.2]:zRT | −0.159 | 0.040 | Coupling attenuated on Day 2 across groups |
| C(condition)[T.structured]:zRT | −0.110 | 0.103 | Non-significant condition effect |

Higher zRT = slower than participant mean. Positive Group B interaction confirms that Group B participants show a more pronounced speed–accuracy tradeoff at the trial level.

---

## Supplementary Table S6. Sex-covariate sensitivity analysis and Day-1 group difference

### S6a. Sex as covariate in the primary endpoint model

Dependent variable: `SeqLearning_Index_all` (ms). Full model: `~ C(Group) × C(day) + Age + fuglmayrshort_sum + MoCa_sum [+ Gender_num]`. HC3-robust SEs. n = 72 participant-days (36 participants × 2 days).

| Model | Term | Estimate (ms) | 95% CI | p |
|:------|:-----|:-------------:|:------:|:-:|
| Base (without sex) | Group B−A (main effect) | +39.88 | [+5.22, +74.54] | 0.024 |
| Base (without sex) | Group×Day interaction | −48.94 | [−101.1, +3.24] | 0.066 |
| +Sex (with Gender) | Group B−A (main effect) | +43.57 | [+7.35, +79.79] | 0.018 |
| +Sex (with Gender) | Group×Day interaction | −48.94 | [−101.8, +3.90] | 0.070 |
| +Sex (with Gender) | Gender coefficient | +10.79 | [−23.85, +45.43] | 0.541 |

**Interpretation:** Sex is not a significant predictor of sequence learning (p = 0.54). Adding sex as covariate does not materially change either the main Group effect or the Group×Day interaction; if anything, the estimated group difference increases slightly after adjustment. The sex imbalance between groups (Group A: ~90% male; Group B: ~59% male) does not confound the primary results.

---

### S6b. Formal test of Day-1 group difference (catch-up concern)

Dependent variable: `SeqLearning_Index_all` (ms; participant-level mean of structured RT advantage per session). Three models per day: (1) unadjusted two-sample t-test, (2) OLS adjusted for Age, FMA, MoCa, (3) additionally adjusted for sex.

**Group-level means on Day 1:** Group A = 46.7 ± 40.6 ms (n = 19); Group B = 85.9 ± 58.0 ms (n = 17).  
**Group-level means on Day 2:** Group A = 75.8 ± 50.3 ms (n = 19); Group B = 66.0 ± 60.1 ms (n = 17).

| Day | Analysis | B−A difference (ms) | 95% CI | p |
|:----|:---------|:-------------------:|:------:|:-:|
| 1 | Unadjusted t-test | +39.20 | [+6.10, +72.29] | **0.024** |
| 1 | Adjusted (Age, FMA, MoCa) | +40.02 | [+4.56, +75.49] | **0.027** |
| 1 | Adjusted + Gender | +46.50 | [+5.55, +87.45] | **0.026** |
| 2 | Unadjusted t-test | −9.75 | [−46.20, +26.71] | 0.600 |
| 2 | Adjusted (Age, FMA, MoCa) | −9.20 | [−49.42, +31.02] | 0.654 |
| 2 | Adjusted + Gender | −8.30 | [−53.11, +36.52] | 0.717 |

**Interpretation:** Group B shows a **significantly larger** structured sequence advantage already on Day 1 (B−A ≈ +39–47 ms, p ≈ 0.024–0.027, consistent across all three models). This directly addresses the catch-up concern. Under a pure catch-up scenario one would expect Group A to approach but not exceed Group B. Instead, by Day 2 the advantage **reverses**: Group A now shows a numerically *larger* sequence advantage (75.8 vs 66.0 ms; B−A = −9.75 ms, p = 0.60), consistent with a genuine differential learning trajectory. The DoD statistic quantifying the group difference in day-to-day change remains significant (−5.0%, 95% CI [−8.1%, −1.8%], p = 0.002) after adjusting for the Day-1 baseline difference. Adjusting for sex does not reduce the Day-1 effect (adjusted estimate +46.50 ms, p = 0.026), confirming that sex imbalance does not account for the starting-point difference.

---

### Supplementary Table S4c. Model fit comparison (secondary)

| model           | formula                                                                                               |      AIC |      BIC |   logLik |   df_modelwc |   n_obs |      LLR |         LLR_p |
|:----------------|:------------------------------------------------------------------------------------------------------|---------:|---------:|---------:|-------------:|--------:|---------:|--------------:|
| linear          | logRT ~ C(Group) * C(day) * C(condition) + BlockNumber + Age + fuglmayrshort_sum + MoCa_sum           | -6174.88 | -6076.02 |  3101.44 |           13 |    8615 | nan      | nan           |
| flexible_bs_df4 | logRT ~ C(Group) * C(day) * C(condition) + bs(BlockNumber, df=4) + Age + fuglmayrshort_sum + MoCa_sum | -6203.48 | -6083.44 |  3118.74 |           16 |    8615 | nan      | nan           |
| comparison      | flexible_vs_linear                                                                                    |   nan    |   nan    |   nan    |            3 |    8615 |  34.6009 |   1.47928e-07 |