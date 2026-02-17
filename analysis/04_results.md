## 4. Results

### 4.1 Sample, data completeness, and preprocessing
A total of **N = 43** participants were available in the metadata table. Behavioral SRTT data were identified for **36** day‑1 files and **37** day‑2 files, yielding **73 participant‑day sessions**. **N = 34** participants contributed data on both days and were included in retention analyses.

Reaction times (RTs) were reconstructed trial‑wise from the cumulative timestamp (“Time Since Block start”) by within‑block differencing and converted to milliseconds. Primary RT analyses used correct responses (`isHit = 1`). Outliers and implausible values were handled using (i) plausibility filtering (**150–3000 ms**), and (ii) within‑participant robust trimming based on the median ± 3×MAD rule within PID×day×condition. In total, **53,033 / 66,410** correct trials were retained for primary RT analyses after MAD‑trimming.

**Data completeness.** The canonical session length in this dataset was **952 rows / 119 blocks** for **72/73** sessions; one session showed substantially more blocks. Because the modal block count differed from the PDF expectation (120 blocks), we explicitly tested robustness to truncation (119 vs 120 blocks) and to excluding the single long session (see Section 4.6 and Supplementary Table S1).

### 4.2 Descriptive performance and learning indices
Performance was summarized per participant and day (participant‑day level) using the pre‑specified outcomes: general learning (`RT_Slope_All`, `RT_Delta_All`), sequence learning (`SeqLearning_Index_all = meanRT_random − meanRT_structured`), accuracy (`ErrorRate_All`), and inverse efficiency (`IES`). Group sizes were day‑dependent (day 1: A=18, B=18; day 2: A=21, B=16).

Across groups, structured blocks were faster than random blocks and the structured advantage tended to be larger on day 2 than day 1 (Figure 1; Table 3).

**Figure 1 (recommended).** Block‑wise RT trajectories by day, condition (structured vs random), and group (A vs B). Suggested file: `analysis/outputs/04_figures/qc_block_trajectory_group_condition_ci.png`.

### 4.3 Primary hypothesis test: dynamic sequence‑learning expression differs by group (block‑level mixed model)
To test whether the intervention groups differed in how sequence‑specific performance evolved from day 1 to day 2, we fitted a linear mixed model to block‑level log‑RT (`logRT`) with a participant random intercept. Fixed effects included `Group × day × condition` (structured vs random), block number, and the pre‑specified minimal adjustment set (**Age, fuglmayrshort_sum, MoCa_sum**).

#### 4.3.1 Main fixed effects (across groups)
Across groups, performance was faster on day 2 than on day 1 and faster in structured than random blocks (Table 3):
- **Day 2 vs Day 1**: RT decreased in both conditions (random and structured), consistent with practice‑related improvements.
- **Structured vs Random**: structured blocks were reliably faster than random blocks within each group/day (all confidence intervals excluded 0 on the log scale; Table 3).

#### 4.3.2 Group‑dependent dynamics (core result)
The critical finding was a **group difference in the day‑to‑day change of the structured–random separation**, i.e., a **Group×day×condition interaction** (see `analysis/outputs/03_models/mixedlm_summary.txt`). Planned contrasts derived from the fitted model’s fixed‑effect variance–covariance matrix quantified these dynamics (Table 3; computed at mean covariate values and mean BlockNumber).

**Structured–random separation within each group/day (RT ratio; negative % indicates faster structured blocks):**
- **Group A**: day 1 **−5.58%** (95% CI −7.14% to −4.00%), day 2 **−9.35%** (−10.72% to −7.95%)
- **Group B**: day 1 **−8.12%** (−9.55% to −6.66%), day 2 **−7.60%** (−9.17% to −6.01%)

Thus, the structured advantage **increased from day 1 to day 2 in Group A**, whereas it was **similar/slightly reduced in Group B**, yielding the observed group‑dependent day‑by‑condition dynamics.

**Figure 2 (recommended).** Model‑based simple effects: structured–random separation (with 95% CI) for each group and day (derived from Table 3).

### 4.4 Secondary analyses: endpoint-style group effects (participant‑day GLM)
For comparison with common endpoint analyses, we estimated heteroscedasticity‑robust OLS models (HC3) at the participant‑day level. Confirmatory models used the same minimal adjustment set (**Age, fuglmayrshort_sum, MoCa_sum**) and `Group × day` interactions for day‑varying outcomes.

Across endpoint outcomes, most group effects remained small and imprecise (Table 2). For the aggregate sequence‑learning index (`SeqLearning_Index_all`), the day‑1 group contrast (B vs A) was **positive and nominally significant** in the minimally adjusted model (coef 37.89 ms, 95% CI 0.40 to 75.37, p=0.048), while the `Group×day` interaction remained non-significant (p=0.128). This endpoint pattern is directionally consistent with the mixed‑model dynamics but should be interpreted together with permutation and hierarchical results.

### 4.5 Mechanistic and robustness analyses supporting the primary interpretation

#### 4.5.1 Time-trend robustness of primary contrasts
As a robustness check, we repeated the primary planned contrasts under an alternative, flexible specification of the block-wise time trend (`bs(BlockNumber, df=4)`) while preserving the same fixed-effect and random-intercept structure. Flexible time trends improved overall fit metrics; however, primary Group × Day × Condition contrast estimates were **materially unchanged** (Supplementary Table S4).
The explicit DoD contrast remained nearly identical across specifications: linear model DoD log-effect = -0.0463 (about -4.52%, p=0.0051) versus flexible model DoD log-effect = -0.0468 (about -4.57%, p=0.0045).

#### 4.5.2 Blue vs green sequences analyzed separately
When separating structured sequences by color, both **blue** and **green** blocks were faster than **yellow** (random) blocks across groups and days, confirming that sequence‑specific facilitation was present for both repeating sequences. In a mixed model on log‑RT with `Group × day × sequence`, contrasts for blue−yellow and green−yellow were consistently negative (see `analysis/outputs/05_additional_analyses/blue_green_contrasts.csv`), supporting the interpretation that the primary mixed‑model result reflects sequence‑specific dynamics rather than a single idiosyncratic sequence type.

#### 4.5.3 Speed–accuracy coupling
Because changes in RT can reflect strategic shifts (criterion setting) as well as learning, we evaluated speed–accuracy coupling:
- In block‑level models, error rate was positively associated with log‑RT (higher error rate associated with slower RT), and the **Group×errorRate** term was significant (`analysis/outputs/05_additional_analyses/speed_accuracy_tradeoff_key_terms.csv`), suggesting group differences in the relationship between accuracy and speed.
- In trial‑level hit models, the **Group×zRT** term was significant (`analysis/outputs/05_additional_analyses/speed_accuracy_hit_model_key_terms.csv`), indicating that the association between trial speed and hit probability differed by group.

Together, these findings motivate interpreting group differences as **differences in the dynamics of sequence expression and response strategy**, rather than a simple mean RT shift.

### 4.6 Sensitivity analyses and confirmatory robustness checks

#### 4.6.1 Data‑quality sensitivity (119 vs 120 blocks; long session exclusion)
Group coefficient estimates for the sequence‑learning endpoint were stable across (i) truncation to the first 119 blocks, (ii) truncation to 120 blocks, and (iii) exclusion of the single long session; estimates remained in the same direction and similar magnitude, with p-values consistently around 0.04–0.05 (Supplementary Table S1).

For time-trend robustness, absolute differences between linear and flexible structured-advantage contrasts were <2 percentage points in all Group×Day cells, and DoD estimates were directionally concordant with very similar confidence intervals (Supplementary Table S4).

#### 4.6.2 Permutation tests (endpoint contrasts)
As confirmatory robustness checks on endpoint‑style between‑group differences, two‑sided permutation tests (10,000 permutations) did not support a reliable unadjusted group difference for day‑2 sequence learning or for retention sequence learning (Supplementary Table S2). This reinforces the conclusion that the most reproducible group signal lies in **day‑by‑condition dynamics** captured by hierarchical models, rather than in single endpoint contrasts.

---

## Tables

### Table 2. Primary group effects (adjusted GLM, HC3)
| Outcome               |   Group effect B vs A (coef) | Group effect 95% CI   |   Group effect p |   Group×Day interaction (coef) | Group×Day 95% CI    | Group×Day p   |
|:----------------------|-----------------------------:|:----------------------|-----------------:|-------------------------------:|:--------------------|:--------------|
| ErrorRate_All         |                    0.0112014 | [-0.031, 0.053]       |            0.6   |                     -0.0291566 | [-0.099, 0.041]     | 0.415         |
| IES                   |                  103.055     | [-122.751, 328.860]   |            0.371 |                    -44.5818    | [-340.680, 251.516] | 0.768         |
| RT_Delta_All          |                   -3.38242   | [-87.082, 80.317]     |            0.937 |                      0.0468158 | [-100.365, 100.459] | 0.999         |
| RT_Slope_All          |                   -0.0120187 | [-0.853, 0.829]       |            0.978 |                      0.134154  | [-0.900, 1.168]     | 0.799         |
| Retention_General     |                   14.3586    | [-96.736, 125.453]    |            0.8   |                    nan         | NA                  | NA            |
| Retention_Sequence    |                   37.8916    | [-54.493, 130.276]    |            0.421 |                    nan         | NA                  | NA            |
| SeqLearning_Index_all |                   37.8891    | [0.404, 75.374]       |            0.048 |                    -43.3999    | [-99.231, 12.432]   | 0.128         |

### Table 3. Mixed‑model planned contrasts (ratio scale)
| Effect                  | Group   | Day   | Condition   |   log-diff |         SE | ratio_fmt            | pct_fmt                    |
|:------------------------|:--------|:------|:------------|-----------:|-----------:|:---------------------|:---------------------------|
| structured_minus_random | A       | 1.0   | -           | -0.0574298 | 0.00847104 | 0.944 [0.929, 0.960] | -5.58% [-7.14%, -4.00%]    |
| structured_minus_random | A       | 2.0   | -           | -0.0981324 | 0.00779394 | 0.907 [0.893, 0.920] | -9.35% [-10.72%, -7.95%]   |
| structured_minus_random | B       | 1.0   | -           | -0.0846545 | 0.00801819 | 0.919 [0.905, 0.933] | -8.12% [-9.55%, -6.66%]    |
| structured_minus_random | B       | 2.0   | -           | -0.0790822 | 0.00872815 | 0.924 [0.908, 0.940] | -7.60% [-9.17%, -6.01%]    |
| day2_minus_day1         | A       | -     | random      | -0.108872  | 0.0101139  | 0.897 [0.879, 0.915] | -10.32% [-12.08%, -8.52%]  |
| day2_minus_day1         | A       | -     | structured  | -0.149575  | 0.00584475 | 0.861 [0.851, 0.871] | -13.89% [-14.87%, -12.90%] |
| day2_minus_day1         | B       | -     | random      | -0.12393   | 0.0103785  | 0.883 [0.866, 0.902] | -11.66% [-13.43%, -9.84%]  |
| day2_minus_day1         | B       | -     | structured  | -0.118358  | 0.00600371 | 0.888 [0.878, 0.899] | -11.16% [-12.20%, -10.11%] |
| B_minus_A               | -       | 1.0   | random      |  0.0425273 | 0.101089   | 1.043 [0.856, 1.272] | 4.34% [-14.41%, 27.21%]    |
| B_minus_A               | -       | 1.0   | structured  |  0.0153025 | 0.100743   | 1.015 [0.833, 1.237] | 1.54% [-16.65%, 23.71%]    |
| B_minus_A               | -       | 2.0   | random      |  0.0274694 | 0.101086   | 1.028 [0.843, 1.253] | 2.79% [-15.69%, 25.31%]    |
| B_minus_A               | -       | 2.0   | structured  |  0.0465196 | 0.100738   | 1.048 [0.860, 1.276] | 4.76% [-14.01%, 27.63%]    |

---

## Notes for figure/table placement (editorial)
- The tables above are derived from `analysis/tables.md` and correspond to the exported CSVs in `analysis/outputs/06_publication_tables/`.
- Supplementary materials recommended: S1 (sensitivity), S2 (permutation), S3 (stratification) from `analysis/tables.md`.

