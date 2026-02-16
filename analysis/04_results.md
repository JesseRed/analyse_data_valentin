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
To test whether the intervention groups differed in how sequence‑specific performance evolved from day 1 to day 2, we fitted a linear mixed model to block‑level log‑RT (`logRT`) with a participant random intercept. Fixed effects included `Group × day × condition` (structured vs random), block number, and a reduced covariate set (Age, AES_sum, MoCa_sum, NIHSS, TSS) for model stability.

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
For comparison with common endpoint analyses, we estimated heteroscedasticity‑robust OLS models (HC3) at the participant‑day level. Models included the planned covariates (AES_sum, Age, Gender, Depression, SportsActivity, Fugl‑Meyer, EQ5D, GDS, MoCa, MORE, TSS, NIHSS) and `Group × day` interactions for day‑varying outcomes.

Across the endpoint outcomes, group effects were generally small and imprecise (Table 2). For the aggregate sequence‑learning index (`SeqLearning_Index_all`), the day‑1 group contrast (B vs A) showed a **trend‑level** positive coefficient (Table 2), and the `Group×day` interaction was negative (trend‑level), consistent in direction with the mixed‑model dynamic pattern; however, these endpoint analyses did not reach conventional significance thresholds.

### 4.5 Mechanistic and robustness analyses supporting the primary interpretation

#### 4.5.1 Non‑linear learning trajectories
Learning curves were not well captured by a single linear block slope. A spline model with `bs(BlockNumber, df=4)` improved fit over the linear block model (likelihood‑ratio test **p = 0.007**; AIC improved from 4175.07 to 4169.66), indicating **non‑linear** learning trajectories. This supports the use of hierarchical and flexible time‑course models when interpreting group differences.

#### 4.5.2 Blue vs green sequences analyzed separately
When separating structured sequences by color, both **blue** and **green** blocks were faster than **yellow** (random) blocks across groups and days, confirming that sequence‑specific facilitation was present for both repeating sequences. In a mixed model on log‑RT with `Group × day × sequence`, contrasts for blue−yellow and green−yellow were consistently negative (see `analysis/outputs/05_additional_analyses/blue_green_contrasts.csv`), supporting the interpretation that the primary mixed‑model result reflects sequence‑specific dynamics rather than a single idiosyncratic sequence type.

#### 4.5.3 Speed–accuracy coupling
Because changes in RT can reflect strategic shifts (criterion setting) as well as learning, we evaluated speed–accuracy coupling:
- In block‑level models, error rate was positively associated with log‑RT (higher error rate associated with slower RT), and the **Group×errorRate** term was significant (`analysis/outputs/05_additional_analyses/speed_accuracy_tradeoff_key_terms.csv`), suggesting group differences in the relationship between accuracy and speed.
- In trial‑level hit models, the **Group×zRT** term was significant (`analysis/outputs/05_additional_analyses/speed_accuracy_hit_model_key_terms.csv`), indicating that the association between trial speed and hit probability differed by group.

Together, these findings motivate interpreting group differences as **differences in the dynamics of sequence expression and response strategy**, rather than a simple mean RT shift.

### 4.6 Sensitivity analyses and confirmatory robustness checks

#### 4.6.1 Data‑quality sensitivity (119 vs 120 blocks; long session exclusion)
Group coefficient estimates for the sequence‑learning endpoint were stable across (i) truncation to the first 119 blocks, (ii) truncation to 120 blocks, and (iii) exclusion of the single long session; estimates remained in the same direction and similar magnitude (Supplementary Table S1).

#### 4.6.2 Permutation tests (endpoint contrasts)
As confirmatory robustness checks on endpoint‑style between‑group differences, two‑sided permutation tests (10,000 permutations) did not support a reliable unadjusted group difference for day‑2 sequence learning or for retention sequence learning (Supplementary Table S2). This reinforces the conclusion that the most reproducible group signal lies in **day‑by‑condition dynamics** captured by hierarchical models, rather than in single endpoint contrasts.

---

## Tables

### Table 2. Primary group effects (adjusted GLM, HC3)
| Outcome               |   Group effect B vs A (coef) | Group effect 95% CI   |   Group effect p |   Group×Day interaction (coef) | Group×Day 95% CI    | Group×Day p   |
|:----------------------|-----------------------------:|:----------------------|-----------------:|-------------------------------:|:--------------------|:--------------|
| ErrorRate_All         |                    0.0377463 | [-0.018, 0.093]       |            0.181 |                     -0.0299427 | [-0.100, 0.040]     | 0.404         |
| IES                   |                   75.8461    | [-107.258, 258.951]   |            0.417 |                    -69.062     | [-289.778, 151.654] | 0.540         |
| RT_Delta_All          |                    5.31047   | [-76.408, 87.029]     |            0.899 |                    -12.0941    | [-102.589, 78.401]  | 0.793         |
| RT_Slope_All          |                   -0.0320715 | [-0.857, 0.793]       |            0.939 |                      0.155599  | [-0.816, 1.127]     | 0.754         |
| Retention_General     |                   55.7526    | [-54.068, 165.573]    |            0.32  |                    nan         | NA                  | NA            |
| Retention_Sequence    |                   53.4338    | [-88.460, 195.328]    |            0.46  |                    nan         | NA                  | NA            |
| SeqLearning_Index_all |                   37.4211    | [-5.252, 80.094]      |            0.086 |                    -50.5116    | [-110.127, 9.104]   | 0.097         |

### Table 3. Mixed‑model planned contrasts (ratio scale)
| Effect                  | Group   | Day   | Condition   |   log-diff |         SE | ratio_fmt            | pct_fmt                    |
|:------------------------|:--------|:------|:------------|-----------:|-----------:|:---------------------|:---------------------------|
| structured_minus_random | A       | 1.0   | -           | -0.0574295 | 0.00847098 | 0.944 [0.929, 0.960] | -5.58% [-7.14%, -4.00%]    |
| structured_minus_random | A       | 2.0   | -           | -0.0981326 | 0.00779389 | 0.907 [0.893, 0.920] | -9.35% [-10.72%, -7.95%]   |
| structured_minus_random | B       | 1.0   | -           | -0.0846543 | 0.00801814 | 0.919 [0.905, 0.933] | -8.12% [-9.55%, -6.66%]    |
| structured_minus_random | B       | 2.0   | -           | -0.0790822 | 0.00872809 | 0.924 [0.908, 0.940] | -7.60% [-9.17%, -6.01%]    |
| day2_minus_day1         | A       | -     | random      | -0.10884   | 0.0101138  | 0.897 [0.879, 0.915] | -10.31% [-12.07%, -8.52%]  |
| day2_minus_day1         | A       | -     | structured  | -0.149543  | 0.0058446  | 0.861 [0.851, 0.871] | -13.89% [-14.87%, -12.90%] |
| day2_minus_day1         | B       | -     | random      | -0.123935  | 0.0103784  | 0.883 [0.866, 0.902] | -11.66% [-13.44%, -9.84%]  |
| day2_minus_day1         | B       | -     | structured  | -0.118363  | 0.00600362 | 0.888 [0.878, 0.899] | -11.16% [-12.20%, -10.11%] |
| B_minus_A               | -       | 1.0   | random      | -0.0248495 | 0.0980763  | 0.975 [0.805, 1.182] | -2.45% [-19.51%, 18.22%]   |
| B_minus_A               | -       | 1.0   | structured  | -0.0520742 | 0.0977211  | 0.949 [0.784, 1.150] | -5.07% [-21.62%, 14.97%]   |
| B_minus_A               | -       | 2.0   | random      | -0.0399447 | 0.0980879  | 0.961 [0.793, 1.165] | -3.92% [-20.72%, 16.45%]   |
| B_minus_A               | -       | 2.0   | structured  | -0.0208943 | 0.0977291  | 0.979 [0.809, 1.186] | -2.07% [-19.14%, 18.61%]   |

---

## Notes for figure/table placement (editorial)
- The tables above are derived from `analysis/tables.md` and correspond to the exported CSVs in `analysis/outputs/06_publication_tables/`.
- Supplementary materials recommended: S1 (sensitivity), S2 (permutation), S3 (stratification) from `analysis/tables.md`.

