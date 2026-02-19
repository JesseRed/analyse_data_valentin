## Abstract

### Background
Serial Reaction Time Task (SRTT) performance is often summarized using endpoint metrics, potentially overlooking clinically meaningful learning dynamics. In post-stroke rehabilitation, intervention effects may manifest as changes in the evolution of sequence-specific performance over time rather than as static between-group differences. We therefore examined whether sequence-learning expression across two training days differed between intervention groups.

### Methods
We analyzed a two-day, two-group post-stroke SRTT dataset (Group A: VR+SRTT; Group B: SRTT-only). Reaction times (RT) were reconstructed trial-wise and restricted to correct responses (150–3000 ms), with within-participant robust trimming (median ± 3 × MAD). The primary analysis used a block-level linear mixed-effects model on log-RT including fixed effects of Group × Day × Condition (structured vs random), block number, and minimal covariate adjustment (age, fuglmayrshort_sum, MoCa_sum), with participant random intercept. Planned contrasts quantified simple effects. Secondary endpoint-style analyses used HC3-robust OLS models. Robustness checks included non-linear trajectory modeling and sensitivity analyses.

### Results
Forty-three participants were available; 73 participant-day sessions were analyzed (34 with both days). Sequence-specific facilitation (structured < random) was robust in both groups. The primary model revealed a significant Group × Day × Condition interaction (p = 0.005), indicating differential day-to-day evolution of sequence-specific RT separation. In Group A, the structured advantage increased from day 1 to day 2 (approximately −5.6% to −9.3%), whereas in Group B it remained stable or slightly decreased (approximately −8.1% to −7.6%). Endpoint-style analyses were less sensitive overall. Non-linear models outperformed linear slopes (p = 0.007).

### Conclusions
In this post-stroke SRTT dataset, the most robust intervention-related signal was **dynamic**: groups differed in the **day-by-condition evolution** of sequence-specific performance, not primarily in static endpoint differences. These findings support hierarchical, trajectory-aware modeling for rehabilitation motor-learning datasets and suggest that intervention effects may be better captured as changes in sequence-learning expression over time.

## 2. Introduction

Stroke remains a major cause of long-term disability, with upper-limb impairment being one of the most persistent barriers to functional recovery and independence. A central challenge in neurorehabilitation is not only to improve immediate task performance, but to promote **durable motor learning** that transfers beyond the training context. In this regard, sequence-based motor paradigms such as the Serial Reaction Time Task (SRTT) provide a useful translational bridge between experimental motor learning theory and clinically relevant rehabilitation outcomes.

The SRTT is well suited for studying **implicit sequence learning**, because repeated structured stimulus-response sequences can be contrasted against random sequences while participants perform speeded responses. Faster responses in structured versus random blocks are interpreted as sequence-specific learning, whereas overall reductions in reaction time (RT) across blocks reflect more general practice-related improvements. Importantly, SRTT performance can be decomposed into at least two learning dimensions:
(1) **general motor learning** (global speed-up with practice), and
(2) **sequence-specific learning** (advantage for repeating over random structure).
This distinction is particularly relevant in clinical populations, where improvements in average speed can arise from multiple sources (practice, strategy, fatigue adaptation) and may not directly index sequence learning.

From a theoretical perspective, post-stroke motor recovery is shaped by time-sensitive neuroplasticity and by motivational-neuromodulatory factors. Reward and engagement are proposed to enhance motor memory formation and retention by influencing dopaminergic learning mechanisms, effort allocation, and adherence. Virtual reality (VR)-based interventions are therefore attractive because they can combine high practice intensity with immediate feedback, salience, and game-like reinforcement. However, intervention effects may not necessarily appear as a simple static endpoint difference. Instead, they may emerge as differences in **how learning is expressed over time**—for example, in the day-to-day evolution of sequence-specific performance.

This has an important methodological consequence: endpoint-only analyses may under-detect meaningful effects when trajectories are heterogeneous, non-linear, or strategy-dependent. In SRTT datasets, learning curves are often not strictly linear, and speed-accuracy trade-offs can obscure interpretation if RT is considered in isolation. Hierarchical repeated-measures models at the block level can preserve within-participant information and provide a more sensitive test of dynamic intervention effects than participant-level endpoint aggregation alone.

In the present study, we investigate two intervention groups (Group A: VR+SRTT; Group B: SRTT-only) across two consecutive days in a post-stroke cohort. We embed our analysis in a framework that integrates:
- **implicit sequence learning theory** (structured vs random advantage),
- **motor memory dynamics** (online expression and between-day change),
- **reward/engagement modulation** (intervention-related differences in performance expression),
- and **strategy/criterion processes** (speed-accuracy coupling).

### Study aims
The study has three linked aims:

1. **Establish robust sequence-learning signatures** in the clinical SRTT data (structured vs random RT differences, including blue/green decomposition).
2. **Test intervention-group differences in learning dynamics**, focusing on whether the structured-random separation evolves differently from day 1 to day 2 between groups.
3. **Characterize explanatory mechanisms and robustness**, including non-linear trajectory structure, speed-accuracy coupling, and sensitivity to data-completeness constraints.

### A priori hypotheses
Based on the theoretical framework above, we formulated the following hypotheses:

- **H1 (task-level learning):** Across participants, structured blocks (blue/green) will yield faster RT than random blocks (yellow), consistent with sequence-specific learning.
- **H2 (dynamic group effect; primary):** Groups will differ in the **day-by-condition evolution** of RT, expressed as a Group x Day x Condition effect at block level.
- **H3 (trajectory form):** Learning trajectories across blocks are expected to show non-linear structure; flexible models should fit better than a purely linear slope.
- **H4 (strategy component):** Speed-accuracy coupling will contribute to observed performance dynamics and may differ between groups.

To maximize interpretability and reduce overadjustment risk in a moderate clinical sample, confirmatory models used a minimal pre-specified adjustment set (`Age`, `fuglmayrshort_sum`, `MoCa_sum`), while broader covariate and moderator analyses were treated as secondary/exploratory.

Taken together, this framework positions intervention effects as potentially **dynamic and mechanistic** rather than purely endpoint-based. Demonstrating such dynamics would have practical implications for both clinical trial analysis strategies and the design of engagement-driven rehabilitation interventions.

## 3. Methods

### 3.1 Study design and procedure
We analyzed data from a prospective, two‑group intervention study with **two consecutive training days**. Participants were randomized to either **Group A (VR intervention + SRTT)** or **Group B (control; SRTT only)**. Each participant completed a daily session on the stroke unit; breaks were permitted in case of fatigue and sessions were discontinued in case of adverse symptoms or non‑compliance, consistent with the study protocol described in `GR_210126.pdf`.

### 3.2 Participants and baseline measures
The participant roster and baseline variables were provided in the metadata table `Datensatz_Round_2.csv` (semicolon‑separated). Group assignment was taken from the column `Group` (A/B).

For confirmatory (primary) models, we used a **minimal pre-specified adjustment set**:
- `Age`
- `fuglmayrshort_sum` (motor impairment / function)
- `MoCa_sum` (cognitive status)

Additional baseline variables (e.g., AES, GDS, MORE, NIHSS, TSS, EQ5D, sports activity, depression history, gender) were retained for **secondary/exploratory** analyses only.

Categorical baseline variables used in exploratory models were recoded as numeric indicators where needed (e.g., gender: female=0/male=1; depression history: no=0/yes=1; sports activity: ordinal mapping no/1h_week/1–3h_week/>3h_week → 0–3).

### 3.3 Serial Reaction Time Task (SRTT)
The SRTT is a standard implicit motor sequence learning paradigm. Each block consists of **8 stimuli** (targets 1–4) mapped to corresponding response buttons on a controller. Participants were instructed to respond **as quickly and accurately as possible**. Per event, the software logged the block number, event number, a cumulative time stamp ("Time Since Block start"), hit indicator (`isHit`), target, pressed button, and the sequence condition (`sequence`).

The experimental conditions were:
- **Structured / repeating sequence**: `sequence ∈ {blue, green}`
- **Random sequence**: `sequence = yellow`

The protocol description specifies 120 blocks; in the present dataset the canonical session length was modal at **119 blocks** (see QC and sensitivity analyses below). This was handled explicitly via sensitivity analyses rather than ad hoc exclusions.

### 3.4 Data sources and file mapping
Behavioral data were stored per participant in `data/<PID2>/` where `PID2` is the 2‑digit participant identifier (e.g., PID=1 → folder `01`). For each participant, two target files were used:
- `*_1_fertig.csv`: day 1
- `*_2_fertig.csv`: day 2

Metadata were linked to behavioral data using `PID` and group membership via `Group`.

### 3.5 Data preprocessing and quality control

#### 3.5.1 Parsing and time format
Behavioral CSV files were semicolon‑separated and contained German decimal commas for time values. The column "Time Since Block start" was parsed by converting commas to decimal points and coercing to floating point seconds.

#### 3.5.2 RT reconstruction
The logged "Time Since Block start" is a cumulative within‑block timestamp. Trial‑wise reaction time (RT) was reconstructed as:
- For the first event within a block: \(RT = TSB\)
- For subsequent events: \(RT = TSB_{i} - TSB_{i-1}\)

RT was stored in seconds and milliseconds (`RT_ms = RT_s × 1000`).

#### 3.5.3 Primary filtering and outlier handling
Primary analyses focused on correct responses (`isHit = 1`) to avoid conflating RT with incorrect keypresses.

We applied a three‑tier approach to identify implausible trials and reduce undue influence of extreme values:
1. **Parse/reset integrity flags**: trials with non‑positive within‑block timestamp differences (suggesting reset/ordering artifacts) were excluded.
2. **Hard plausibility limits**: trials with \(RT_{ms} < 150\) ms or \(RT_{ms} > 3000\) ms were marked implausible and excluded from primary RT summaries.
3. **Robust within‑subject trimming**: within each participant × day × condition (structured vs random), we excluded correct‑trial RTs outside **median ± 3×MAD**, where MAD was scaled to approximate \(\sigma\) (MAD×1.4826).

Two sensitivity RT datasets were also exported:
- **Hard‑filter only** (no MAD trimming)
- **Winsorized RT** (1% tails within participant × day × condition)

#### 3.5.4 Session‑level QC and completeness
For each participant × day file, QC flags were computed for schema compliance, monotonicity of within‑block timestamps, duplicate block/event indices, and valid category/value ranges. Because modal session structure differed from the PDF expectation, we report both the **modal canonical length** and the protocol expectation, and we ran explicit sensitivity analyses under standardized truncation and single‑session exclusion (Section 3.8).

### 3.6 Outcome definitions

#### 3.6.1 Block‑level outcomes (derived from trial‑level RT)
Trials were aggregated to block summaries per participant × day × block × condition:
- `meanRT_hit_ms`: mean RT of correct trials (after primary filtering/trimming)
- `medianRT_hit_ms`, `sdRT_hit_ms`, `nHits`
- `errorRate`: mean(1−isHit) per block (all trials)

#### 3.6.2 Participant‑day outcomes (endpoint summaries)
From block‑level data, we derived participant‑day outcomes:
- **Sequence‑learning index**:
  \(\text{SeqLearning\_Index\_all} = \overline{RT}_{random} - \overline{RT}_{structured}\)
  where structured = blue+green and random = yellow (positive values indicate faster structured performance).
- **General learning slope**: `RT_Slope_All`, computed as the linear slope of mean RT across block number (negative values indicate faster performance over blocks).
- **Early–late improvement**: `RT_Delta_All = meanRT(blocks 1–20) − meanRT(blocks 101–120)` computed on available blocks.
- **Accuracy**: `ErrorRate_All` per day and per condition.
- **Inverse efficiency score (IES)**: `medianRT_hit_ms / (1 − ErrorRate_All)` (reported descriptively).

#### 3.6.3 Retention outcomes (between‑day change)
Retention metrics were defined for participants with both days:
- **Retention_General**: \(RT_{early,day2} - RT_{late,day1}\) (negative indicates faster re‑entry performance).
- **Retention_Sequence**: \(\text{SeqLearning}_{early,day2} - \text{SeqLearning}_{late,day1}\).

### 3.7 Statistical analysis

#### 3.7.1 Primary model: hierarchical test of dynamic sequence expression
The primary inferential model tested whether the **structured–random RT separation** evolved differently across days between groups. We fitted a linear mixed model on block‑level RT (log scale):

\[
\log(RT) \sim \text{Group} \times \text{Day} \times \text{Condition} + \text{BlockNumber} + \text{Covariates} + (1 \mid \text{Participant})
\]

with:
- Group: A vs B
- Day: 1 vs 2
- Condition: structured vs random
- Covariates (minimal set): Age, fuglmayrshort_sum, MoCa_sum
- Random effects: participant‑level random intercept

The log‑RT scale was used to stabilize variance and allow ratio‑scale interpretation. Planned contrasts (estimated marginal means and simple effects) were computed from the fixed‑effect covariance matrix (delta method) at mean covariate values and mean block number, including structured−random within each group/day and day‑to‑day changes within each group/condition.

#### 3.7.2 Secondary endpoint models (participant‑day GLM)
To enable comparison with common endpoint analyses, we fitted heteroscedasticity‑robust OLS models (HC3) to participant‑day outcomes:
- Day‑varying outcomes: `Outcome ~ Group × Day + covariates`
- Retention outcomes: `Outcome ~ Group + covariates`

Covariates in these confirmatory endpoint models used the same minimal set as the primary mixed model (Age, fuglmayrshort_sum, MoCa_sum). Extended covariate sets were evaluated only in exploratory/sensitivity analyses.

#### 3.7.3 Additional analyses (mechanistic and robustness)
We conducted the following pre‑defined extensions:
- **Non‑linear learning curves**: spline basis on BlockNumber (`bs(df=4)`) compared with a linear block model.
- **Blue vs green separated**: mixed model on `Group × Day × sequence (yellow/blue/green)`.
- **Speed–accuracy coupling**: models assessing whether accuracy moderates RT and whether the speed–hit association differs by group/day/condition (cluster‑robust SEs by participant).
- **Robust and penalized regression**: Huber robust regression and ridge/lasso as stability/sensitivity checks for covariate‑rich endpoint models.
- **Predictor interaction screening**: Group×predictor interactions for selected baseline variables with FDR correction.
- **Clinical stratification**: stratified group comparisons using common thresholds (e.g., MoCa, GDS) where feasible.
- **Permutation tests**: two‑sided label permutation tests (10,000 permutations) for unadjusted group contrasts on day‑2 sequence learning and retention.

### 3.8 Sensitivity analyses addressing data completeness
Because the modal block count differed from the protocol expectation, we performed standardized sensitivity analyses:
- truncation to the first **119** blocks,
- truncation to the first **120** blocks,
- exclusion of the single session with substantially more blocks than the canonical structure.

Sensitivity results are reported in Supplementary Table S1.

### 3.9 Software and reproducibility
All analyses were implemented in Python (pandas, numpy, statsmodels, scikit‑learn) with reproducible scripts and exported intermediate datasets (trial‑level, block‑level, participant‑level) and model outputs (coefficients, summaries, tables). Key scripts:
- `analysis/run_srtt_analysis.py` (end‑to‑end preprocessing + primary models + figures)
- `analysis/compute_mixedlm_emm.py` (mixed‑model planned contrasts / EMMs)
- `analysis/run_additional_analyses.py` (extensions and robustness checks)
- `analysis/build_publication_tables.py` (publication tables)

### 3.10 Ethics
The study protocol received ethics approval as documented in `GR_210126.pdf`, and all participants provided informed consent prior to participation.

## 4. Results

### 4.1 Sample, data completeness, and preprocessing
A total of **N = 43** participants were available in the metadata table. Behavioral SRTT data were identified for **36** day‑1 files and **37** day‑2 files, yielding **73 participant‑day sessions**. **N = 34** participants contributed data on both days and were included in retention analyses.

Reaction times (RTs) were reconstructed trial‑wise from the cumulative timestamp ("Time Since Block start") by within‑block differencing and converted to milliseconds. Primary RT analyses used correct responses (`isHit = 1`). Outliers and implausible values were handled using (i) plausibility filtering (**150–3000 ms**), and (ii) within‑participant robust trimming based on the median ± 3×MAD rule within PID×day×condition. In total, **53,033 / 66,410** correct trials were retained for primary RT analyses after MAD‑trimming.

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
The critical finding was a **group difference in the day‑to‑day change of the structured–random separation**, i.e., a **Group×day×condition interaction** (see `analysis/outputs/03_models/mixedlm_summary.txt`). Planned contrasts derived from the fitted model's fixed‑effect variance–covariance matrix quantified these dynamics (Table 3; computed at mean covariate values and mean BlockNumber).

**Structured–random separation within each group/day (RT ratio; negative % indicates faster structured blocks):**
- **Group A**: day 1 **−5.58%** (95% CI −7.14% to −4.00%), day 2 **−9.35%** (−10.72% to −7.95%)
- **Group B**: day 1 **−8.12%** (−9.55% to −6.66%), day 2 **−7.60%** (−9.17% to −6.01%)

Thus, the structured advantage **increased from day 1 to day 2 in Group A**, whereas it was **similar/slightly reduced in Group B**, yielding the observed group‑dependent day‑by‑condition dynamics.

**Figure 2 (recommended).** Model‑based simple effects: structured–random separation (with 95% CI) for each group and day (derived from Table 3).

### 4.4 Secondary analyses: endpoint-style group effects (participant‑day GLM)
For comparison with common endpoint analyses, we estimated heteroscedasticity‑robust OLS models (HC3) at the participant‑day level. Confirmatory models used the same minimal adjustment set (**Age, fuglmayrshort_sum, MoCa_sum**) and `Group × day` interactions for day‑varying outcomes.

Across endpoint outcomes, most group effects remained small and imprecise (Table 2). For the aggregate sequence‑learning index (`SeqLearning_Index_all`), the day‑1 group contrast (B vs A) was **positive and nominally significant** in the minimally adjusted model (coef 37.89 ms, 95% CI 0.40 to 75.37, p=0.048), while the `Group×day` interaction remained non-significant (p=0.128). This endpoint pattern is directionally consistent with the mixed‑model dynamics but should be interpreted together with permutation and hierarchical results.

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
Group coefficient estimates for the sequence‑learning endpoint were stable across (i) truncation to the first 119 blocks, (ii) truncation to 120 blocks, and (iii) exclusion of the single long session; estimates remained in the same direction and similar magnitude, with p-values consistently around 0.04–0.05 (Supplementary Table S1).

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

## 5. Discussion

### 5.1 Principal findings
This study examined whether intervention groups differed in the expression of SRTT learning across two consecutive days in a post-stroke cohort. Three main findings emerge.

First, the dataset showed clear and robust **sequence-specific learning signatures**: structured blocks (blue/green) were consistently faster than random blocks (yellow), and this pattern was evident across both groups. This confirms that the task captured the intended implicit sequence-learning process in this clinical sample.

Second, the most informative group-related signal was **dynamic rather than static**. In block-level mixed modeling, the critical `Group x Day x Condition` interaction indicated that the day-to-day evolution of the structured-random separation differed between groups. Planned contrasts showed that the structured advantage increased from day 1 to day 2 in Group A, whereas it remained relatively stable (or slightly reduced) in Group B.

Third, endpoint-style participant-level contrasts were less decisive. Under the minimal adjustment set (`Age`, `fuglmayrshort_sum`, `MoCa_sum`), the group effect for `SeqLearning_Index_all` was nominally significant at day 1, but the corresponding `Group x Day` interaction was not significant, and endpoint permutation tests for day‑2 contrasts remained null. Together, this pattern suggests that intervention effects are better represented as **changes in learning expression over time** than as single endpoint offsets.

### 5.2 Interpretation within the theoretical framework
Our interpretation aligns with a framework that combines implicit sequence learning, motor-memory dynamics, and intervention-related modulation of engagement/reward processes.

In classical SRTT terms, a structured-random RT advantage reflects sequence-specific knowledge expression. The present results extend this by showing that group differences are most pronounced in the **trajectory** of this advantage across days. This pattern is compatible with the notion that interventions may influence not only immediate performance but also how learned structure is expressed at re-exposure (i.e., between-day expression/consolidation-like dynamics).

The additional speed-accuracy analyses support this interpretation. Significant error-rate moderation of RT and a group-dependent RT-hit relationship indicate that response strategy (criterion setting) likely contributed to observed performance patterns. In other words, the intervention signal may include both sequence-learning expression and strategic control components, which is plausible in motivated, game-like training contexts.

### 5.3 Relation to existing literature
The findings are consistent with prior motor-learning and rehabilitation literature in three ways:

1. **Implicit sequence learning remains measurable in neurologic populations**, even under constrained clinical protocols.
2. **Learning trajectories are frequently non-linear**, especially in short repeated sessions; linear block slopes alone can underrepresent meaningful temporal structure.
3. **Engagement/reward-oriented interventions may alter how learning is expressed**, including the balance between speed and accuracy, without necessarily producing large static endpoint differences in moderate samples.

Thus, our results support recent methodological recommendations to prioritize hierarchical repeated-measures models and planned contrasts over endpoint-only summaries when testing intervention effects in motor-learning paradigms.

### 5.4 Clinical and methodological implications
Clinically, the results suggest that intervention efficacy may be underestimated if evaluated solely with endpoint group contrasts. For bedside or early-phase neurorehabilitation studies, where heterogeneity and limited sample size are common, dynamic model-based measures can provide a more sensitive and mechanistically interpretable signal.

Methodologically, this study reinforces several practical points:
- model sequence-learning as **condition-specific trajectories** rather than global RT slopes only,
- preserve repeated-measures structure via mixed models,
- separate confirmatory and exploratory covariate strategies,
- and explicitly quantify data-quality sensitivity (e.g., variable session lengths).

### 5.5 Strengths
Key strengths include:
- an end-to-end reproducible analysis workflow from raw event logs to publication tables,
- transparent RT reconstruction and robust outlier handling,
- a confirmatory minimal adjustment strategy to reduce overadjustment/multicollinearity risk,
- and multiple robustness analyses (non-linear models, sequence decomposition, missingness/session sensitivity, permutation tests).

### 5.6 Limitations
Several limitations should be acknowledged.

First, session structure differed from protocol expectation (modal 119 blocks rather than 120), likely reflecting implementation or logging differences. Although sensitivity analyses showed stable directional conclusions, this remains a design-to-data discrepancy.

Second, sample size was moderate for subgroup and moderator discovery. As expected, exploratory interaction screens and stratified analyses yielded wide confidence intervals and no robust corrected signals.

Third, despite randomization, observational covariate adjustment cannot fully resolve all potential confounding and measurement error in clinical scales. Relatedly, endpoint permutation tests were null for key day‑2 contrasts, supporting a cautious interpretation of endpoint-level group effects.

Fourth, retention was assessed over a short interval (day 1 to day 2). Longer follow-up would be required to infer durable consolidation effects in a stricter sense.

### 5.7 Future directions
Future work should prioritize:
- pre-registered dynamic primary endpoints (e.g., difference-in-differences of structured-random separation),
- larger samples for moderator and subgroup inference,
- longer-term retention windows,
- and integrated modeling of speed-accuracy trade-offs as co-primary mechanistic outcomes.

From an intervention perspective, adaptive reward/feedback schemes in VR should be evaluated as potential modulators of sequence-expression dynamics, ideally with replication cohorts and harmonized task logging standards.

### 5.8 Conclusion
In this post-stroke SRTT dataset, the most robust intervention-related signal was a **dynamic group difference in sequence-learning expression across day and condition**, rather than a consistent static endpoint effect. This supports a trajectory-aware, hierarchical analytic framework for rehabilitation motor-learning studies and suggests that clinically relevant intervention effects may be better captured by how performance evolves over time than by endpoint contrasts alone.
