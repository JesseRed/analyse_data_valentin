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
The SRTT is a standard implicit motor sequence learning paradigm. Each block consists of **8 stimuli** (targets 1–4) mapped to corresponding response buttons on a controller. Participants were instructed to respond **as quickly and accurately as possible**. Per event, the software logged the block number, event number, a cumulative time stamp (“Time Since Block start”), hit indicator (`isHit`), target, pressed button, and the sequence condition (`sequence`).

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
Behavioral CSV files were semicolon‑separated and contained German decimal commas for time values. The column “Time Since Block start” was parsed by converting commas to decimal points and coercing to floating point seconds.

#### 3.5.2 RT reconstruction
The logged “Time Since Block start” is a cumulative within‑block timestamp. Trial‑wise reaction time (RT) was reconstructed as:
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
- **Time-trend robustness**: we repeated the primary planned contrasts under an alternative, flexible specification of the block-wise time trend (`bs(BlockNumber, df=4)`), while keeping the same fixed-effect structure and random-intercept specification.
- **Blue vs green separated**: mixed model on `Group × Day × sequence (yellow/blue/green)`.
- **Speed–accuracy coupling**: models assessing whether accuracy moderates RT and whether the speed–hit association differs by group/day/condition (cluster‑robust SEs by participant).
- **Robust and penalized regression**: Huber robust regression and ridge/lasso as stability/sensitivity checks for covariate‑rich endpoint models.
- **Predictor interaction screening**: Group×predictor interactions for selected baseline variables with FDR correction.
- **Clinical stratification**: stratified group comparisons using common thresholds (e.g., MoCa, GDS) where feasible.
- **Permutation tests**: two‑sided label permutation tests (10,000 permutations) for unadjusted group contrasts on day‑2 sequence learning and retention.

As a robustness check, we repeated the primary planned contrasts under an alternative, flexible specification of the block-wise time trend; estimates for the Group × Day × Condition contrasts were materially unchanged.
For this robustness check, “materially unchanged” was predefined as an absolute change <2 percentage points in structured-advantage contrasts within each Group×Day cell, together with directionally concordant DoD estimates of the Group×Day×Condition contrast.

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

