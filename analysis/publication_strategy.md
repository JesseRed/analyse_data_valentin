## Publication strategy and what is “publishable” in this dataset

### Executive summary (one paragraph)
Across both groups, the dataset shows robust and theoretically coherent **implicit sequence-learning signatures** in the Serial Reaction Time Task (SRTT): RTs are faster in structured (blue/green) than random (yellow) blocks, learning curves are **non-linear**, and both repeating sequences (blue and green) show reliable facilitation relative to random. Under the **pre-specified minimal adjustment set** (Age, Fugl-Meyer, MoCa), aggregated endpoint analysis for `SeqLearning_Index_all` shows a **nominal group effect** (B>A at day 1), but unadjusted permutation tests for day‑2 endpoints remain null. The most mechanistically interpretable and robust group finding remains the block-level mixed-model result: the **day-to-day evolution of the structured–random separation differs by group** (Group×day×condition), consistent with group differences in sequence-specific skill expression dynamics rather than a single static mean shift.

---

## 1) What parts are strongest and publication-worthy (high confidence)

### A. Core phenomenon: sequence-specific facilitation exists and is robust
**Why publishable:** This is the foundational manipulation check for SRTT learning and it is strong in the data.

**Evidence already in outputs**
- Mixed-model simple effects show **structured < random RT** within each group/day with tight CIs (Table 3 in `analysis/tables.md`; sources: `analysis/outputs/03_models/mixedlm_contrasts.csv` and `analysis/outputs/05_additional_analyses/planned_contrasts_simple_effects.csv`).  
  - Example (ratio scale): Group A day 1 ~ −5.6% and day 2 ~ −9.3%; Group B day 1 ~ −8.1% and day 2 ~ −7.6%.
- Blue/green separated: both sequences are faster than yellow in both groups/days (`analysis/outputs/05_additional_analyses/blue_green_contrasts.csv`).

**How to present**
- Make this the first Results subsection: “Task manipulation check / sequence learning signature”.
- Primary figure: group-wise learning curves by day and condition (already: `analysis/outputs/04_figures/qc_block_trajectory_group_condition_ci.png`), plus a panel showing structured–random contrasts with CIs (Table 3).

### B. Learning curves are non-linear (important methodological contribution)
**Why publishable:** Many SRTT analyses assume linear change over blocks; your data show the spline model improves fit (LLR p=0.007). This supports a more modern analysis strategy and strengthens interpretability of group comparisons.

**Evidence**
- Spline vs linear comparison (`analysis/outputs/05_additional_analyses/nonlinear_model_comparison.csv`, `nonlinear_llr_test.csv`): AIC improves and LLR p=0.007.

**How to present**
- Frame as “learning unfolds nonlinearly; linear slopes are incomplete summaries”.  
- In Methods/Stats: justify splines/piecewise models as sensitivity/secondary.

### C. Data pipeline + QC is clean and transparent (publishable as rigor)
**Why publishable:** Clinical datasets often fail on reproducibility/QC. You have an end-to-end pipeline with explicit RT reconstruction, trimming, sensitivity, and exported artifacts.

**Evidence**
- Reproducible scripts: `analysis/run_srtt_analysis.py`, `analysis/run_additional_analyses.py`, `analysis/compute_mixedlm_emm.py`, `analysis/build_publication_tables.py`.
- QC summary: canonical session length is modal (952 rows / 119 blocks for 72/73 sessions), and trimming is quantified (`analysis/outputs/report.md`).
- Sensitivity table S1 shows conclusions are stable to truncation/excluding the long session (`analysis/tables.md`).

**How to present**
- Include a concise “Data processing and RT derivation” subsection (this is often valued by reviewers).
- Put QC and sensitivity into Supplement (S1).

---

## 2) What can be published about Group A vs B (and what claim strength is justified)

### A. Strongest group-related result: differential day-to-day change in structured–random separation
**What is supported:** A group difference in **how** the structured advantage changes from day 1 to day 2 (interaction structure), rather than a stable group offset at a single time point.

**Evidence**
- MixedLM: significant Group×day×condition term (p=0.005 in `analysis/outputs/03_models/mixedlm_summary.txt`).
- EMM simple effects show:  
  - Group A: structured advantage increases from day 1 to day 2 (more negative structured−random)  
  - Group B: structured advantage is similar/slightly reduced from day 1 to day 2  
  This is exactly the mechanistic pattern implied by the interaction (Table 3 in `analysis/tables.md`).

**What NOT to overclaim**
- Avoid claiming “VR improves sequence learning overall” as a blanket statement, because:
  - Even though minimally adjusted GLM now shows a nominal `SeqLearning_Index_all` group effect (p≈0.048 at day 1), the `Group×day` interaction is not significant (p≈0.128) and endpoint permutation tests remain null.
  - Permutation tests on unadjusted day‑2 endpoint are null (p≈0.61).
- Instead claim: **group-dependent expression/consolidation of sequence-specific performance across days** (a dynamic effect).

### B. Complementary group-related signal: speed–accuracy coupling differs by group
**Evidence**
- Block-level RT model: errorRate positively associated with logRT; Group×errorRate significant (`analysis/outputs/05_additional_analyses/speed_accuracy_tradeoff_key_terms.csv`).
- Trial-level hit model: Group×zRT and Day×zRT terms significant (`analysis/outputs/05_additional_analyses/speed_accuracy_hit_model_key_terms.csv`).

**Interpretation options**
- Groups may differ in **strategic control** (cautious vs fast responding), not simply mean speed.
- VR/gamification may alter **reward sensitivity** or **criterion setting** (speed emphasis vs accuracy emphasis), which can change how sequence learning manifests.

**How to publish without overreach**
- Present as secondary mechanistic analysis: “strategy/criterion differences may moderate observed RT learning effects”.
- Tie to the broader rehab context: patients may trade accuracy for speed differently depending on engagement/motivation.

---

## 3) What is currently weaker (still useful, but best as exploratory/supplement)

### A. Endpoint-style group differences on aggregated metrics
- SeqLearning_Index_all: nominally significant group main effect in the minimally adjusted GLM, but non-significant `Group×day` interaction and null day‑2 permutation endpoint.
- Retention metrics: group effects not significant; wide CIs.

**How to include**
- Keep as: “Endpoint-style effects provide supportive but limited evidence; strongest inference comes from block-level dynamic contrasts.”
- Use Table 2 (GLM) and permutation table (S2) as transparent reporting.

### B. High-dimensional covariate modeling as “predictors of responsiveness”
- This risk is now mitigated in confirmatory analyses by the minimal adjustment set (Age, Fugl-Meyer, MoCa).
- Penalized models (Lasso shrinking all coefficients) signal limited stable discovery capacity.
- Interaction screening: no robust interactions after FDR.
- Stratification: wide CIs and small strata.

**How to include**
- Publish as “we explored moderators; none survived correction; results hypothesis-generating”.

---

## 4) Recommended publication angles (choose one main “story”)

### Option 1 (most defensible): “Dynamic sequence-learning expression differs by intervention group”
**Core claim:** Group differences appear in the **day-to-day evolution** of sequence-specific performance, not as a simple average offset.

**Ideal targets:** neurorehabilitation / motor learning journals, or VR rehab outlets, where mechanistic learning metrics are valued.

**Key figures/tables**
- Fig 1: RT curves by block with condition×group×day
- Table 3: mixed-model simple effects with CIs (ratio/% scale)
- Supplement: spline evidence + QC/sensitivity + permutation endpoints

### Option 2 (methodological): “Why linear slopes are insufficient in short clinical SRTT datasets”
**Core claim:** Nonlinear modeling improves fit and interpretation; mixed models yield more sensitive inference than endpoint aggregation under missingness and heterogeneity.

**Strengths:** less dependent on intervention effect size; emphasizes analytic rigor and generalizable lessons.

### Option 3 (strategy/criterion): “VR may shift speed–accuracy coupling, changing how implicit learning is expressed”
**Core claim:** group differences arise in the *relationship* between speed and accuracy rather than raw speed.

**Caveat:** would need careful framing and ideally replication; best as secondary paper or strong secondary aim.

---

## 5) Theoretical framework to motivate the story

### A. Implicit sequence learning and consolidation
SRTT is a classic paradigm for **implicit motor sequence learning**, where repeated structured sequences (blue/green) produce faster RTs than random sequences (yellow). A key theoretical distinction is between:
- **online learning** (performance changes within a session/day),
- **offline consolidation / between-day change** (how the learned structure expresses on the next day).

Your strongest group-related signal is specifically about **between-day change in the structured–random separation**, which maps naturally onto consolidation/expression frameworks.

### B. Reward, motivation, and engagement as modulators of motor learning
In your experimental context (VR vs non‑VR), the intervention can be framed as manipulating:
- **engagement/attention**, and/or
- **reward signals** (salience, feedback, motivation),
which are theorized to influence learning rate and especially **retention/consolidation**.

Even if group differences in raw endpoints are small, a shift in the day-to-day structured advantage is theoretically consistent with reward-modulated consolidation (and with the dissertation framing around reward-oriented VR games and motivation).

### C. Strategy and criterion setting (speed–accuracy trade-offs)
In clinical cohorts, observed “learning” can partly reflect strategic shifts (criterion setting) rather than purely improved motor execution. A group difference in the coupling between error rate and RT (and in the RT–hit relation) supports a framework where the intervention affects **how participants choose to respond** (more cautious vs more speeded), which then changes the apparent size of the structured–random RT separation.

This is not a nuisance only: it is theoretically meaningful because reward/engagement can shift effort allocation and response criteria, altering performance dynamics without necessarily changing average RT endpoints.

---

## 6) Concrete publication plan (what to put in main text vs supplement)

### Main text (publishable core)
- **Primary outcome presentation**: block-level mixed model with planned contrasts (Table 3), focusing on the **day-to-day evolution of structured vs random** within each group and the resulting group-dependent dynamic.
- **Manipulation check**: structured faster than random; blue/green vs yellow.
- **Nonlinear learning**: brief statement + one key statistic (spline improves fit) to justify non-linear modeling.

### Supplement (transparency, robustness, exploration)
- QC summary + session length mismatch and trimming diagnostics (from `analysis/outputs/report.md` and QC figures).
- Sensitivity analyses (S1), permutation endpoints (S2), stratification (S3).
- Full covariate GLM table outputs and robust/penalized model coefficients.
- Speed–accuracy analyses as secondary mechanistic section if journal allows; otherwise as supplement.

---

## 7) Messaging: what can be claimed (and phrasing that reviewers accept)

### Safe, evidence-aligned claims
- “Both groups show robust sequence-specific facilitation (structured < random RT).”
- “Learning trajectories are non-linear; spline models fit better than linear slopes.”
- “Group differences are most apparent in the **day-by-condition dynamics** (mixed-model interaction and planned contrasts), consistent with group-dependent expression of sequence-specific performance across days.”
- “Aggregated endpoint analyses are compatible with this pattern but are less definitive than hierarchical dynamic contrasts.”

### Claims to avoid (given current evidence)
- Avoid “VR improves learning” as a global statement without qualifiers.
- Avoid strong claims of retention differences between groups based on the current retention endpoints and permutation results.

---

## 8) Limitations (and how to proactively address them in a paper)

### A. Session length mismatch (119 blocks modal vs 120 expected)
- Treat this as a **data logging/protocol implementation** issue rather than a fatal flaw.
- Report it explicitly (QC summary), and emphasize that **sensitivity analyses** (truncate 119 vs 120; exclude long session) do not materially change group-effect estimates (S1).

### B. Sample size vs covariate dimensionality
- Minimal confirmatory adjustment (Age, Fugl-Meyer, MoCa) reduces multicollinearity risk and improves interpretability.
- Keep high-dimensional models as exploratory/sensitivity only.

### C. Multiple comparisons / researcher degrees of freedom
- Pre-specify the primary mixed-model contrast set (structured−random within group/day + DiD summary).
- Keep exploratory moderator/strata screens in supplement with FDR control (already done).

---

## 9) “What can we publish?” (practical recommendation)

### Most publishable package right now (recommended)
A single coherent manuscript centered on:
- **Implicit sequence learning is present and non-linear** in this clinical SRTT dataset.
- **VR vs control differences emerge in the evolution of sequence-specific effects across days**, best captured with hierarchical modeling and planned contrasts.
- **Speed–accuracy coupling differs by group**, providing a mechanistic hypothesis about strategy/criterion effects of the intervention.

This is publishable as a mechanistic/analytic contribution even if endpoint group differences are modest, as long as the narrative is explicit about uncertainty and focuses on the dynamic effects where the evidence is strongest.

### If you want a more “clinical-effect” paper
You likely need either:
- larger N, or
- a pre-registered primary endpoint focused on the dynamic contrast (DiD of structured–random separation), plus a replication cohort.

