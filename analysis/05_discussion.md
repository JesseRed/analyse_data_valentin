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
2. **Intervention effects may emerge in repeated-measures dynamics** rather than static endpoint contrasts.
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
- and multiple robustness analyses (time-trend specification, sequence decomposition, missingness/session sensitivity, permutation tests).

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

