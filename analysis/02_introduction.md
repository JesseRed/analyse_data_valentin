## 2. Introduction

Stroke remains a major cause of long-term disability, with upper-limb impairment being one of the most persistent barriers to functional recovery and independence. A central challenge in neurorehabilitation is not only to improve immediate task performance, but to promote **durable motor learning** that transfers beyond the training context. In this regard, sequence-based motor paradigms such as the Serial Reaction Time Task (SRTT) provide a useful translational bridge between experimental motor learning theory and clinically relevant rehabilitation outcomes.

The SRTT is well suited for studying **implicit sequence learning**, because repeated structured stimulus-response sequences can be contrasted against random sequences while participants perform speeded responses. Faster responses in structured versus random blocks are interpreted as sequence-specific learning, whereas overall reductions in reaction time (RT) across blocks reflect more general practice-related improvements. Importantly, SRTT performance can be decomposed into at least two learning dimensions:  
(1) **general motor learning** (global speed-up with practice), and  
(2) **sequence-specific learning** (advantage for repeating over random structure).  
This distinction is particularly relevant in clinical populations, where improvements in average speed can arise from multiple sources (practice, strategy, fatigue adaptation) and may not directly index sequence learning.

From a theoretical perspective, post-stroke motor recovery is shaped by time-sensitive neuroplasticity and by motivational-neuromodulatory factors. Reward and engagement are proposed to enhance motor memory formation and retention by influencing dopaminergic learning mechanisms, effort allocation, and adherence. Virtual reality (VR)-based interventions are therefore attractive because they can combine high practice intensity with immediate feedback, salience, and game-like reinforcement. However, intervention effects may not necessarily appear as a simple static endpoint difference. Instead, they may emerge as differences in **how learning is expressed over time**—for example, in the day-to-day evolution of sequence-specific performance.

This has an important methodological consequence: endpoint-only analyses may under-detect meaningful effects when trajectories are heterogeneous or strategy-dependent. Speed-accuracy trade-offs can obscure interpretation if RT is considered in isolation. Hierarchical repeated-measures models at the block level preserve within-participant information and provide sensitive tests of dynamic intervention effects relative to participant-level endpoint aggregation alone.

In the present study, we investigate two intervention groups (Group A: VR+SRTT; Group B: SRTT-only) across two consecutive days in a post-stroke cohort. We embed our analysis in a framework that integrates:
- **implicit sequence learning theory** (structured vs random advantage),
- **motor memory dynamics** (online expression and between-day change),
- **reward/engagement modulation** (intervention-related differences in performance expression),
- and **strategy/criterion processes** (speed-accuracy coupling).

### Study aims
The study has three linked aims:

1. **Establish robust sequence-learning signatures** in the clinical SRTT data (structured vs random RT differences, including blue/green decomposition).
2. **Test intervention-group differences in learning dynamics**, focusing on whether the structured-random separation evolves differently from day 1 to day 2 between groups.
3. **Characterize explanatory mechanisms and robustness**, including speed-accuracy coupling, session-structure sensitivity, and robustness to alternative time-trend specification.

### A priori hypotheses
Based on the theoretical framework above, we formulated the following hypotheses:

- **H1 (task-level learning):** Across participants, structured blocks (blue/green) will yield faster RT than random blocks (yellow), consistent with sequence-specific learning.
- **H2 (dynamic group effect; primary):** Groups will differ in the **day-by-condition evolution** of RT, expressed as a Group x Day x Condition effect at block level.
- **H3 (strategy component):** Speed-accuracy coupling will contribute to observed performance dynamics and may differ between groups.

To maximize interpretability and reduce overadjustment risk in a moderate clinical sample, confirmatory models used a minimal pre-specified adjustment set (`Age`, `fuglmayrshort_sum`, `MoCa_sum`), while broader covariate and moderator analyses were treated as secondary/exploratory.

Taken together, this framework positions intervention effects as potentially **dynamic and mechanistic** rather than purely endpoint-based. Demonstrating such dynamics would have practical implications for both clinical trial analysis strategies and the design of engagement-driven rehabilitation interventions.

