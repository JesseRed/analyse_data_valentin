## Journal Submission Positioning Note (Option 1)

### Working title options
1. **Dynamic expression of sequence-specific learning differs by intervention group in post-stroke SRTT**
2. **Beyond endpoints: hierarchical modeling reveals group-dependent day-by-condition dynamics in clinical SRTT**
3. **Intervention-related differences in day-to-day sequence learning expression after stroke**

---

## 1) One-sentence positioning
This manuscript shows that while endpoint group differences are modest, **hierarchical block-level modeling reveals a robust group difference in the day-to-day evolution of sequence-specific SRTT performance** (Group×day×condition), with complementary evidence for group-specific speed–accuracy coupling.

## 2) Why this is publishable now
- **Robust task signal:** structured (blue/green) consistently faster than random (yellow) across groups/days.
- **Mechanistic primary effect:** significant Group×day×condition interaction in mixed model; planned contrasts quantify interpretable dynamics.
- **Methodological rigor:** full reproducible pipeline, explicit RT reconstruction, robust outlier handling, and pre-specified minimal adjustment set.
- **Robustness package:** non-linear curve confirmation, blue/green decomposition, sensitivity to truncation/session anomaly, permutation endpoints.

## 3) Main claim (and claim boundary)
### Core claim to make
Group differences are best captured as **dynamic sequence-expression differences over day × condition**, not as a single static endpoint offset.

### Claims to avoid
- Avoid blanket wording such as “VR improves learning overall.”
- Avoid strong retention claims between groups (retention endpoint contrasts are imprecise).

---

## 4) Abstract-ready result backbone (4 lines)
1. In mixed-effects analysis of block-level log-RT, day and condition effects were robust (day 2 faster; structured faster than random).
2. The key group signal was a significant **Group×day×condition** interaction, indicating differential evolution of the structured–random separation across days.
3. Planned contrasts showed stronger increase of structured advantage from day 1 to day 2 in Group A than Group B.
4. Endpoint-style group contrasts were weaker; sensitivity and permutation analyses supported a dynamic rather than static group-effect interpretation.

---

## 5) Suggested manuscript architecture
- **Methods (main):** minimal adjustment set (`Age`, `fuglmayrshort_sum`, `MoCa_sum`), mixed model as primary inferential framework.
- **Results (main):**
  - Figure 1: block RT trajectories by group/day/condition.
  - Table 2: endpoint GLM effects (context only).
  - Table 3: mixed-model planned contrasts (primary interpretation).
- **Supplement:**
  - S1: missingness/session-structure sensitivity.
  - S2: permutation tests.
  - S3: clinical stratification.
  - Non-linear and speed–accuracy extensions.

---

## 6) Cover-letter paragraph (ready to paste)
We submit a mechanistic reanalysis of a two-day clinical SRTT intervention dataset in post-stroke participants. Rather than relying only on aggregated endpoints, we pre-specified a minimal-adjustment hierarchical framework and found that sequence-specific performance dynamics (structured vs random RT) evolved differently across days between intervention groups (Group×day×condition). This dynamic signal remained interpretable under extensive robustness checks, including non-linear trajectory modeling, sequence decomposition (blue/green vs random), and data-quality sensitivity analyses. We believe this provides clinically relevant insight into how intervention effects may manifest as changes in learning expression dynamics rather than static endpoint shifts.

---

## 7) Journal fit (pragmatic)
Best fit is a **neurorehabilitation / motor-learning / digital rehabilitation** journal that values:
- repeated-measures hierarchical modeling,
- clinically constrained datasets with transparent robustness,
- mechanistic interpretation beyond simple endpoint null/non-null reporting.

## 8) Final submission rule
Keep the manuscript’s “center of gravity” on the **dynamic mixed-model finding**. Treat endpoint GLMs and broad covariate moderation as secondary context, not the headline.

