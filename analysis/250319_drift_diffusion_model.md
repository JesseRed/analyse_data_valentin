# Trial-level speed-accuracy via diffusion modeling (EZ-DDM)

## Method
- We used an EZ-diffusion approximation (Wagenmakers et al.) on participant × day × condition cells, based on trial-level accuracy and correct-trial RT moments.
- Parameters: drift rate (`v_drift`), boundary separation (`a_boundary`, caution), and non-decision time (`ter_nondecision_s`).
- Inclusion for valid EZ-DDM cells: `n_trials >= 20`, `n_correct >= 8`, `0.55 <= p(correct) <= 0.99`, finite RT moments, and positive `Ter`.
- Mixed models: `parameter ~ Group × Day × Condition + Age + fuglmayrshort_sum + MoCa_sum + (1|PID)`.

## Data coverage
- Participants with valid DDM cells: **36**
- Valid cells (PID×day×condition): **120**

## Model results
| Parameter                           |   n_obs |   Group×Day×Condition coef |        SE |        p |   DoD estimate |    DoD SE |    DoD p |
|:------------------------------------|--------:|---------------------------:|----------:|---------:|---------------:|----------:|---------:|
| Drift rate (information processing) |     120 |                -0.00136722 | 0.0129937 | 0.9162   |    -0.00136722 | 0.0129937 | 0.9162   |
| Boundary separation (caution)       |     120 |                -0.0100474  | 0.0117417 | 0.39216  |    -0.0100474  | 0.0117417 | 0.39216  |
| Non-decision time                   |     120 |                 0.0575758  | 0.0506568 | 0.255711 |     0.0575758  | 0.0506568 | 0.255711 |

## Interpretation (VR-related mechanism)
- **Drift (`v`)**: 3-way p = **0.9162**; DoD p = **0.9162**.
- **Caution (`a`)**: 3-way p = **0.3922**; DoD p = **0.3922**.
- **Non-decision (`Ter`)**: 3-way p = **0.2557**; DoD p = **0.2557**.
- Practical reading: evidence for a VR-related change in information processing is strongest if `v_drift` terms are significant; evidence for strategy/caution differences is strongest if `a_boundary` terms are significant.

## Output files
- `/home/ck/Code/analyse_data_valentin/analysis/outputs/08_ddm_250319/ddm_cell_estimates.csv`
- `/home/ck/Code/analyse_data_valentin/analysis/outputs/08_ddm_250319/ddm_mixed_model_summary.csv`