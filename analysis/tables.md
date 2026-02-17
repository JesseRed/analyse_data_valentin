## Table 2. Primary group effects (adjusted GLM, HC3)

| Outcome               |   Group effect B vs A (coef) | Group effect 95% CI   |   Group effect p |   Group×Day interaction (coef) | Group×Day 95% CI    | Group×Day p   |
|:----------------------|-----------------------------:|:----------------------|-----------------:|-------------------------------:|:--------------------|:--------------|
| ErrorRate_All         |                    0.0196151 | [-0.028, 0.067]       |            0.42  |                     -0.0266408 | [-0.097, 0.044]     | 0.459         |
| IES                   |                   60.973     | [-171.634, 293.580]   |            0.607 |                    -43.1465    | [-357.552, 271.259] | 0.788         |
| RT_Delta_All          |                   33.4643    | [-69.654, 136.583]    |            0.525 |                    -28.7908    | [-139.758, 82.176]  | 0.611         |
| RT_Slope_All          |                   -0.247279  | [-1.135, 0.640]       |            0.585 |                      0.341997  | [-0.749, 1.433]     | 0.539         |
| Retention_General     |                   38.3055    | [-75.425, 152.036]    |            0.509 |                    nan         | NA                  | NA            |
| Retention_Sequence    |                   33.6114    | [-49.246, 116.469]    |            0.427 |                    nan         | NA                  | NA            |
| SeqLearning_Index_all |                   41.804     | [6.312, 77.296]       |            0.021 |                    -44.6672    | [-98.403, 9.069]    | 0.103         |


## Table 3. Mixed-model simple effects (ratio scale)

| Effect                  | Group   | Day   | Condition   |   log-diff |         SE | ratio_fmt            | pct_fmt                    |
|:------------------------|:--------|:------|:------------|-----------:|-----------:|:---------------------|:---------------------------|
| structured_minus_random | A       | 1.0   | -           | -0.0482669 | 0.00830435 | 0.953 [0.937, 0.969] | -4.71% [-6.25%, -3.15%]    |
| structured_minus_random | A       | 2.0   | -           | -0.0945421 | 0.0082705  | 0.910 [0.895, 0.925] | -9.02% [-10.48%, -7.53%]   |
| structured_minus_random | B       | 1.0   | -           | -0.0787095 | 0.00829501 | 0.924 [0.909, 0.939] | -7.57% [-9.06%, -6.05%]    |
| structured_minus_random | B       | 2.0   | -           | -0.0789269 | 0.00878733 | 0.924 [0.908, 0.940] | -7.59% [-9.17%, -5.98%]    |
| day2_minus_day1         | A       | -     | random      | -0.0974089 | 0.0101972  | 0.907 [0.889, 0.925] | -9.28% [-11.08%, -7.45%]   |
| day2_minus_day1         | A       | -     | structured  | -0.143684  | 0.00577936 | 0.866 [0.856, 0.876] | -13.38% [-14.36%, -12.40%] |
| day2_minus_day1         | B       | -     | random      | -0.119531  | 0.0105476  | 0.887 [0.869, 0.906] | -11.27% [-13.08%, -9.41%]  |
| day2_minus_day1         | B       | -     | structured  | -0.119749  | 0.00606268 | 0.887 [0.877, 0.898] | -11.29% [-12.33%, -10.23%] |
| B_minus_A               | -       | 1.0   | random      |  0.0806322 | 0.102073   | 1.084 [0.887, 1.324] | 8.40% [-11.26%, 32.40%]    |
| B_minus_A               | -       | 1.0   | structured  |  0.0501896 | 0.101727   | 1.051 [0.861, 1.283] | 5.15% [-13.86%, 28.35%]    |
| B_minus_A               | -       | 2.0   | random      |  0.0585098 | 0.102105   | 1.060 [0.868, 1.295] | 6.03% [-13.20%, 29.52%]    |
| B_minus_A               | -       | 2.0   | structured  |  0.074125  | 0.101738   | 1.077 [0.882, 1.315] | 7.69% [-11.78%, 31.46%]    |


## Supplementary Table S1. Missingness / data-quality sensitivity

| scenario             |   n_participant_day |   n_retention | long_session_excluded   |   seq_group_coef |   Seq Group p |   ret_group_coef |   Retention Group p |
|:---------------------|--------------------:|--------------:|:------------------------|-----------------:|--------------:|-----------------:|--------------------:|
| baseline             |                  71 |            35 | nan                     |          41.804  |         0.021 |          33.6114 |               0.427 |
| truncate_119         |                  71 |            35 | nan                     |          43.1564 |         0.017 |          34.521  |               0.415 |
| truncate_120         |                  71 |            35 | nan                     |          43.3022 |         0.017 |          34.2995 |               0.418 |
| exclude_long_session |                  70 |            34 | 9_day1                  |          42.7713 |         0.024 |          33.3508 |               0.446 |


## Supplementary Table S2. Permutation tests

| outcome                    |   observed_B_minus_A |   Permutation p (two-sided) |   n_perm |
|:---------------------------|---------------------:|----------------------------:|---------:|
| SeqLearning_Index_all_day2 |             -6.83387 |                       0.722 |    10000 |
| Retention_Sequence         |              7.98637 |                       0.847 |    10000 |


## Supplementary Table S3. Clinical stratification

| stratum    |   n | B vs A (95% CI)           |     p |
|:-----------|----:|:--------------------------|------:|
| MoCa_low   |  12 | 19.922 [-84.002, 123.847] | 0.707 |
| MoCa_high  |  22 | -15.271 [-61.076, 30.534] | 0.513 |
| GDS_low    |  30 | -2.517 [-54.776, 49.742]  | 0.925 |
| NIHSS_low  |  22 | -23.980 [-83.480, 35.519] | 0.43  |
| NIHSS_high |  13 | -6.781 [-93.404, 79.842]  | 0.878 |


## Supplementary Table S4. Time-trend robustness (linear vs flexible)

Planned contrasts (linear vs flexible block-wise trend), including material change criterion (|delta| < 2 percentage points).

| contrast                | Group   |   Day | Condition   |   log_diff_linear |   SE_linear |   p_linear |   pct_change_linear |   log_diff_flex |    SE_flex |      p_flex |   pct_change_flex |   delta_pct_points | materially_unchanged_2pp   |
|:------------------------|:--------|------:|:------------|------------------:|------------:|-----------:|--------------------:|----------------:|-----------:|------------:|------------------:|-------------------:|:---------------------------|
| structured_minus_random | A       |     1 | nan         |        -0.0482669 |  0.00830435 | 6.1643e-09 |          -0.0471205 |      -0.0474398 | 0.00830276 | 1.10518e-08 |        -0.0463321 |         0.0788411  | True                       |
| structured_minus_random | A       |     2 | nan         |        -0.0945421 |  0.0082705  | 0          |          -0.0902106 |      -0.0943871 | 0.00825291 | 0           |        -0.0900696 |         0.0141052  | True                       |
| structured_minus_random | B       |     1 | nan         |        -0.0787095 |  0.00829501 | 0          |          -0.0756916 |      -0.0786972 | 0.00830541 | 0           |        -0.0756802 |         0.00113334 | True                       |
| structured_minus_random | B       |     2 | nan         |        -0.0789269 |  0.00878733 | 0          |          -0.0758925 |      -0.0788398 | 0.00876862 | 0           |        -0.075812  |         0.00805341 | True                       |
| day2_minus_day1         | A       |   nan | random      |        -0.0974089 |  0.0101972  | 0          |          -0.092815  |      -0.0970783 | 0.0101851  | 0           |        -0.092515  |         0.0299984  | True                       |
| day2_minus_day1         | A       |   nan | structured  |        -0.143684  |  0.00577936 | 0          |          -0.133839  |      -0.144026  | 0.00576912 | 0           |        -0.134134  |        -0.0295666  | True                       |
| day2_minus_day1         | B       |   nan | random      |        -0.119531  |  0.0105476  | 0          |          -0.112664  |      -0.119659  | 0.0105582  | 0           |        -0.112777  |        -0.0113343  | True                       |
| day2_minus_day1         | B       |   nan | structured  |        -0.119749  |  0.00606268 | 0          |          -0.112857  |      -0.119802  | 0.00606507 | 0           |        -0.112904  |        -0.00468929 | True                       |
| B_minus_A               | nan     |     1 | random      |         0.0806322 |  0.102073   | 0.429558   |           0.0839722 |       0.0811134 | 0.102065   | 0.426776    |         0.0844939 |         0.0521662  | True                       |
| B_minus_A               | nan     |     1 | structured  |         0.0501896 |  0.101727   | 0.621747   |           0.0514705 |       0.049856  | 0.10172    | 0.624042    |         0.0511197 |        -0.0350775  | True                       |
| B_minus_A               | nan     |     2 | random      |         0.0585098 |  0.102105   | 0.56662    |           0.0602554 |       0.0585326 | 0.102096   | 0.566435    |         0.0602795 |         0.00241441 | True                       |
| B_minus_A               | nan     |     2 | structured  |         0.074125  |  0.101738   | 0.466256   |           0.0769415 |       0.0740799 | 0.101731   | 0.466495    |         0.0768929 |        -0.00485793 | True                       |


### Supplementary Table S4b. Difference-in-Differences (DoD) comparison

| model           |    log_DoD |        SE |          p |   pct_change |
|:----------------|-----------:|----------:|-----------:|-------------:|
| linear          | -0.0460578 | 0.016831  | 0.00620986 |   -0.0450133 |
| flexible_bs_df4 | -0.0468047 | 0.0167972 | 0.00532879 |   -0.0457263 |


### Supplementary Table S4c. Model fit comparison (secondary)

| model           | formula                                                                                               |      AIC |      BIC |   logLik |   df_modelwc |   n_obs |      LLR |         LLR_p |
|:----------------|:------------------------------------------------------------------------------------------------------|---------:|---------:|---------:|-------------:|--------:|---------:|--------------:|
| linear          | logRT ~ C(Group) * C(day) * C(condition) + BlockNumber + Age + fuglmayrshort_sum + MoCa_sum           | -6162.36 | -6064.09 |  3095.18 |           13 |    8261 | nan      | nan           |
| flexible_bs_df4 | logRT ~ C(Group) * C(day) * C(condition) + bs(BlockNumber, df=4) + Age + fuglmayrshort_sum + MoCa_sum | -6193.18 | -6073.85 |  3113.59 |           16 |    8261 | nan      | nan           |
| comparison      | flexible_vs_linear                                                                                    |   nan    |   nan    |   nan    |            3 |    8261 |  36.8164 |   5.03198e-08 |