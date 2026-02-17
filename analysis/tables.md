## Table 2. Primary group effects (adjusted GLM, HC3)

| Outcome               |   Group effect B vs A (coef) | Group effect 95% CI   |   Group effect p |   Group×Day interaction (coef) | Group×Day 95% CI    | Group×Day p   |
|:----------------------|-----------------------------:|:----------------------|-----------------:|-------------------------------:|:--------------------|:--------------|
| ErrorRate_All         |                    0.0112014 | [-0.031, 0.053]       |            0.6   |                     -0.0291566 | [-0.099, 0.041]     | 0.415         |
| IES                   |                  103.055     | [-122.751, 328.860]   |            0.371 |                    -44.5818    | [-340.680, 251.516] | 0.768         |
| RT_Delta_All          |                   -3.38242   | [-87.082, 80.317]     |            0.937 |                      0.0468158 | [-100.365, 100.459] | 0.999         |
| RT_Slope_All          |                   -0.0120187 | [-0.853, 0.829]       |            0.978 |                      0.134154  | [-0.900, 1.168]     | 0.799         |
| Retention_General     |                   14.3586    | [-96.736, 125.453]    |            0.8   |                    nan         | NA                  | NA            |
| Retention_Sequence    |                   37.8916    | [-54.493, 130.276]    |            0.421 |                    nan         | NA                  | NA            |
| SeqLearning_Index_all |                   37.8891    | [0.404, 75.374]       |            0.048 |                    -43.3999    | [-99.231, 12.432]   | 0.128         |


## Table 3. Mixed-model simple effects (ratio scale)

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


## Supplementary Table S1. Missingness / data-quality sensitivity

| scenario             |   n_participant_day |   n_retention | long_session_excluded   |   seq_group_coef |   Seq Group p |   ret_group_coef |   Retention Group p |
|:---------------------|--------------------:|--------------:|:------------------------|-----------------:|--------------:|-----------------:|--------------------:|
| baseline             |                  73 |            34 | nan                     |          37.8891 |         0.048 |          37.8916 |               0.421 |
| truncate_119         |                  73 |            34 | nan                     |          38.954  |         0.041 |          37.8354 |               0.423 |
| truncate_120         |                  73 |            34 | nan                     |          39.0688 |         0.04  |          37.8491 |               0.422 |
| exclude_long_session |                  72 |            33 | 9_day1                  |          39.1919 |         0.045 |          37.9142 |               0.427 |


## Supplementary Table S2. Permutation tests

| outcome                    |   observed_B_minus_A |   Permutation p (two-sided) |   n_perm |
|:---------------------------|---------------------:|----------------------------:|---------:|
| SeqLearning_Index_all_day2 |             -9.66436 |                       0.611 |    10000 |
| Retention_Sequence         |              2.18207 |                       0.961 |    10000 |


## Supplementary Table S3. Clinical stratification

| stratum   |   n | B vs A (95% CI)           |     p |
|:----------|----:|:--------------------------|------:|
| MoCa_low  |  14 | 5.874 [-80.944, 92.692]   | 0.895 |
| MoCa_high |  22 | -16.018 [-61.947, 29.911] | 0.494 |
| GDS_low   |  32 | -12.875 [-67.001, 41.252] | 0.641 |
| Fugl_high |  27 | -4.522 [-59.936, 50.893]  | 0.873 |


## Supplementary Table S4. Time-trend robustness (linear vs flexible)

Planned contrasts (linear vs flexible block-wise trend), including material change criterion (|delta| < 2 percentage points).

| contrast                | Group   |   Day | Condition   |   log_diff_linear |   SE_linear |    p_linear |   pct_change_linear |   log_diff_flex |    SE_flex |      p_flex |   pct_change_flex |   delta_pct_points | materially_unchanged_2pp   |
|:------------------------|:--------|------:|:------------|------------------:|------------:|------------:|--------------------:|----------------:|-----------:|------------:|------------------:|-------------------:|:---------------------------|
| structured_minus_random | A       |     1 | nan         |        -0.0574298 |  0.00847104 | 1.20555e-11 |          -0.0558118 |      -0.0561174 | 0.00846694 | 3.40648e-11 |        -0.0545719 |         0.123989   | True                       |
| structured_minus_random | A       |     2 | nan         |        -0.0981324 |  0.00779394 | 0           |          -0.0934712 |      -0.0981026 | 0.00777665 | 0           |        -0.0934442 |         0.00270027 | True                       |
| structured_minus_random | B       |     1 | nan         |        -0.0846545 |  0.00801819 | 0           |          -0.0811703 |      -0.0839385 | 0.00802776 | 0           |        -0.0805122 |         0.0658144  | True                       |
| structured_minus_random | B       |     2 | nan         |        -0.0790822 |  0.00872815 | 0           |          -0.0760361 |      -0.0791275 | 0.00870866 | 0           |        -0.0760779 |        -0.00418391 | True                       |
| day2_minus_day1         | A       |   nan | random      |        -0.108872  |  0.0101139  | 0           |          -0.103155  |      -0.108032  | 0.0101003  | 0           |        -0.102401  |         0.0754215  | True                       |
| day2_minus_day1         | A       |   nan | structured  |        -0.149575  |  0.00584475 | 0           |          -0.138926  |      -0.150017  | 0.00583377 | 0           |        -0.139306  |        -0.0380446  | True                       |
| day2_minus_day1         | B       |   nan | random      |        -0.12393   |  0.0103785  | 0           |          -0.116558  |      -0.123424  | 0.0103877  | 0           |        -0.116112  |         0.0446764  | True                       |
| day2_minus_day1         | B       |   nan | structured  |        -0.118358  |  0.00600371 | 0           |          -0.111622  |      -0.118613  | 0.00600571 | 0           |        -0.111849  |        -0.0227156  | True                       |
| B_minus_A               | nan     |     1 | random      |         0.0425273 |  0.101089   | 0.673979    |           0.0434445 |       0.0429328 | 0.101075   | 0.671012    |         0.0438677 |         0.0423188  | True                       |
| B_minus_A               | nan     |     1 | structured  |         0.0153025 |  0.100743   | 0.879269    |           0.0154202 |       0.0151117 | 0.100731   | 0.880748    |         0.0152265 |        -0.0193733  | True                       |
| B_minus_A               | nan     |     2 | random      |         0.0274694 |  0.101086   | 0.785819    |           0.0278502 |       0.0275399 | 0.101072   | 0.785254    |         0.0279226 |         0.007242   | True                       |
| B_minus_A               | nan     |     2 | structured  |         0.0465196 |  0.100738   | 0.644234    |           0.0476186 |       0.046515  | 0.100726   | 0.644225    |         0.0476138 |        -0.00048342 | True                       |


### Supplementary Table S4b. Difference-in-Differences (DoD) comparison

| model           |    log_DoD |        SE |          p |   pct_change |
|:----------------|-----------:|----------:|-----------:|-------------:|
| linear          | -0.0462749 | 0.0165189 | 0.00508908 |   -0.0452206 |
| flexible_bs_df4 | -0.0467962 | 0.0164834 | 0.00452575 |   -0.0457181 |


### Supplementary Table S4c. Model fit comparison (secondary)

| model           | formula                                                                                               |      AIC |      BIC |   logLik |   df_modelwc |   n_obs |      LLR |         LLR_p |
|:----------------|:------------------------------------------------------------------------------------------------------|---------:|---------:|---------:|-------------:|--------:|---------:|--------------:|
| linear          | logRT ~ C(Group) * C(day) * C(condition) + BlockNumber + Age + fuglmayrshort_sum + MoCa_sum           | -6447.1  | -6348.43 |  3237.55 |           13 |    8506 | nan      | nan           |
| flexible_bs_df4 | logRT ~ C(Group) * C(day) * C(condition) + bs(BlockNumber, df=4) + Age + fuglmayrshort_sum + MoCa_sum | -6481.07 | -6361.25 |  3257.54 |           16 |    8506 | nan      | nan           |
| comparison      | flexible_vs_linear                                                                                    |   nan    |   nan    |   nan    |            3 |    8506 |  39.9684 |   1.08208e-08 |