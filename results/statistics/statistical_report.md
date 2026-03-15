# Statistical Analysis Report


## Dataset Statistics

- name: Test Set Target
- n: 631
- mean: 2.1853
- std: 1.2072
- min: -1.4800
- max: 4.4800
- median: 2.3600
- q25: 1.4150
- q75: 3.1500
- skewness: -0.6363
- kurtosis: -0.0838
- cv: 0.5524

## Normality Tests

- shapiro_statistic: 0.9520
- shapiro_pvalue: 0.0000
- shapiro_normal: False
- dagostino_statistic: 73.6965
- dagostino_pvalue: 0.0000
- dagostino_normal: False
- n: 631

## Ensemble Bootstrap Results

- r2: {'mean': 0.7232771356768716, 'std': 0.028189942061782356, 'ci_lower': 0.6632955431272762, 'ci_upper': 0.7752344479781758}
- rmse: {'mean': 0.6342963576316833, 'std': 0.031216993927955627, 'ci_lower': 0.5739140376448632, 'ci_upper': 0.6972868084907532}
- mae: {'mean': 0.4519859552383423, 'std': 0.017621256411075592, 'ci_lower': 0.4179944068193436, 'ci_upper': 0.48604564368724823}

## Pairwise Comparisons

                        model1                         model2  difference  p_value  significant  ci_lower  ci_upper
        XGB_rich_desc_fp_maccs    CatBoost_rich_desc_fp_maccs   -0.010213 0.459441        False -0.039107  0.016066
        XGB_rich_desc_fp_maccs PyTorch_MLP_rich_desc_fp_maccs   -0.011097 0.632751        False -0.065856  0.034848
        XGB_rich_desc_fp_maccs         Weighted_Ensemble_rich    0.027499 0.022572         True -0.000466  0.049585
   CatBoost_rich_desc_fp_maccs PyTorch_MLP_rich_desc_fp_maccs   -0.000884 0.967089        False -0.046091  0.044496
   CatBoost_rich_desc_fp_maccs         Weighted_Ensemble_rich    0.037713 0.002139         True  0.015730  0.063605
PyTorch_MLP_rich_desc_fp_maccs         Weighted_Ensemble_rich    0.038597 0.001224         True  0.014327  0.066292
