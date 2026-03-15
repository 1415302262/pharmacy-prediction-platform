# Enhanced QSAR Results

## Dataset
- Molecules: 4200
- Target: experimental lipophilicity (logD)
- Range: -1.50 to 4.50

## Test Metrics
| Model | R2 | RMSE | MAE |
|---|---:|---:|---:|
| Weighted_Ensemble_rich | 0.7232 | 0.6346 | 0.4517 |
| XGB_rich_desc_fp_maccs | 0.7043 | 0.6559 | 0.4810 |
| CatBoost_rich_desc_fp_maccs | 0.6973 | 0.6637 | 0.4862 |
| PyTorch_MLP_rich_desc_fp_maccs | 0.6967 | 0.6643 | 0.4634 |
| XGB_baseline_desc_fp | 0.6835 | 0.6787 | 0.5075 |
| SVR_basic_desc | 0.4802 | 0.8697 | 0.6615 |
| RF_basic_desc | 0.4728 | 0.8758 | 0.6578 |

## Improvement vs Baseline
| Version | Model | R2 | RMSE | MAE |
|---|---|---:|---:|---:|
| baseline_20260307 | XGB_desc_fp | 0.6835 | 0.6787 | 0.5075 |
| enhanced_20260308 | Weighted_Ensemble_rich | 0.7232 | 0.6346 | 0.4517 |

## Ensemble Weights
- XGB_rich_desc_fp_maccs: 0.3798
- CatBoost_rich_desc_fp_maccs: 0.1421
- PyTorch_MLP_rich_desc_fp_maccs: 0.4781

## Best Model
- Weighted_Ensemble_rich
