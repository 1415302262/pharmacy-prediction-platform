# Step 05 Final Results

## 时间

2026-03-08

## 本步目标

记录增强版正式运行后的最终结果，并与上一版做直接对比。

## 正式结果

### 增强版测试集表现

| 模型 | R² | RMSE | MAE |
|---|---:|---:|---:|
| Weighted_Ensemble_rich | 0.7232 | 0.6346 | 0.4517 |
| XGB_rich_desc_fp_maccs | 0.7043 | 0.6559 | 0.4810 |
| CatBoost_rich_desc_fp_maccs | 0.6973 | 0.6637 | 0.4862 |
| PyTorch_MLP_rich_desc_fp_maccs | 0.6967 | 0.6643 | 0.4634 |
| XGB_baseline_desc_fp | 0.6835 | 0.6787 | 0.5075 |

### 增强前后最佳模型对比

| 版本 | 最佳模型 | R² | RMSE | MAE |
|---|---|---:|---:|---:|
| baseline_20260307 | XGB_desc_fp | 0.6835 | 0.6787 | 0.5075 |
| enhanced_20260308 | Weighted_Ensemble_rich | 0.7232 | 0.6346 | 0.4517 |

## 提升总结

- `R²` 提升：`+0.0397`
- `RMSE` 降低：`-0.0440`
- `MAE` 降低：`-0.0558`

## 新增输出内容

- 结构可视化图
- scaffold 统计图
- 化学空间图
- 集成权重图
- 增强前后对比图
- `results/final_results_v2.md`
- `results/final_results_v2.m`

## 最终判断

增强版已满足本轮目标：

1. 结果优于上一版；
2. 项目更完整、更丰满；
3. 结构内容显著增加；
4. 深度学习部分保留且有效；
5. 留痕完整，适合后续继续扩展。
