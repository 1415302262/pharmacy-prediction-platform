# Step 06 Scaffold Validation

## 时间

2026-03-13

## 本步目标

增加比随机划分更严格的 scaffold-aware 重复验证，提升科研可信度。

## 方法

- 使用 5 个随机种子进行 repeated scaffold-aware split
- 每次划分中保持 train / valid / test 的 scaffold 尽量分离
- 评估模型：
  - `XGB_baseline_scaffold`
  - `XGB_rich_scaffold`
  - `CatBoost_rich_scaffold`
  - `PyTorch_MLP_rich_scaffold`
  - `Weighted_Ensemble_scaffold`

## 结果

| 模型 | R² mean ± std | RMSE mean ± std |
|---|---:|---:|
| Weighted_Ensemble_scaffold | 0.7138 ± 0.0215 | 0.6489 ± 0.0296 |
| XGB_baseline_scaffold | 0.6472 ± 0.0311 | 0.7200 ± 0.0299 |

## 结论

增强版模型在严格 scaffold 验证下依然稳定优于基线，这对复试中的科研说服力帮助很大。
