# Step 07 SHAP Interpretation

## 时间

2026-03-13

## 本步目标

加入 SHAP 全局与局部解释，让项目从“会预测”升级到“会解释”。

## 全局结果

根据 `results/shap_top20_features.csv`，最重要的前 5 个特征为：

1. `MolLogP`
2. `fr_COO`
3. `VSA_EState5`
4. `MACCS_104`
5. `SMR_VSA10`

## 局部案例

选取了 3 个代表性样本：

- `HighPrediction`
- `LowPrediction`
- `LargeError`

并输出：

- 局部 SHAP 条形图
- 分子结构图
- 文字案例分析

## 结论

SHAP 解释使项目更容易在复试中回答“模型为什么这么判断”，能明显提升表达深度与科研感。
