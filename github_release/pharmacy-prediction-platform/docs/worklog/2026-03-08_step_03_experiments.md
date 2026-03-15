# Step 03 Experiments

## 时间

2026-03-08

## 本步目标

先通过探索性实验验证升级路线是否确实有提升潜力，再正式重写主流程。

## 实验设置

- 数据：Lipophilicity 真实公开数据
- 划分：与上一版一致的分层随机划分
- 增强特征：
  - 训练集筛选后的 RDKit 全量描述符
  - Morgan Fingerprint
  - MACCS Keys
- 模型：XGBoost / CatBoost / PyTorch MLP / 验证集加权集成

## 试验结果（探索性）

- `XGB`：Test `RMSE ≈ 0.6575`
- `CatBoost`：Test `RMSE ≈ 0.6686`
- `MLP`：Test `RMSE ≈ 0.6676`
- `Ensemble`：Test `RMSE ≈ 0.6370`

## 结论

探索性实验已经明显优于上一版最佳模型：

- 上一版：`RMSE = 0.6787`
- 探索性集成：`RMSE ≈ 0.6370`

因此确认升级路线可行，进入正式实现阶段。
