# Step 04 Implementation

## 时间

2026-03-08

## 本步目标

将探索性实验固化为正式项目代码与输出流程。

## 实现内容

### 1. 重写特征工程模块

更新 `src/featurization.py`：

- 增加 RDKit 全量描述符提取
- 增加 MACCS Keys 提取
- 新增 `DescriptorProcessor`，在训练集内完成：
  - 缺失列剔除
  - 零方差列剔除
  - 高相关列剔除
  - 标准化

### 2. 重写训练模块

更新 `src/train_model.py`：

- 增加 `train_catboost`
- 保留并扩展 `train_xgboost`
- 优化 `PyTorch MLP` 结构
- 增加 `optimize_ensemble_weights`
- 增加 `blend_predictions`

### 3. 增加结构分析模块

新增 `src/structure_analysis.py`：

- 化学空间投影
- 高频 scaffold 统计
- scaffold 网格图
- 极端样本分子结构图

### 4. 重写主流程

更新 `run.py`：

- 同时输出基线模型与增强模型
- 保存增强前后对比结果
- 自动生成 `final_results_v2.md`
- 自动生成 `final_results_v2.m`

### 5. 更新文档

- 更新 `README.md`
- 更新 `QUICK_START.md`
- 更新 `report.md`
- 更新 `requirements.txt`

## 实现原则

- 不更换数据集，保证与上一版可直接对比
- 不额外创建新环境，因为 `cptac` 环境已经满足增强方案依赖
- 所有输出保持可复现
