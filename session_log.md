# QSAR项目对话记录

**项目日期**: 2026-03-06
**项目路径**: `/public/home/zhw/cptac/projects/experiment/qsar_project`

---

## 项目目标

使用真实数据创建一个适合药学研究生复试的QSAR（定量构效关系）机器学习项目。

**项目要求**:
1. 使用真实数据（ChEMBL数据库EGFR抑制剂数据）
2. 适合药学复试展示
3. 基于coding的项目
4. 观察CPU利用率，避免服务器过载
5. 代码后台运行，保存log
6. 生成完整的项目结果报告

---

## 对话内容记录

### 用户初始需求

> "/public/home/zhw/cptac/projects/experiment/qsar_project这个目录下，要求用真实数据做一个适合药学研究生申请专业复试的基于coding的项目，请帮我完成一次之后再教导我怎么做，要求时刻观察CPU利用率，不要炸服务器，任何代码运行后台运行保存log，并且生成一个文件保存我们的聊天记录，并且最后把项目结果写成一个md文件，不要求做得很快，但要真实"

---

## 项目执行记录

*（以下内容会在对话过程中动态更新）*

---

## 2026-03-08 增强版升级记录

### 升级目标

在保持真实公开数据不变的前提下，让项目更完整、更适合复试展示，并保证效果优于上一版。

### 本次升级内容

1. 增加结构特征：
   - RDKit 全量2D描述符
   - Morgan Fingerprint
   - MACCS Keys
2. 增加模型：
   - CatBoost
   - 强化版 XGBoost
   - PyTorch MLP
   - 验证集加权集成
3. 增加结构分析：
   - 化学空间图
   - scaffold 统计图
   - scaffold 结构网格图
   - 极端样本结构图
4. 增加留痕与导出：
   - `docs/worklog/*.md`
   - `results/final_results_v2.md`
   - `results/final_results_v2.m`

### 最终提升结果

- 上一版最佳：`XGB_desc_fp`
  - Test `R² = 0.6835`
  - Test `RMSE = 0.6787`

- 增强版最佳：`Weighted_Ensemble_rich`
  - Test `R² = 0.7232`
  - Test `RMSE = 0.6346`

### 结论

本次增强版已成功达到“更完整、更丰满、效果更好”的目标，当前项目已具备更强的药学复试展示价值。

---

## 技术栈

- Python 3.9+
- RDKit（分子指纹计算）
- Scikit-learn（机器学习模型）
- Pandas（数据处理）
- Matplotlib（结果可视化）

---

## 项目结构

```
qsar_project/
├── data/                   # 数据目录
│   └── sample_data.csv    # SMILES和pIC50数据
├── src/                   # 源代码
│   ├── featurization.py  # 特征工程
│   ├── train_model.py    # 模型训练
│   └── evaluate.py       # 评估和可视化
├── figures/              # 结果图片
├── models/               # 模型文件
├── logs/                 # 运行日志
├── run.py               # 主运行脚本
└── session_log.md       # 本文件：对话记录
```

---

---

## 2026-03-13 严格验证与 SHAP 升级记录

### 升级目的

进一步提升项目的科研可信度与复试竞争力，不再只依赖随机划分结果，而是加入：

- repeated scaffold split
- SHAP 可解释性分析
- 代表性分子案例解释

### 严格验证结果

- 最佳严格验证模型：`Weighted_Ensemble_scaffold`
- Mean `R² = 0.7138 ± 0.0215`
- Mean `RMSE = 0.6489 ± 0.0296`

### SHAP 全局重要特征（前5）

1. `MolLogP`
2. `fr_COO`
3. `VSA_EState5`
4. `MACCS_104`
5. `SMR_VSA10`

### 结论

当前项目已经同时具备：

1. 真实公开数据
2. 结构增强建模
3. 深度学习与集成学习
4. 严格 scaffold 验证
5. SHAP 可解释性分析

这使它更接近一份“能讲科研逻辑”的复试项目，而不只是单纯的机器学习演示。

---

## 2026-03-13 Hugging Face Flask 平台开发记录

### 目标

将现有模型与结果打包成可公开访问的交互式网页平台，并保留 API 能力。

### 最终方案

- 平台目录：`hf_space/`
- 架构：`Flask + Docker`
- 场景：`Hugging Face Spaces`
- 功能：
  - 单分子预测
  - 结果展示
  - 严格验证展示
  - SHAP 解释展示
  - JSON API

### 本地验证

已完成：

- `GET /health`
- `POST /api/predict`

均已在 `cptac` 环境本地测试通过。

### 结论

项目现在已经同时具备：

1. 研究性
2. 工程性
3. 展示性
4. 可部署性

对复试来说，这比单纯提交代码或图表更有竞争力。
