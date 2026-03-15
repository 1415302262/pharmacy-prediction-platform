# 药学复试增强版项目：Lipophilicity QSAR + 结构特征增强 + 集成学习

这是一个基于**真实公开数据集**完成的药物性质预测项目，任务是：

> **根据分子结构预测实验脂溶性（lipophilicity / logD）**

相较于上一版，本次增强版加入了：

- 更丰富的结构信息：`RDKit 全量2D描述符 + Morgan Fingerprint + MACCS keys`
- 更完整的结构分析：化学空间、Bemis–Murcko scaffold、代表性结构网格图
- 更强的建模方案：`XGBoost + CatBoost + PyTorch MLP + 加权集成`
- 更完整的留痕：所有关键步骤保存到 `docs/worklog/*.md`
- 结果导出：同时生成 `results/final_results_v2.md` 和 `results/final_results_v2.m`

---

## 1. 数据来源

- 数据集：MoleculeNet / DeepChem `Lipophilicity`
- 下载地址：`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv`
- 样本量：4200 个分子
- 字段：`CMPD_CHEMBLID`、`exp`、`smiles`
- 目标：实验脂溶性 `exp`

该数据属于公开真实数据，不是模拟数据，且含有 `ChEMBL` 编号，适合在药学复试中说明“数据真实、可追溯、与药物数据库相关”。

---

## 2. 本次增强点

### 2.1 结构特征增强

项目现在同时使用三类分子表征：

1. **基础理化描述符（13维）**
2. **RDKit 全量2D描述符（训练集筛选后保留 173 维）**
3. **结构指纹**
   - Morgan Fingerprint：2048维
   - MACCS Keys：167维

增强后，模型不再只看“少量理化性质”，而是同时利用：

- 分子整体大小、极性、氢键能力
- 局部子结构片段
- 常见结构键模式与药效团相关编码

### 2.2 结构分析增强

新增结构相关输出：

- `figures/08_chemical_space.png`：化学空间投影图
- `figures/09_top_scaffolds.png`：高频 scaffold 统计
- `figures/10_top_scaffold_grid.png`：高频 scaffold 结构图
- `figures/11_extreme_molecule_grid.png`：高/低脂溶性代表分子结构图
- `figures/12_feature_family_importance.png`：描述符 / Morgan / MACCS 贡献比较
- `figures/13_ensemble_weights.png`：集成权重图
- `figures/14_baseline_vs_enhanced.png`：增强前后效果对比图

### 2.3 模型增强

当前项目包含两类模型：

- **基线模型**
  - `RF_basic_desc`
  - `SVR_basic_desc`
  - `XGB_baseline_desc_fp`

- **增强模型**
  - `XGB_rich_desc_fp_maccs`
  - `CatBoost_rich_desc_fp_maccs`
  - `PyTorch_MLP_rich_desc_fp_maccs`
  - `Weighted_Ensemble_rich`

---

## 3. 最终结果

### 3.1 当前最佳结果

| 模型 | R² | RMSE | MAE |
|---|---:|---:|---:|
| Weighted_Ensemble_rich | 0.7232 | 0.6346 | 0.4517 |
| XGB_rich_desc_fp_maccs | 0.7043 | 0.6559 | 0.4810 |
| CatBoost_rich_desc_fp_maccs | 0.6973 | 0.6637 | 0.4862 |
| PyTorch_MLP_rich_desc_fp_maccs | 0.6967 | 0.6643 | 0.4634 |
| XGB_baseline_desc_fp | 0.6835 | 0.6787 | 0.5075 |

### 3.2 相比上一版的提升

- 上一版最佳模型：`XGB_desc_fp`
  - Test `R² = 0.6835`
  - Test `RMSE = 0.6787`

- 本次最佳模型：`Weighted_Ensemble_rich`
  - Test `R² = 0.7232`
  - Test `RMSE = 0.6346`

### 3.3 提升幅度

- `RMSE` 降低：`0.0440`
- `R²` 提升：`0.0397`

这说明：

1. 增加结构表征是有效的；
2. CatBoost / XGBoost / 深度学习模型之间存在互补性；
3. 验证集优化的加权集成能够进一步提升泛化性能。

---


## 3.4 严格验证（scaffold-aware repeated split）

为了提升复试中的科研说服力，我进一步加入了 **5 次重复的 scaffold-aware 严格验证**。

> 说明：该验证方式比随机划分更严格，因此分数通常略低，但对“真实泛化能力”的证明更强。

严格验证平均结果如下：

| 模型 | R² mean ± std | RMSE mean ± std |
|---|---:|---:|
| Weighted_Ensemble_scaffold | 0.7138 ± 0.0215 | 0.6489 ± 0.0296 |
| PyTorch_MLP_rich_scaffold | 0.6945 ± 0.0261 | 0.6698 ± 0.0200 |
| XGB_rich_scaffold | 0.6847 ± 0.0234 | 0.6811 ± 0.0347 |
| CatBoost_rich_scaffold | 0.6750 ± 0.0234 | 0.6914 ± 0.0295 |
| XGB_baseline_scaffold | 0.6472 ± 0.0311 | 0.7200 ± 0.0299 |

这说明即使在更严格的 scaffold 划分下，增强版模型依然明显优于基线模型。

## 3.5 SHAP 解释增强

为了避免项目只停留在“黑箱预测”，我又加入了 SHAP 解释分析：

- `results/shap_top20_features.csv`
- `results/shap_case_studies.md`
- `figures/19_shap_top20_bar.png`
- `figures/20_shap_beeswarm_top20.png`
- `figures/21_shap_local_highprediction.png`
- `figures/21_shap_local_lowprediction.png`
- `figures/21_shap_local_largeerror.png`
- `figures/22_shap_case_molecules.png`

前 5 个全局重要特征为：

1. `MolLogP`
2. `fr_COO`
3. `VSA_EState5`
4. `MACCS_104`
5. `SMR_VSA10`

这让你在复试时不仅能讲“模型准不准”，还能讲“模型为什么这么判断”。

## 4. 为什么这个项目更适合复试

增强版更适合复试的原因：

- **数据真实**：公开数据集，可追溯
- **药学相关**：脂溶性与 ADME 密切相关
- **结构信息更充分**：不仅有指纹，还有 scaffold 与化学空间分析
- **模型更完整**：传统机器学习 + 深度学习 + 集成学习
- **结果更强**：实际测试性能优于上一版
- **材料更齐全**：报告、图、表、工作留痕、结果导出脚本全部齐备

---

## 5. 关键输出文件

### 结果文件

- `results/metrics_summary.csv`
- `results/test_predictions.csv`
- `results/run_summary.json`
- `results/baseline_vs_enhanced.csv`
- `results/final_results_v2.md`
- `results/final_results_v2.m`

### 模型文件

- `models/xgb_baseline_desc_fp.json`
- `models/xgb_rich_desc_fp_maccs.json`
- `models/catboost_rich_desc_fp_maccs.cbm`
- `models/pytorch_mlp_rich_desc_fp_maccs.pt`
- `models/rich_descriptor_processor.pkl`

### 留痕文件

- `docs/worklog/2026-03-08_step_01_audit.md`
- `docs/worklog/2026-03-08_step_02_design.md`
- `docs/worklog/2026-03-08_step_03_experiments.md`
- `docs/worklog/2026-03-08_step_04_implementation.md`
- `docs/worklog/2026-03-08_step_05_final_results.md`

---


## 5.1 在线演示平台（Hugging Face Space）

项目已新增一个可公开部署的 Web 演示平台，位置：

- `hf_space/app.py`
- `hf_space/templates/index.html`
- `hf_space/static/style.css`
- `hf_space/Dockerfile`
- `hf_space/README.md`

平台支持：

- 单分子 `SMILES` 预测
- 集成模型结果展示
- 严格 scaffold 验证结果展示
- SHAP 可解释性内容展示
- JSON API 调用（`POST /api/predict`）

### 平台技术选型

这次采用的是：

- **Flask**：负责网页与 API
- **Docker**：用于 Hugging Face Spaces 部署
- **预训练模型加载**：不在平台上训练，只做推理

### 为什么这样设计

- 更适合做成“作品级展示页面”
- 能兼顾网页可视化和 JSON API
- 适合后续部署到 Hugging Face Spaces 的 `docker` SDK

### 本地启动方式

```bash
cd /public/home/zhw/cptac/projects/experiment/qsar_project/hf_space
python app.py
```

默认端口：`7860`

### 刷新 Hugging Face 打包资产

如果主项目模型或结果更新，可重新执行：

```bash
python scripts/build_hf_space_bundle.py
```

这会把平台所需的模型、结果和精选图表同步到 `hf_space/assets/`。

## 6. 运行方式

```bash
conda activate cptac
cd /public/home/zhw/cptac/projects/experiment/qsar_project
python run.py
```

---

## 7. 复试时可以怎么讲

### 30秒版本

> 我做了一个基于真实公开药物数据的 QSAR 项目，任务是根据分子结构预测实验脂溶性。第一版使用基础描述符和分子指纹，第二版进一步加入 RDKit 全量描述符、MACCS 结构键，并结合 XGBoost、CatBoost 和 PyTorch MLP 做加权集成，最终把测试集 R² 从 0.683 提升到 0.723，RMSE 从 0.679 降到 0.635。同时我还做了 scaffold 和化学空间分析，使这个项目不仅有预测结果，也有结构层面的解释。

---

## 8. 项目结构

```text
qsar_project/
├── data/
├── docs/
│   └── worklog/
├── figures/
├── models/
├── results/
├── src/
│   ├── data_utils.py
│   ├── featurization.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── structure_analysis.py
├── run.py
├── QUICK_START.md
└── report.md
```

---

## 9. 下一步还能扩展什么

- 更严格的 `scaffold split`
- Graph Neural Network（GNN）
- SHAP 解释增强
- 外部验证集
- 多任务性质联合预测
