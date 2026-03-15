# QUICK START

## 1. 激活环境

```bash
conda activate cptac
```

## 2. 进入项目目录

```bash
cd /public/home/zhw/cptac/projects/experiment/qsar_project
```

## 3. 一键运行增强版

```bash
python run.py
```

## 4. 当前最佳结果

- 最佳模型：`Weighted_Ensemble_rich`
- Test `R² = 0.7232`
- Test `RMSE = 0.6346`
- Test `MAE = 0.4517`

## 5. 相比上一版的提升

- 上一版最佳：`XGB_desc_fp`
  - Test `R² = 0.6835`
  - Test `RMSE = 0.6787`

- 当前增强版最佳：`Weighted_Ensemble_rich`
  - Test `R² = 0.7232`
  - Test `RMSE = 0.6346`

## 5.1 严格验证与 SHAP（推荐额外展示）

如需重跑更严格的科研验证与解释分析：

```bash
python scripts/run_scaffold_shap_validation.py
```

当前严格验证最佳平均结果：

- 最佳模型：`Weighted_Ensemble_scaffold`
- Mean `R² = 0.7138 ± 0.0215`
- Mean `RMSE = 0.6489 ± 0.0296`

## 6. 重点看哪些文件

### 图像
- `figures/02_model_comparison.png`
- `figures/05_xgb_rich_test_scatter.png`
- `figures/06_ensemble_test_scatter.png`
- `figures/08_chemical_space.png`
- `figures/09_top_scaffolds.png`
- `figures/10_top_scaffold_grid.png`
- `figures/14_baseline_vs_enhanced.png`

### 结果文件
- `results/metrics_summary.csv`
- `results/baseline_vs_enhanced.csv`
- `results/final_results_v2.md`
- `results/final_results_v2.m`

### 留痕文件
- `docs/worklog/2026-03-08_step_01_audit.md`
- `docs/worklog/2026-03-08_step_02_design.md`
- `docs/worklog/2026-03-08_step_03_experiments.md`
- `docs/worklog/2026-03-08_step_04_implementation.md`
- `docs/worklog/2026-03-08_step_05_final_results.md`

## 7. 复试版一句话介绍

> 本项目基于公开真实 Lipophilicity 数据集，在原有 QSAR 流程上进一步加入全量 RDKit 描述符、Morgan 指纹、MACCS 结构键、结构可视化分析以及 XGBoost / CatBoost / PyTorch MLP 集成建模，使测试集表现从 R² 0.683 提升到 0.723，形成了一个更完整、更适合药学复试展示的计算药学项目。


## 8. 在线平台（Flask + Hugging Face Spaces）

### 本地运行

```bash
cd /public/home/zhw/cptac/projects/experiment/qsar_project/hf_space
python app.py
```

打开：`http://127.0.0.1:7860`

### JSON API

接口：`POST /api/predict`

请求示例：

```json
{
  "smiles": "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14"
}
```

### 打包到 Hugging Face Space

```bash
python scripts/build_hf_space_bundle.py
```

然后把 `hf_space/` 目录内容推送到 Hugging Face Space 仓库即可。
