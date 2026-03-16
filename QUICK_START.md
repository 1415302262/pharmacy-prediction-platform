# Quick Start

最短路径运行这个项目时，看这份文件；项目背景、方法和结果说明请看 `README.md`。

## 1. Environment

推荐直接使用现有环境：

```bash
conda activate cptac
```

如果你准备新建环境，请至少确保安装了 `requirements.txt` 中列出的核心依赖，并具备 `RDKit`、`XGBoost`、`CatBoost` 和 `PyTorch`。

## 2. Run the Main Pipeline

在项目根目录执行：

```bash
python run.py
```

这一步会完成主流程训练、评估和结果导出。

## 3. Run Strict Validation and SHAP

如果你想重跑更严格的 scaffold-aware 验证和解释分析：

```bash
python scripts/run_scaffold_shap_validation.py
```

## 4. Launch the Web Demo

本地启动 Flask 演示平台：

```bash
cd hf_space
python app.py
```

默认地址：`http://127.0.0.1:7860`

## 5. Key Outputs

- Main results: `results/final_results_v2.md`
- Test predictions: `results/test_predictions.csv`
- Scaffold validation: `results/scaffold_validation_results.md`
- Models: `models/`
- Work logs: `docs/worklog/`

## 6. Current Best Metrics

- Best model: `Weighted_Ensemble_rich`
- Test `R² = 0.7232`
- Test `RMSE = 0.6346`
- Test `MAE = 0.4517`

## 7. API Example

接口：`POST /api/predict`

```json
{
  "smiles": "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14"
}
```
