from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from data_utils import DATASET_URL, download_dataset, load_and_prepare_dataset, summarize_dataset, stratified_random_split
from evaluate import (
    plot_dataset_overview,
    plot_descriptor_importance,
    plot_ensemble_weights,
    plot_feature_family_importance,
    plot_model_comparison,
    plot_performance_gain,
    plot_prediction_scatter,
    plot_residuals,
    plot_training_curve,
    save_metrics_table,
)
from featurization import DescriptorProcessor, featurize_dataframe
from structure_analysis import plot_chemical_space, plot_top_scaffolds, save_extreme_molecule_grid, save_top_scaffold_grid
from train_model import (
    blend_predictions,
    optimize_ensemble_weights,
    predict_mlp,
    regression_metrics,
    save_json,
    save_pickle,
    save_torch_model,
    set_global_seed,
    train_catboost,
    train_mlp,
    train_random_forest,
    train_svr,
    train_xgboost,
)


def ensure_dirs() -> None:
    for folder in ["data/raw", "data/processed", "figures", "models", "results", "results/archive", "docs/worklog"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)


def get_index(df: pd.DataFrame, split: str) -> np.ndarray:
    return df.index[df["split"] == split].to_numpy()


def feature_family_importance(model, n_desc: int, n_fp: int, n_maccs: int) -> dict[str, float]:
    importance = model.feature_importances_
    return {
        "RDKit descriptors": float(importance[:n_desc].sum()),
        "Morgan fingerprints": float(importance[n_desc : n_desc + n_fp].sum()),
        "MACCS keys": float(importance[n_desc + n_fp : n_desc + n_fp + n_maccs].sum()),
    }


def write_results_markdown(
    path: Path,
    dataset_summary: dict,
    metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    ensemble_weights: dict[str, float],
    best_model_name: str,
) -> None:
    test_df = metrics_df[metrics_df["split"] == "test"].sort_values("rmse").copy()
    lines = [
        "# Enhanced QSAR Results",
        "",
        "## Dataset",
        f"- Molecules: {dataset_summary['n_molecules']}",
        f"- Target: {dataset_summary['target_name']}",
        f"- Range: {dataset_summary['target_min']:.2f} to {dataset_summary['target_max']:.2f}",
        "",
        "## Test Metrics",
        "| Model | R2 | RMSE | MAE |",
        "|---|---:|---:|---:|",
    ]
    for row in test_df.itertuples(index=False):
        lines.append(f"| {row.model} | {row.r2:.4f} | {row.rmse:.4f} | {row.mae:.4f} |")

    lines.extend([
        "",
        "## Improvement vs Baseline",
        "| Version | Model | R2 | RMSE | MAE |",
        "|---|---|---:|---:|---:|",
    ])
    for row in comparison_df.itertuples(index=False):
        lines.append(f"| {row.version} | {row.model} | {row.r2:.4f} | {row.rmse:.4f} | {row.mae:.4f} |")

    lines.extend([
        "",
        "## Ensemble Weights",
    ])
    for model_name, weight in ensemble_weights.items():
        lines.append(f"- {model_name}: {weight:.4f}")

    lines.extend([
        "",
        f"## Best Model",
        f"- {best_model_name}",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_matlab_script(path: Path) -> None:
    script = """
metrics = readtable('results/metrics_summary.csv');
test_metrics = metrics(strcmp(metrics.split, 'test'), :);
[~, idx] = sort(test_metrics.rmse);
test_metrics = test_metrics(idx, :);
disp(test_metrics)

figure('Position', [100, 100, 1000, 420]);
subplot(1,2,1)
bar(categorical(test_metrics.model), test_metrics.rmse)
title('Enhanced QSAR Test RMSE')
ylabel('RMSE')

subplot(1,2,2)
bar(categorical(test_metrics.model), test_metrics.r2)
title('Enhanced QSAR Test R2')
ylabel('R2')

sgtitle('Enhanced Lipophilicity Modeling Results')
""".strip()
    path.write_text(script + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    set_global_seed(42)

    print("=" * 78)
    print("Enhanced QSAR / ADME Project: Structural Features + Ensemble Learning")
    print("=" * 78)
    print(f"Dataset URL: {DATASET_URL}")

    raw_path = download_dataset()
    print(f"[1/7] Raw dataset ready: {raw_path}")

    clean_df = load_and_prepare_dataset(raw_path)
    split_df = stratified_random_split(clean_df)
    summary = summarize_dataset(split_df)
    print(f"[2/7] Clean molecules: {summary['n_molecules']}")
    print(f"       target range: {summary['target_min']:.2f} to {summary['target_max']:.2f}")

    bundle = featurize_dataframe(split_df)
    train_idx = get_index(split_df, "train")
    valid_idx = get_index(split_df, "valid")
    test_idx = get_index(split_df, "test")

    X_basic_train_raw = bundle.X_desc_basic[train_idx]
    X_basic_valid_raw = bundle.X_desc_basic[valid_idx]
    X_basic_test_raw = bundle.X_desc_basic[test_idx]
    X_fp_train = bundle.X_fp[train_idx]
    X_fp_valid = bundle.X_fp[valid_idx]
    X_fp_test = bundle.X_fp[test_idx]
    X_maccs_train = bundle.X_maccs[train_idx]
    X_maccs_valid = bundle.X_maccs[valid_idx]
    X_maccs_test = bundle.X_maccs[test_idx]
    y_train = bundle.y[train_idx]
    y_valid = bundle.y[valid_idx]
    y_test = bundle.y[test_idx]

    basic_scaler = StandardScaler()
    X_basic_train_scaled = basic_scaler.fit_transform(X_basic_train_raw)
    X_basic_valid_scaled = basic_scaler.transform(X_basic_valid_raw)
    X_basic_test_scaled = basic_scaler.transform(X_basic_test_raw)
    X_baseline_train = np.concatenate([X_basic_train_scaled, X_fp_train], axis=1).astype(np.float32)
    X_baseline_valid = np.concatenate([X_basic_valid_scaled, X_fp_valid], axis=1).astype(np.float32)
    X_baseline_test = np.concatenate([X_basic_test_scaled, X_fp_test], axis=1).astype(np.float32)

    rich_processor = DescriptorProcessor(corr_threshold=0.98)
    X_rich_desc_train = rich_processor.fit_transform(bundle.X_desc_full.iloc[train_idx])
    X_rich_desc_valid = rich_processor.transform(bundle.X_desc_full.iloc[valid_idx])
    X_rich_desc_test = rich_processor.transform(bundle.X_desc_full.iloc[test_idx])
    X_rich_train = np.concatenate([X_rich_desc_train, X_fp_train, X_maccs_train], axis=1).astype(np.float32)
    X_rich_valid = np.concatenate([X_rich_desc_valid, X_fp_valid, X_maccs_valid], axis=1).astype(np.float32)
    X_rich_test = np.concatenate([X_rich_desc_test, X_fp_test, X_maccs_test], axis=1).astype(np.float32)

    plot_dataset_overview(split_df, ROOT / "figures/01_dataset_overview.png")
    plot_chemical_space(
        np.concatenate([X_rich_train, X_rich_valid, X_rich_test], axis=0),
        bundle.y,
        split_df["split"],
        ROOT / "figures/08_chemical_space.png",
    )
    plot_top_scaffolds(split_df, ROOT / "figures/09_top_scaffolds.png")
    save_top_scaffold_grid(split_df, ROOT / "figures/10_top_scaffold_grid.png")
    save_extreme_molecule_grid(split_df, ROOT / "figures/11_extreme_molecule_grid.png")

    metrics_rows = []
    predictions_test = pd.DataFrame(
        {
            "chembl_id": split_df.loc[test_idx, "chembl_id"].values,
            "canonical_smiles": split_df.loc[test_idx, "canonical_smiles"].values,
            "observed": y_test,
        }
    )

    print("[3/7] Training baseline models...")
    rf_model = train_random_forest(X_basic_train_raw, y_train)
    svr_model = train_svr(X_basic_train_raw, y_train)
    xgb_baseline_model = train_xgboost(
        X_baseline_train,
        y_train,
        X_baseline_valid,
        y_valid,
        use_gpu=True,
        n_estimators=700,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.6,
        reg_lambda=1.0,
    )

    baseline_models = {
        "RF_basic_desc": (rf_model, X_basic_valid_raw, X_basic_test_raw),
        "SVR_basic_desc": (svr_model, X_basic_valid_raw, X_basic_test_raw),
        "XGB_baseline_desc_fp": (xgb_baseline_model, X_baseline_valid, X_baseline_test),
    }

    for model_name, (model, X_valid, X_test) in baseline_models.items():
        valid_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)
        valid_metrics = regression_metrics(y_valid, valid_pred)
        test_metrics = regression_metrics(y_test, test_pred)
        metrics_rows.append({"model": model_name, "split": "valid", **valid_metrics})
        metrics_rows.append({"model": model_name, "split": "test", **test_metrics})
        predictions_test[model_name] = test_pred
        print(f"       {model_name:24s} | valid RMSE={valid_metrics['rmse']:.3f} | test RMSE={test_metrics['rmse']:.3f}")

    print("[4/7] Training enhanced models with richer structural features...")
    xgb_rich_model = train_xgboost(
        X_rich_train,
        y_train,
        X_rich_valid,
        y_valid,
        use_gpu=True,
        n_estimators=1200,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.70,
        reg_lambda=1.2,
        min_child_weight=2.0,
    )
    catboost_rich_model = train_catboost(
        X_rich_train,
        y_train,
        X_rich_valid,
        y_valid,
        use_gpu=True,
        depth=8,
        learning_rate=0.03,
        iterations=2000,
        l2_leaf_reg=5.0,
    )
    mlp_rich_model, mlp_history, mlp_device = train_mlp(
        X_train=X_rich_train,
        y_train=y_train,
        X_valid=X_rich_valid,
        y_valid=y_valid,
        use_gpu=True,
        epochs=80,
        batch_size=128,
        lr=8e-4,
        patience=14,
    )

    rich_valid_predictions = {
        "XGB_rich_desc_fp_maccs": xgb_rich_model.predict(X_rich_valid),
        "CatBoost_rich_desc_fp_maccs": catboost_rich_model.predict(X_rich_valid),
        "PyTorch_MLP_rich_desc_fp_maccs": predict_mlp(mlp_rich_model, X_rich_valid, device=mlp_rich_model.network[0].weight.device),
    }
    rich_test_predictions = {
        "XGB_rich_desc_fp_maccs": xgb_rich_model.predict(X_rich_test),
        "CatBoost_rich_desc_fp_maccs": catboost_rich_model.predict(X_rich_test),
        "PyTorch_MLP_rich_desc_fp_maccs": predict_mlp(mlp_rich_model, X_rich_test, device=mlp_rich_model.network[0].weight.device),
    }

    for model_name, valid_pred in rich_valid_predictions.items():
        test_pred = rich_test_predictions[model_name]
        valid_metrics = regression_metrics(y_valid, valid_pred)
        test_metrics = regression_metrics(y_test, test_pred)
        metrics_rows.append({"model": model_name, "split": "valid", **valid_metrics})
        metrics_rows.append({"model": model_name, "split": "test", **test_metrics})
        predictions_test[model_name] = test_pred
        print(f"       {model_name:24s} | valid RMSE={valid_metrics['rmse']:.3f} | test RMSE={test_metrics['rmse']:.3f}")

    ensemble_weights = optimize_ensemble_weights(y_valid, rich_valid_predictions)
    ensemble_valid_pred = blend_predictions(rich_valid_predictions, ensemble_weights)
    ensemble_test_pred = blend_predictions(rich_test_predictions, ensemble_weights)
    ensemble_valid_metrics = regression_metrics(y_valid, ensemble_valid_pred)
    ensemble_test_metrics = regression_metrics(y_test, ensemble_test_pred)
    metrics_rows.append({"model": "Weighted_Ensemble_rich", "split": "valid", **ensemble_valid_metrics})
    metrics_rows.append({"model": "Weighted_Ensemble_rich", "split": "test", **ensemble_test_metrics})
    predictions_test["Weighted_Ensemble_rich"] = ensemble_test_pred
    print(f"       {'Weighted_Ensemble_rich':24s} | valid RMSE={ensemble_valid_metrics['rmse']:.3f} | test RMSE={ensemble_test_metrics['rmse']:.3f}")

    metrics_df = save_metrics_table(metrics_rows, ROOT / "results/metrics_summary.csv")
    predictions_test.to_csv(ROOT / "results/test_predictions.csv", index=False)

    plot_model_comparison(metrics_df, ROOT / "figures/02_model_comparison.png")
    plot_descriptor_importance(rf_model, bundle.basic_feature_names, ROOT / "figures/03_rf_descriptor_importance.png")
    plot_training_curve(mlp_history, ROOT / "figures/04_mlp_training_curve.png")
    plot_feature_family_importance(
        feature_family_importance(xgb_rich_model, X_rich_desc_train.shape[1], X_fp_train.shape[1], X_maccs_train.shape[1]),
        ROOT / "figures/12_feature_family_importance.png",
    )
    plot_ensemble_weights(ensemble_weights, ROOT / "figures/13_ensemble_weights.png")

    test_metric_lookup = {
        row["model"]: {k: row[k] for k in ["r2", "rmse", "mae"]}
        for row in metrics_rows
        if row["split"] == "test"
    }
    plot_prediction_scatter(
        y_test,
        predictions_test["XGB_rich_desc_fp_maccs"].to_numpy(),
        "Enhanced XGBoost on rich structural features",
        test_metric_lookup["XGB_rich_desc_fp_maccs"],
        ROOT / "figures/05_xgb_rich_test_scatter.png",
    )
    plot_prediction_scatter(
        y_test,
        predictions_test["Weighted_Ensemble_rich"].to_numpy(),
        "Weighted ensemble of XGB + CatBoost + PyTorch MLP",
        test_metric_lookup["Weighted_Ensemble_rich"],
        ROOT / "figures/06_ensemble_test_scatter.png",
    )
    plot_residuals(
        y_test,
        predictions_test["Weighted_Ensemble_rich"].to_numpy(),
        "Residual analysis of best enhanced model",
        ROOT / "figures/07_best_model_residuals.png",
    )

    baseline_archive = pd.read_csv(ROOT / "results/archive/metrics_summary_baseline_20260307.csv")
    baseline_best = baseline_archive[baseline_archive["split"] == "test"].sort_values("rmse").iloc[0]
    enhanced_best = metrics_df[metrics_df["split"] == "test"].sort_values("rmse").iloc[0]
    comparison_df = pd.DataFrame(
        [
            {
                "version": "baseline_20260307",
                "model": baseline_best["model"],
                "r2": float(baseline_best["r2"]),
                "rmse": float(baseline_best["rmse"]),
                "mae": float(baseline_best["mae"]),
            },
            {
                "version": "enhanced_20260308",
                "model": enhanced_best["model"],
                "r2": float(enhanced_best["r2"]),
                "rmse": float(enhanced_best["rmse"]),
                "mae": float(enhanced_best["mae"]),
            },
        ]
    )
    comparison_df.to_csv(ROOT / "results/baseline_vs_enhanced.csv", index=False)
    plot_performance_gain(comparison_df, ROOT / "figures/14_baseline_vs_enhanced.png")

    save_pickle(basic_scaler, ROOT / "models/basic_descriptor_scaler.pkl")
    save_pickle(rich_processor, ROOT / "models/rich_descriptor_processor.pkl")
    save_pickle(rf_model, ROOT / "models/rf_basic_desc.pkl")
    save_pickle(svr_model, ROOT / "models/svr_basic_desc.pkl")
    xgb_baseline_model.save_model(str(ROOT / "models/xgb_baseline_desc_fp.json"))
    xgb_rich_model.save_model(str(ROOT / "models/xgb_rich_desc_fp_maccs.json"))
    catboost_rich_model.save_model(str(ROOT / "models/catboost_rich_desc_fp_maccs.cbm"))
    save_torch_model(
        mlp_rich_model,
        ROOT / "models/pytorch_mlp_rich_desc_fp_maccs.pt",
        {"device": mlp_device, "input_dim": int(X_rich_train.shape[1])},
    )

    best_model_name = str(enhanced_best["model"])
    run_summary = {
        "best_model": best_model_name,
        "mlp_device": mlp_device,
        "split_strategy": "stratified_random_split",
        "dataset_summary": summary,
        "ensemble_weights": ensemble_weights,
        "test_metrics": {row["model"]: {k: row[k] for k in ["r2", "rmse", "mae"]} for row in metrics_rows if row["split"] == "test"},
        "improvement_vs_baseline": {
            "baseline_best_model": baseline_best["model"],
            "baseline_best_rmse": float(baseline_best["rmse"]),
            "enhanced_best_model": best_model_name,
            "enhanced_best_rmse": float(enhanced_best["rmse"]),
            "rmse_gain": float(baseline_best["rmse"] - enhanced_best["rmse"]),
            "r2_gain": float(enhanced_best["r2"] - baseline_best["r2"]),
        },
    }
    save_json(run_summary, ROOT / "results/run_summary.json")
    write_results_markdown(ROOT / "results/final_results_v2.md", summary, metrics_df, comparison_df, ensemble_weights, best_model_name)
    write_matlab_script(ROOT / "results/final_results_v2.m")

    print("[5/7] Saved figures, metrics, predictions and trained models.")
    print("[6/7] Baseline vs enhanced:")
    print(comparison_df.to_string(index=False))
    print("[7/7] Test-set summary:")
    print(metrics_df[metrics_df["split"] == "test"].sort_values("rmse").to_string(index=False))
    print(f"Best enhanced model: {best_model_name}")
    print("Done.")


if __name__ == "__main__":
    main()
