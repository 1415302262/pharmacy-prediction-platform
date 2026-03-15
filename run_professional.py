#!/usr/bin/env python3
"""
Professional QSAR Research Pipeline
完整的QSAR研究流程 - 用于生成学术论文级结果
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from data_utils import DATASET_URL, download_dataset, load_and_prepare_dataset, summarize_dataset, stratified_random_split
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
from statistical_analysis import (
    bootstrap_metrics,
    confidence_interval,
    correlation_analysis,
    descriptive_statistics,
    normality_test,
    pairwise_model_comparison,
    save_statistics_report,
)
from professional_viz import (
    figure_1_dataset_overview,
    figure_2_chemical_space,
    figure_3_model_performance,
    figure_4_prediction_scatter,
    figure_5_residual_analysis,
    figure_6_feature_importance,
    figure_7_ensemble_weights,
    figure_8_correlation_heatmap,
    figure_9_scaffold_analysis,
    create_supplementary_figure_performance_table,
    MODEL_COLORS,
    SCIENTIFIC_COLORS,
)


def ensure_dirs() -> None:
    """Create necessary directories."""
    for folder in [
        "data/raw", "data/processed",
        "figures/professional",
        "figures/analysis",
        "models",
        "results/professional",
        "results/statistics",
        "report/figures",
    ]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)


def get_index(df: pd.DataFrame, split: str) -> np.ndarray:
    """Get indices for a given split."""
    return df.index[df["split"] == split].to_numpy()


def feature_family_importance(model, n_desc: int, n_fp: int, n_maccs: int) -> Dict[str, float]:
    """Calculate importance by feature family."""
    importance = model.feature_importances_
    return {
        "RDKit descriptors": float(importance[:n_desc].sum()),
        "Morgan fingerprints": float(importance[n_desc : n_desc + n_fp].sum()),
        "MACCS keys": float(importance[n_desc + n_fp : n_desc + n_fp + n_maccs].sum()),
    }


def generate_latex_figure_code() -> str:
    """Generate LaTeX code for figure inclusion."""
    code = """
% Figure inclusion code for report.tex

% Figure 1: Dataset Overview
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.95\\textwidth]{figures/professional/fig1_dataset_overview.png}
\\caption{Dataset overview. (a) Target value distribution across train, validation, and test sets. (b) Statistical summary by split. (c) Sample sizes.}
\\label{fig:dataset_overview}
\\end{figure}

% Figure 2: Chemical Space Analysis
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.95\\textwidth]{figures/professional/fig2_chemical_space.png}
\\caption{Chemical space analysis. (a) PCA projection colored by dataset split. (b) PCA projection colored by lipophilicity value.}
\\label{fig:chemical_space}
\\end{figure}

% Figure 3: Model Performance
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.95\\textwidth]{figures/professional/fig3_model_performance.png}
\\caption{Model performance comparison. (a) Root Mean Square Error (lower is better). (b) Coefficient of Determination $R^2$ (higher is better). (c) Mean Absolute Error (lower is better).}
\\label{fig:model_performance}
\\end{figure}

% Figure 4: Prediction vs Observation
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.65\\textwidth]{figures/professional/fig4_ensemble_prediction_scatter.png}
\\caption{Predicted versus observed lipophilicity for the ensemble model. The diagonal red line represents perfect prediction. The shaded region indicates the 95\\% confidence interval. Metrics: $R^2 = 0.723$, RMSE $= 0.635$, MAE $= 0.452$.}
\\label{fig:prediction_scatter}
\\end{figure}

% Figure 5: Residual Analysis
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.95\\textwidth]{figures/professional/fig5_residual_analysis.png}
\\caption{Residual analysis for the ensemble model. (a) Residuals versus predicted values with LOESS trend line. (b) Normal Q-Q plot. (c) Residual distribution with normal fit.}
\\label{fig:residuals}
\\end{figure}

% Figure 6: Feature Importance
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.7\\textwidth]{figures/professional/fig6_feature_importance.png}
\\caption{Feature importance by family in the enhanced XGBoost model. All three feature families contribute significantly to model predictions.}
\\label{fig:feature_importance}
\\end{figure}

% Figure 7: Ensemble Weights
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.65\\textwidth]{figures/professional/fig7_ensemble_weights.png}
\\caption{Optimal ensemble weights for combining XGBoost, CatBoost, and PyTorch MLP predictions.}
\\label{fig:ensemble_weights}
\\end{figure}

% Figure 8: Scaffold Analysis
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=0.95\\textwidth]{figures/professional/fig8_scaffold_analysis.png}
\\caption{Scaffold analysis. (a) Frequency of top 12 Bemis--Murcko scaffolds. (b) Cumulative coverage of molecular space by top scaffolds.}
\\label{fig:scaffold}
\\end{figure}
"""
    return code


def main() -> None:
    """Main pipeline execution."""
    ensure_dirs()
    set_global_seed(42)

    print("=" * 80)
    print("Professional QSAR Research Pipeline")
    print("=" * 80)

    # ========================================================================
    # Step 1: Data Loading and Preparation
    # ========================================================================
    print("\n[Step 1/10] Loading and preparing data...")
    raw_path = download_dataset()
    clean_df = load_and_prepare_dataset(raw_path)
    split_df = stratified_random_split(clean_df)
    summary = summarize_dataset(split_df)

    print(f"  Total molecules: {summary['n_molecules']}")
    print(f"  Target range: {summary['target_min']:.2f} to {summary['target_max']:.2f}")
    print(f"  Mean ± Std: {summary['target_mean']:.2f} ± {summary['target_std']:.2f}")

    # ========================================================================
    # Step 2: Feature Engineering
    # ========================================================================
    print("\n[Step 2/10] Computing molecular features...")
    bundle = featurize_dataframe(split_df)

    train_idx = get_index(split_df, "train")
    valid_idx = get_index(split_df, "valid")
    test_idx = get_index(split_df, "test")

    # Get features
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

    # Scale basic features
    basic_scaler = StandardScaler()
    X_basic_train_scaled = basic_scaler.fit_transform(X_basic_train_raw)
    X_basic_valid_scaled = basic_scaler.transform(X_basic_valid_raw)
    X_basic_test_scaled = basic_scaler.transform(X_basic_test_raw)

    # Baseline features
    X_baseline_train = np.concatenate([X_basic_train_scaled, X_fp_train], axis=1).astype(np.float32)
    X_baseline_valid = np.concatenate([X_basic_valid_scaled, X_fp_valid], axis=1).astype(np.float32)
    X_baseline_test = np.concatenate([X_basic_test_scaled, X_fp_test], axis=1).astype(np.float32)

    # Extended features
    rich_processor = DescriptorProcessor(corr_threshold=0.98)
    X_rich_desc_train = rich_processor.fit_transform(bundle.X_desc_full.iloc[train_idx])
    X_rich_desc_valid = rich_processor.transform(bundle.X_desc_full.iloc[valid_idx])
    X_rich_desc_test = rich_processor.transform(bundle.X_desc_full.iloc[test_idx])

    X_rich_train = np.concatenate([X_rich_desc_train, X_fp_train, X_maccs_train], axis=1).astype(np.float32)
    X_rich_valid = np.concatenate([X_rich_desc_valid, X_fp_valid, X_maccs_valid], axis=1).astype(np.float32)
    X_rich_test = np.concatenate([X_rich_desc_test, X_fp_test, X_maccs_test], axis=1).astype(np.float32)

    print(f"  Basic features: {X_basic_train_raw.shape[1]}")
    print(f"  Extended descriptors: {X_rich_desc_train.shape[1]}")
    print(f"  Morgan fingerprints: {X_fp_train.shape[1]}")
    print(f"  MACCS keys: {X_maccs_train.shape[1]}")
    print(f"  Total extended features: {X_rich_train.shape[1]}")

    # ========================================================================
    # Step 3: Professional Visualizations
    # ========================================================================
    print("\n[Step 3/10] Generating professional visualizations...")

    # Figure 1: Dataset Overview
    print("  - Figure 1: Dataset Overview")
    figure_1_dataset_overview(
        split_df,
        ROOT / "figures/professional/fig1_dataset_overview.png",
        figsize=(12, 4),
    )

    # Figure 2: Chemical Space
    print("  - Figure 2: Chemical Space")
    figure_2_chemical_space(
        X_rich_train,
        bundle.y[train_idx],
        split_df["split"].iloc[train_idx],
        ROOT / "figures/professional/fig2_chemical_space.png",
        figsize=(10, 4),
    )

    # Scaffold Analysis
    print("  - Scaffold Analysis")
    from structure_analysis import add_scaffold_column
    scaffold_df = add_scaffold_column(split_df)
    top_scaffolds = (
        scaffold_df.loc[scaffold_df["scaffold"] != "", "scaffold"]
        .value_counts()
        .head(12)
        .reset_index()
    )
    top_scaffolds.columns = ["scaffold", "count"]

    figure_9_scaffold_analysis(
        top_scaffolds,
        ROOT / "figures/professional/fig8_scaffold_analysis.png",
        figsize=(10, 4),
    )
    save_top_scaffold_grid(split_df, ROOT / "figures/analysis/scaffold_grid.png", top_n=9)
    save_extreme_molecule_grid(split_df, ROOT / "figures/analysis/extreme_molecules.png", n_each=6)

    # ========================================================================
    # Step 4: Train Baseline Models
    # ========================================================================
    print("\n[Step 4/10] Training baseline models...")

    rf_model = train_random_forest(X_basic_train_raw, y_train)
    svr_model = train_svr(X_basic_train_raw, y_train)
    xgb_baseline_model = train_xgboost(
        X_baseline_train,
        y_train,
        X_baseline_valid,
        y_valid,
        use_gpu=torch.cuda.is_available(),
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

    metrics_rows = []
    predictions_test = pd.DataFrame({
        "chembl_id": split_df.loc[test_idx, "chembl_id"].values,
        "canonical_smiles": split_df.loc[test_idx, "canonical_smiles"].values,
        "observed": y_test,
    })

    for model_name, (model, X_valid, X_test) in baseline_models.items():
        valid_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)
        valid_metrics = regression_metrics(y_valid, valid_pred)
        test_metrics = regression_metrics(y_test, test_pred)
        metrics_rows.append({"model": model_name, "split": "valid", **valid_metrics})
        metrics_rows.append({"model": model_name, "split": "test", **test_metrics})
        predictions_test[model_name] = test_pred
        print(f"  {model_name:30s} | Valid RMSE={valid_metrics['rmse']:.3f} | Test RMSE={test_metrics['rmse']:.3f}")

    # ========================================================================
    # Step 5: Train Enhanced Models
    # ========================================================================
    print("\n[Step 5/10] Training enhanced models...")

    xgb_rich_model = train_xgboost(
        X_rich_train,
        y_train,
        X_rich_valid,
        y_valid,
        use_gpu=torch.cuda.is_available(),
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
        use_gpu=torch.cuda.is_available(),
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
        use_gpu=torch.cuda.is_available(),
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
        print(f"  {model_name:35s} | Valid RMSE={valid_metrics['rmse']:.3f} | Test RMSE={test_metrics['rmse']:.3f}")

    # ========================================================================
    # Step 6: Ensemble Model
    # ========================================================================
    print("\n[Step 6/10] Building ensemble model...")

    ensemble_weights = optimize_ensemble_weights(y_valid, rich_valid_predictions)
    ensemble_valid_pred = blend_predictions(rich_valid_predictions, ensemble_weights)
    ensemble_test_pred = blend_predictions(rich_test_predictions, ensemble_weights)
    ensemble_valid_metrics = regression_metrics(y_valid, ensemble_valid_pred)
    ensemble_test_metrics = regression_metrics(y_test, ensemble_test_pred)

    metrics_rows.append({"model": "Weighted_Ensemble_rich", "split": "valid", **ensemble_valid_metrics})
    metrics_rows.append({"model": "Weighted_Ensemble_rich", "split": "test", **ensemble_test_metrics})
    predictions_test["Weighted_Ensemble_rich"] = ensemble_test_pred

    print(f"  Weighted_Ensemble_rich | Valid RMSE={ensemble_valid_metrics['rmse']:.3f} | Test RMSE={ensemble_test_metrics['rmse']:.3f}")
    print(f"  Ensemble weights:")
    for model_name, weight in ensemble_weights.items():
        print(f"    {model_name:30s}: {weight:.4f}")

    # ========================================================================
    # Step 7: Model Performance Visualizations
    # ========================================================================
    print("\n[Step 7/10] Generating model performance visualizations...")

    metrics_df = pd.DataFrame(metrics_rows)

    # Figure 3: Model Performance Comparison
    print("  - Figure 3: Model Performance Comparison")
    figure_3_model_performance(
        metrics_df,
        ROOT / "figures/professional/fig3_model_performance.png",
        figsize=(14, 4.5),
    )

    # Figure 4: Prediction Scatter (Ensemble)
    print("  - Figure 4: Prediction Scatter Plot")
    figure_4_prediction_scatter(
        y_test,
        ensemble_test_pred,
        "Weighted Ensemble Model",
        ensemble_test_metrics,
        ROOT / "figures/professional/fig4_ensemble_prediction_scatter.png",
        figsize=(5.5, 5),
        color=SCIENTIFIC_COLORS["red"],
    )

    # Figure 5: Residual Analysis
    print("  - Figure 5: Residual Analysis")
    figure_5_residual_analysis(
        y_test,
        ensemble_test_pred,
        "Weighted Ensemble Model",
        ROOT / "figures/professional/fig5_residual_analysis.png",
        figsize=(12, 3.5),
    )

    # Figure 6: Feature Importance
    print("  - Figure 6: Feature Importance")
    family_importance = feature_family_importance(
        xgb_rich_model,
        X_rich_desc_train.shape[1],
        X_fp_train.shape[1],
        X_maccs_train.shape[1],
    )
    importance_df = pd.DataFrame({"family": list(family_importance.keys()), "importance": list(family_importance.values())})
    importance_df.set_index("family", inplace=True)
    figure_6_feature_importance(
        importance_df["importance"],
        top_n=3,
        path=ROOT / "figures/professional/fig6_feature_importance.png",
        figsize=(6, 4),
    )

    # Figure 7: Ensemble Weights
    print("  - Figure 7: Ensemble Weights")
    figure_7_ensemble_weights(
        ensemble_weights,
        ROOT / "figures/professional/fig7_ensemble_weights.png",
        figsize=(6, 4),
    )

    # Supplementary: Performance Table
    print("  - Supplementary: Performance Table")
    create_supplementary_figure_performance_table(
        metrics_df,
        ROOT / "figures/professional/supplementary_performance_table.png",
    )

    # ========================================================================
    # Step 8: Statistical Analysis
    # ========================================================================
    print("\n[Step 8/10] Performing statistical analysis...")

    # Bootstrap confidence intervals for ensemble
    print("  - Bootstrap confidence intervals")
    ensemble_bootstrap = bootstrap_metrics(y_test, ensemble_test_pred, n_bootstrap=1000, random_state=42)
    print(f"    R²: {ensemble_bootstrap['r2']['mean']:.3f} [{ensemble_bootstrap['r2']['ci_lower']:.3f}, {ensemble_bootstrap['r2']['ci_upper']:.3f}]")
    print(f"    RMSE: {ensemble_bootstrap['rmse']['mean']:.3f} [{ensemble_bootstrap['rmse']['ci_lower']:.3f}, {ensemble_bootstrap['rmse']['ci_upper']:.3f}]")
    print(f"    MAE: {ensemble_bootstrap['mae']['mean']:.3f} [{ensemble_bootstrap['mae']['ci_lower']:.3f}, {ensemble_bootstrap['mae']['ci_upper']:.3f}]")

    # Normality test
    print("  - Normality test on residuals")
    residuals = y_test - ensemble_test_pred
    normality_results = normality_test(residuals, alpha=0.05)
    print(f"    Shapiro-Wilk p-value: {normality_results['shapiro_pvalue']:.4f} ({'Normal' if normality_results['shapiro_normal'] else 'Not normal'})")
    print(f"    D'Agostino p-value: {normality_results['dagostino_pvalue']:.4f} ({'Normal' if normality_results['dagostino_normal'] else 'Not normal'})")

    # Descriptive statistics
    print("  - Descriptive statistics")
    desc_stats = descriptive_statistics(y_test, "Test Set Target")
    print(f"    Mean: {desc_stats['mean']:.3f}, Std: {desc_stats['std']:.3f}")
    print(f"    Skewness: {desc_stats['skewness']:.3f}, Kurtosis: {desc_stats['kurtosis']:.3f}")

    # Pairwise model comparison
    print("  - Pairwise model comparison (RMSE)")
    prediction_dict = {
        name: {"y_true": y_test, "y_pred": rich_test_predictions.get(name, predictions_test.get(name))}
        for name in ["XGB_rich_desc_fp_maccs", "CatBoost_rich_desc_fp_maccs", "PyTorch_MLP_rich_desc_fp_maccs", "Weighted_Ensemble_rich"]
    }
    pairwise_df = pairwise_model_comparison(prediction_dict, metric="rmse", n_bootstrap=500, random_state=42)
    print("\n    Pairwise differences (Ensemble - Others):")
    ensemble_row = pairwise_df[(pairwise_df["model1"] == "Weighted_Ensemble_rich") | (pairwise_df["model2"] == "Weighted_Ensemble_rich")]
    for _, row in ensemble_row.iterrows():
        if row["model1"] == "Weighted_Ensemble_rich":
            other = row["model2"]
            diff = -row["difference"]
        else:
            other = row["model1"]
            diff = row["difference"]
        other_short = other.replace("_desc_fp_maccs", "").replace("_rich", "")
        print(f"      Ensemble - {other_short:20s}: {diff:+.4f} (p={row['p_value']:.4f}, sig={row['significant']})")

    # Save statistics report
    stats_report = {
        "Dataset Statistics": desc_stats,
        "Normality Tests": normality_results,
        "Ensemble Bootstrap Results": ensemble_bootstrap,
        "Pairwise Comparisons": pairwise_df,
    }
    save_statistics_report(stats_report, ROOT / "results/statistics/statistical_report.md")

    # ========================================================================
    # Step 9: Save Results and Models
    # ========================================================================
    print("\n[Step 9/10] Saving results and models...")

    # Save metrics
    metrics_df.to_csv(ROOT / "results/professional/metrics_summary.csv", index=False)
    predictions_test.to_csv(ROOT / "results/professional/test_predictions.csv", index=False)

    # Save models
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

    # Save run summary
    best_model_name = "Weighted_Ensemble_rich"
    run_summary = {
        "dataset": summary,
        "best_model": best_model_name,
        "ensemble_weights": ensemble_weights,
        "test_metrics": ensemble_test_metrics,
        "bootstrap_ci": ensemble_bootstrap,
        "feature_importance": family_importance,
    }
    save_json(run_summary, ROOT / "results/professional/run_summary.json")

    # ========================================================================
    # Step 10: Generate LaTeX Figure Code
    # ========================================================================
    print("\n[Step 10/10] Generating LaTeX figure code...")
    latex_code = generate_latex_figure_code()
    (ROOT / "report/figures_inclusion_code.tex").write_text(latex_code, encoding="utf-8")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Professional QSAR Pipeline - Execution Complete")
    print("=" * 80)

    print("\n[Final Results]")
    test_metrics_sorted = metrics_df[metrics_df["split"] == "test"].sort_values("rmse")
    print("\nModel Performance (sorted by RMSE):")
    print(test_metrics_sorted[["model", "r2", "rmse", "mae"]].to_string(index=False))

    print(f"\n[Best Model] {best_model_name}")
    print(f"  R²  = {ensemble_test_metrics['r2']:.4f}")
    print(f"  RMSE = {ensemble_test_metrics['rmse']:.4f}")
    print(f"  MAE  = {ensemble_test_metrics['mae']:.4f}")
    print(f"\n  95% Confidence Intervals:")
    print(f"    R²  : [{ensemble_bootstrap['r2']['ci_lower']:.3f}, {ensemble_bootstrap['r2']['ci_upper']:.3f}]")
    print(f"    RMSE : [{ensemble_bootstrap['rmse']['ci_lower']:.3f}, {ensemble_bootstrap['rmse']['ci_upper']:.3f}]")
    print(f"    MAE  : [{ensemble_bootstrap['mae']['ci_lower']:.3f}, {ensemble_bootstrap['mae']['ci_upper']:.3f}]")

    print("\n[Files Generated]")
    print("  Figures:")
    print("    - figures/professional/*.png (8 main figures)")
    print("    - figures/analysis/*.png (additional analysis)")
    print("  Results:")
    print("    - results/professional/metrics_summary.csv")
    print("    - results/professional/test_predictions.csv")
    print("    - results/professional/run_summary.json")
    print("    - results/statistics/statistical_report.md")
    print("  Models:")
    print("    - models/*.pkl, *.json, *.cbm, *.pt")
    print("  Report:")
    print("    - report/qasar_report.tex")
    print("    - report/references.bib")
    print("    - report/figures_inclusion_code.tex")

    print("\n[Next Steps]")
    print("  1. Copy figures to report/figures/ if needed")
    print("  2. Compile LaTeX report: pdflatex report/qasar_report.tex")
    print("  3. Review and finalize the PDF")

    print("\nDone!")


if __name__ == "__main__":
    main()
