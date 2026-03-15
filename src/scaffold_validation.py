from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from data_utils import scaffold_from_smiles
from featurization import DescriptorProcessor, FeatureBundle
from train_model import (
    blend_predictions,
    optimize_ensemble_weights,
    predict_mlp,
    regression_metrics,
    train_catboost,
    train_mlp,
    train_xgboost,
)


sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 130


def build_scaffold_series(df: pd.DataFrame) -> pd.Series:
    return df["canonical_smiles"].map(scaffold_from_smiles)


def grouped_scaffold_split(
    df: pd.DataFrame,
    random_state: int,
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    if not np.isclose(train_frac + valid_frac + test_frac, 1.0):
        raise ValueError("split fractions must sum to 1.0")

    work = df.copy().reset_index(drop=True)
    work["scaffold"] = build_scaffold_series(work)
    groups = work["scaffold"].fillna("")
    all_idx = np.arange(len(work))

    train_valid_frac = train_frac + valid_frac
    test_split = GroupShuffleSplit(n_splits=1, train_size=train_valid_frac, test_size=test_frac, random_state=random_state)
    train_valid_idx, test_idx = next(test_split.split(all_idx, groups=groups))

    valid_ratio_in_train_valid = valid_frac / (train_frac + valid_frac)
    train_valid_groups = groups.iloc[train_valid_idx]
    valid_split = GroupShuffleSplit(
        n_splits=1,
        train_size=(1.0 - valid_ratio_in_train_valid),
        test_size=valid_ratio_in_train_valid,
        random_state=random_state + 1000,
    )
    train_rel_idx, valid_rel_idx = next(valid_split.split(train_valid_idx, groups=train_valid_groups))
    train_idx = train_valid_idx[train_rel_idx]
    valid_idx = train_valid_idx[valid_rel_idx]

    split = np.array(["unassigned"] * len(work), dtype=object)
    split[train_idx] = "train"
    split[valid_idx] = "valid"
    split[test_idx] = "test"
    work["split"] = split
    return work


def prepare_feature_sets(bundle: FeatureBundle, split_df: pd.DataFrame) -> dict[str, object]:
    train_idx = split_df.index[split_df["split"] == "train"].to_numpy()
    valid_idx = split_df.index[split_df["split"] == "valid"].to_numpy()
    test_idx = split_df.index[split_df["split"] == "test"].to_numpy()

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

    feature_names = (
        rich_processor.get_feature_names_out()
        + [f"Morgan_{idx:04d}" for idx in range(X_fp_train.shape[1])]
        + [f"MACCS_{idx:03d}" for idx in range(X_maccs_train.shape[1])]
    )

    return {
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "X_baseline_train": X_baseline_train,
        "X_baseline_valid": X_baseline_valid,
        "X_baseline_test": X_baseline_test,
        "X_rich_train": X_rich_train,
        "X_rich_valid": X_rich_valid,
        "X_rich_test": X_rich_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "feature_names": feature_names,
        "n_desc": X_rich_desc_train.shape[1],
        "n_fp": X_fp_train.shape[1],
        "n_maccs": X_maccs_train.shape[1],
        "rich_processor": rich_processor,
    }


def save_repeat0_distribution(split_df: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.histplot(split_df, x="target", hue="split", bins=30, kde=True, palette="Set2", ax=axes[0])
    axes[0].set_title("Scaffold-aware split distribution")
    axes[0].set_xlabel("Experimental lipophilicity")

    counts = split_df["split"].value_counts().reindex(["train", "valid", "test"]).reset_index()
    counts.columns = ["split", "count"]
    sns.barplot(data=counts, x="split", y="count", hue="split", dodge=False, legend=False, palette="Set2", ax=axes[1])
    axes[1].set_title("Scaffold-aware split sizes")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Molecule count")

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_repeat_summary(summary_df: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.barplot(data=summary_df, x="model", y="rmse_mean", hue="model", dodge=False, legend=False, palette="Blues_d", ax=axes[0])
    axes[0].errorbar(np.arange(len(summary_df)), summary_df["rmse_mean"], yerr=summary_df["rmse_std"], fmt="none", c="black", capsize=4)
    axes[0].set_title("Repeated scaffold split RMSE")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean RMSE ± SD")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=summary_df, x="model", y="r2_mean", hue="model", dodge=False, legend=False, palette="Greens_d", ax=axes[1])
    axes[1].errorbar(np.arange(len(summary_df)), summary_df["r2_mean"], yerr=summary_df["r2_std"], fmt="none", c="black", capsize=4)
    axes[1].set_title("Repeated scaffold split R²")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean R² ± SD")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_repeat_boxplot(metrics_df: pd.DataFrame, path: str | Path) -> None:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.boxplot(data=test_df, x="model", y="rmse", ax=axes[0], palette="Blues")
    axes[0].set_title("RMSE across scaffold repeats")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)

    sns.boxplot(data=test_df, x="model", y="r2", ax=axes[1], palette="Greens")
    axes[1].set_title("R² across scaffold repeats")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_scaffold_repeats(
    df: pd.DataFrame,
    bundle: FeatureBundle,
    repeat_seeds: list[int],
    use_gpu: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metrics_rows: list[dict] = []
    first_repeat_artifacts: dict[str, object] = {}

    for repeat_id, seed in enumerate(repeat_seeds):
        split_df = grouped_scaffold_split(df, random_state=seed)
        features = prepare_feature_sets(bundle, split_df)

        xgb_baseline = train_xgboost(
            features["X_baseline_train"],
            features["y_train"],
            features["X_baseline_valid"],
            features["y_valid"],
            use_gpu=use_gpu,
            n_estimators=700,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.60,
            reg_lambda=1.0,
        )
        xgb_rich = train_xgboost(
            features["X_rich_train"],
            features["y_train"],
            features["X_rich_valid"],
            features["y_valid"],
            use_gpu=use_gpu,
            n_estimators=1200,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.70,
            reg_lambda=1.2,
            min_child_weight=2.0,
        )
        catboost_rich = train_catboost(
            features["X_rich_train"],
            features["y_train"],
            features["X_rich_valid"],
            features["y_valid"],
            use_gpu=use_gpu,
            depth=8,
            learning_rate=0.03,
            iterations=2000,
            l2_leaf_reg=5.0,
        )
        mlp_rich, mlp_history, mlp_device = train_mlp(
            X_train=features["X_rich_train"],
            y_train=features["y_train"],
            X_valid=features["X_rich_valid"],
            y_valid=features["y_valid"],
            use_gpu=use_gpu,
            epochs=80,
            batch_size=128,
            lr=8e-4,
            patience=14,
        )

        valid_predictions = {
            "XGB_rich_scaffold": xgb_rich.predict(features["X_rich_valid"]),
            "CatBoost_rich_scaffold": catboost_rich.predict(features["X_rich_valid"]),
            "PyTorch_MLP_rich_scaffold": predict_mlp(mlp_rich, features["X_rich_valid"], device=mlp_rich.network[0].weight.device),
        }
        test_predictions = {
            "XGB_rich_scaffold": xgb_rich.predict(features["X_rich_test"]),
            "CatBoost_rich_scaffold": catboost_rich.predict(features["X_rich_test"]),
            "PyTorch_MLP_rich_scaffold": predict_mlp(mlp_rich, features["X_rich_test"], device=mlp_rich.network[0].weight.device),
        }
        ensemble_weights = optimize_ensemble_weights(features["y_valid"], valid_predictions)
        valid_predictions["Weighted_Ensemble_scaffold"] = blend_predictions(valid_predictions, ensemble_weights)
        test_predictions["Weighted_Ensemble_scaffold"] = blend_predictions(test_predictions, ensemble_weights)

        baseline_valid = xgb_baseline.predict(features["X_baseline_valid"])
        baseline_test = xgb_baseline.predict(features["X_baseline_test"])
        valid_predictions["XGB_baseline_scaffold"] = baseline_valid
        test_predictions["XGB_baseline_scaffold"] = baseline_test

        n_train = int(len(features["train_idx"]))
        n_valid = int(len(features["valid_idx"]))
        n_test = int(len(features["test_idx"]))
        scaffold_counts = split_df.groupby("split")["scaffold"].nunique().to_dict()

        for model_name, valid_pred in valid_predictions.items():
            valid_metrics = regression_metrics(features["y_valid"], valid_pred)
            test_metrics = regression_metrics(features["y_test"], test_predictions[model_name])
            metrics_rows.append(
                {
                    "repeat": repeat_id,
                    "seed": seed,
                    "model": model_name,
                    "split": "valid",
                    "n_train": n_train,
                    "n_valid": n_valid,
                    "n_test": n_test,
                    "train_scaffolds": int(scaffold_counts.get("train", 0)),
                    "valid_scaffolds": int(scaffold_counts.get("valid", 0)),
                    "test_scaffolds": int(scaffold_counts.get("test", 0)),
                    **valid_metrics,
                }
            )
            metrics_rows.append(
                {
                    "repeat": repeat_id,
                    "seed": seed,
                    "model": model_name,
                    "split": "test",
                    "n_train": n_train,
                    "n_valid": n_valid,
                    "n_test": n_test,
                    "train_scaffolds": int(scaffold_counts.get("train", 0)),
                    "valid_scaffolds": int(scaffold_counts.get("valid", 0)),
                    "test_scaffolds": int(scaffold_counts.get("test", 0)),
                    **test_metrics,
                }
            )

        if repeat_id == 0:
            first_repeat_artifacts = {
                "split_df": split_df,
                "xgb_rich_model": xgb_rich,
                "catboost_rich_model": catboost_rich,
                "xgb_test_features": features["X_rich_test"],
                "xgb_test_predictions": test_predictions["XGB_rich_scaffold"],
                "catboost_test_predictions": test_predictions["CatBoost_rich_scaffold"],
                "xgb_test_truth": features["y_test"],
                "feature_names": features["feature_names"],
                "test_index": features["test_idx"],
                "ensemble_weights": ensemble_weights,
                "mlp_device": mlp_device,
                "mlp_history": mlp_history,
            }

    metrics_df = pd.DataFrame(metrics_rows)
    summary_df = (
        metrics_df[metrics_df["split"] == "test"]
        .groupby("model")
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
        )
        .reset_index()
        .sort_values("rmse_mean")
    )
    return metrics_df, summary_df, first_repeat_artifacts


def save_scaffold_summary_markdown(summary_df: pd.DataFrame, path: str | Path) -> None:
    lines = [
        "# Scaffold Validation Summary",
        "",
        "说明：该结果采用 scaffold-aware 重复划分，数值通常低于随机划分，但科研可信度更高。",
        "",
        "| Model | R2 mean | R2 std | RMSE mean | RMSE std | MAE mean | MAE std |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.r2_mean:.4f} | {row.r2_std:.4f} | {row.rmse_mean:.4f} | {row.rmse_std:.4f} | {row.mae_mean:.4f} | {row.mae_std:.4f} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_scaffold_matlab_script(path: str | Path) -> None:
    script = """
+metrics = readtable('results/scaffold_repeat_summary.csv');
+disp(metrics)
+
+figure('Position', [100, 100, 1100, 420]);
+subplot(1,2,1)
+bar(categorical(metrics.model), metrics.rmse_mean)
+title('Scaffold Repeat RMSE Mean')
+ylabel('RMSE')
+
+subplot(1,2,2)
+bar(categorical(metrics.model), metrics.r2_mean)
+title('Scaffold Repeat R2 Mean')
+ylabel('R2')
+
+sgtitle('Rigorous Scaffold Validation Summary')
+""".strip().replace("+", "")
    Path(path).write_text(script + "\n", encoding="utf-8")


def save_shap_artifacts(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    case_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model.__class__.__name__.lower()
    if "catboost" in model_name:
        from catboost import Pool

        pool = Pool(X_test, label=y_test)
        shap_matrix = model.get_feature_importance(pool, type="ShapValues")
        shap_values = shap_matrix[:, :-1]
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:20]
    top_features = pd.DataFrame(
        {
            "feature": [feature_names[idx] for idx in top_idx],
            "mean_abs_shap": mean_abs[top_idx],
        }
    )
    top_features.to_csv(output_dir / "shap_top20_features.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    sns.barplot(data=top_features, x="mean_abs_shap", y="feature", hue="feature", dodge=False, legend=False, palette="magma", ax=ax)
    ax.set_title("Top 20 SHAP features (scaffold split repeat 0)")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / ".." / "figures" / "19_shap_top20_bar.png", bbox_inches="tight")
    plt.close(fig)

    shap.summary_plot(
        shap_values[:, top_idx],
        features=X_test[:, top_idx],
        feature_names=[feature_names[idx] for idx in top_idx],
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(output_dir / ".." / "figures" / "20_shap_beeswarm_top20.png", bbox_inches="tight")
    plt.close()

    case_definitions = {
        "HighPrediction": int(np.argmax(y_pred)),
        "LowPrediction": int(np.argmin(y_pred)),
        "LargeError": int(np.argmax(np.abs(y_test - y_pred))),
    }

    case_lines = ["# SHAP Case Studies", ""]
    mols = []
    legends = []
    for case_name, local_idx in case_definitions.items():
        shap_row = shap_values[local_idx]
        top_local_idx = np.argsort(np.abs(shap_row))[::-1][:8]
        local_df = pd.DataFrame(
            {
                "feature": [feature_names[idx] for idx in top_local_idx],
                "feature_value": X_test[local_idx, top_local_idx],
                "shap_value": shap_row[top_local_idx],
            }
        )
        local_df.to_csv(output_dir / f"shap_case_{case_name.lower()}.csv", index=False)

        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        plot_df = local_df.sort_values("shap_value")
        colors = ["#1f77b4" if value < 0 else "#d62728" for value in plot_df["shap_value"]]
        ax.barh(plot_df["feature"], plot_df["shap_value"], color=colors)
        ax.axvline(0.0, color="black", linewidth=1.2)
        ax.set_title(f"Local SHAP explanation: {case_name}")
        ax.set_xlabel("SHAP contribution")
        fig.tight_layout()
        fig.savefig(output_dir / ".." / "figures" / f"21_shap_local_{case_name.lower()}.png", bbox_inches="tight")
        plt.close(fig)

        record = case_df.iloc[local_idx]
        mols.append(Chem.MolFromSmiles(record["canonical_smiles"]))
        legends.append(f"{case_name}\nobs={record['observed']:.2f}\npred={record['predicted']:.2f}")

        case_lines.append(f"## {case_name}")
        case_lines.append(f"- Observed: {record['observed']:.4f}")
        case_lines.append(f"- Predicted: {record['predicted']:.4f}")
        case_lines.append("")
        case_lines.append("| Feature | Value | SHAP |")
        case_lines.append("|---|---:|---:|")
        for row in local_df.itertuples(index=False):
            case_lines.append(f"| {row.feature} | {row.feature_value:.4f} | {row.shap_value:.4f} |")
        case_lines.append("")

    grid = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(280, 220), legends=legends, useSVG=False)
    grid.save(str(output_dir / ".." / "figures" / "22_shap_case_molecules.png"))
    (output_dir / "shap_case_studies.md").write_text("\n".join(case_lines) + "\n", encoding="utf-8")

    return {
        "top_features": top_features,
        "case_definitions": case_definitions,
    }
