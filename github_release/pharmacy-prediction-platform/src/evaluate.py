from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 130


def save_metrics_table(metrics_rows: list[dict], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(metrics_rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def plot_dataset_overview(df: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    sns.histplot(df, x="target", hue="split", bins=30, kde=True, ax=axes[0], palette="Set2")
    axes[0].set_title("Lipophilicity distribution by stratified random split")
    axes[0].set_xlabel("Experimental lipophilicity (exp)")

    split_counts = df["split"].value_counts().reindex(["train", "valid", "test"]).reset_index()
    split_counts.columns = ["split", "count"]
    sns.barplot(data=split_counts, x="split", y="count", hue="split", dodge=False, legend=False, ax=axes[1], palette="Set2")
    axes[1].set_title("Split sizes")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Molecule count")

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics_df: pd.DataFrame, path: str | Path) -> None:
    plot_df = metrics_df[metrics_df["split"] == "test"].sort_values("rmse").copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    sns.barplot(data=plot_df, x="model", y="rmse", hue="model", dodge=False, legend=False, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Test RMSE comparison")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(data=plot_df, x="model", y="r2", hue="model", dodge=False, legend=False, ax=axes[1], palette="Greens_d")
    axes[1].set_title("Test R² comparison")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R²")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    metrics: Dict[str, float],
    path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    sns.scatterplot(x=y_true, y=y_pred, s=42, alpha=0.75, ax=ax)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="crimson", linewidth=1.8)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}\nR²={metrics['r2']:.3f} | RMSE={metrics['rmse']:.3f} | MAE={metrics['mae']:.3f}")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(history: Dict[str, list], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    epochs = np.arange(1, len(history["train_rmse"]) + 1)
    ax.plot(epochs, history["train_rmse"], label="Train RMSE", linewidth=2)
    ax.plot(epochs, history["valid_rmse"], label="Valid RMSE", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("PyTorch MLP training curve")
    ax.legend(frameon=True)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_descriptor_importance(model, feature_names: list[str], path: str | Path, top_n: int = 10) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n).reset_index()
    importance.columns = ["feature", "importance"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    sns.barplot(data=importance, x="importance", y="feature", hue="feature", dodge=False, legend=False, palette="magma", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title("Random forest basic descriptor importance")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_family_importance(family_importance: Dict[str, float], path: str | Path) -> None:
    plot_df = pd.DataFrame({"family": list(family_importance.keys()), "importance": list(family_importance.values())})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.barplot(data=plot_df, x="family", y="importance", hue="family", dodge=False, legend=False, palette="viridis", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Aggregated importance")
    ax.set_title("Feature-family contribution in enhanced XGBoost")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_weights(weight_dict: Dict[str, float], path: str | Path) -> None:
    plot_df = pd.DataFrame({"model": list(weight_dict.keys()), "weight": list(weight_dict.values())}).sort_values("weight", ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.barplot(data=plot_df, x="model", y="weight", hue="model", dodge=False, legend=False, palette="rocket", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Weight")
    ax.set_title("Validation-optimized ensemble weights")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_performance_gain(comparison_df: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    sns.barplot(data=comparison_df, x="version", y="rmse", hue="version", dodge=False, legend=False, palette="crest", ax=axes[0])
    axes[0].set_title("Best-model RMSE: baseline vs enhanced")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("RMSE")

    sns.barplot(data=comparison_df, x="version", y="r2", hue="version", dodge=False, legend=False, palette="flare", ax=axes[1])
    axes[1].set_title("Best-model R²: baseline vs enhanced")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("R²")

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: str | Path) -> None:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.scatterplot(x=y_pred, y=residuals, s=42, alpha=0.75, ax=ax)
    ax.axhline(0.0, linestyle="--", color="crimson", linewidth=1.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Observed - Predicted)")
    ax.set_title(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
