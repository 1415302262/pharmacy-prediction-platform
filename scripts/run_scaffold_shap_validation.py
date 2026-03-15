from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from data_utils import download_dataset, load_and_prepare_dataset
from featurization import featurize_dataframe
from scaffold_validation import (
    grouped_scaffold_split,
    plot_repeat_boxplot,
    plot_repeat_summary,
    run_scaffold_repeats,
    save_repeat0_distribution,
    save_scaffold_matlab_script,
    save_scaffold_summary_markdown,
    save_shap_artifacts,
)


def main() -> None:
    print("=" * 78)
    print("Rigorous Validation: Repeated Scaffold Split + SHAP Interpretation")
    print("=" * 78)

    download_dataset()
    df = load_and_prepare_dataset()
    bundle = featurize_dataframe(df)

    repeat_seeds = [11, 23, 37, 51, 65]
    print(f"Running repeated scaffold validation on {len(repeat_seeds)} scaffold-aware splits...")
    metrics_df, summary_df, artifacts = run_scaffold_repeats(df, bundle, repeat_seeds=repeat_seeds, use_gpu=True)

    metrics_path = ROOT / "results/scaffold_repeat_metrics.csv"
    summary_path = ROOT / "results/scaffold_repeat_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    repeat0_split = artifacts["split_df"]
    repeat0_split.to_csv(ROOT / "results/scaffold_repeat0_assignments.csv", index=False)
    save_repeat0_distribution(repeat0_split, ROOT / "figures/15_scaffold_repeat0_distribution.png")
    plot_repeat_summary(summary_df, ROOT / "figures/16_scaffold_repeat_summary.png")
    plot_repeat_boxplot(metrics_df, ROOT / "figures/17_scaffold_repeat_boxplot.png")

    test_idx = artifacts["test_index"]
    case_df = repeat0_split.iloc[test_idx][["chembl_id", "canonical_smiles", "target"]].copy()
    case_df = case_df.rename(columns={"target": "observed"}).reset_index(drop=True)
    case_df["predicted"] = artifacts["catboost_test_predictions"]
    shap_info = save_shap_artifacts(
        model=artifacts["catboost_rich_model"],
        X_test=artifacts["xgb_test_features"],
        y_test=artifacts["xgb_test_truth"],
        y_pred=artifacts["catboost_test_predictions"],
        feature_names=artifacts["feature_names"],
        case_df=case_df,
        output_dir=ROOT / "results",
    )

    save_scaffold_summary_markdown(summary_df, ROOT / "results/scaffold_validation_results.md")
    save_scaffold_matlab_script(ROOT / "results/scaffold_validation_results.m")

    summary_json = {
        "repeat_seeds": repeat_seeds,
        "best_model_by_rmse_mean": summary_df.iloc[0]["model"],
        "best_rmse_mean": float(summary_df.iloc[0]["rmse_mean"]),
        "best_r2_mean": float(summary_df.iloc[0]["r2_mean"]),
        "note": "Scaffold-aware repeated validation is stricter than random split; lower metrics are expected but credibility is higher.",
        "ensemble_weights_repeat0": artifacts["ensemble_weights"],
        "top_shap_features": shap_info["top_features"].head(10).to_dict(orient="records"),
    }
    with open(ROOT / "results/scaffold_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print("Saved rigorous validation outputs:")
    print(f"- {metrics_path}")
    print(f"- {summary_path}")
    print(f"- {ROOT / 'results/scaffold_validation_results.md'}")
    print(f"- {ROOT / 'results/scaffold_validation_results.m'}")
    print(f"- {ROOT / 'results/shap_case_studies.md'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
