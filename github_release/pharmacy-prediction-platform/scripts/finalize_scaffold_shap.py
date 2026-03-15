from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))

from data_utils import download_dataset, load_and_prepare_dataset
from featurization import featurize_dataframe
from scaffold_validation import (
    grouped_scaffold_split,
    prepare_feature_sets,
    save_scaffold_matlab_script,
    save_scaffold_summary_markdown,
    save_shap_artifacts,
)
from train_model import train_catboost


def main() -> None:
    download_dataset()
    df = load_and_prepare_dataset()
    bundle = featurize_dataframe(df)

    summary_df = __import__('pandas').read_csv(ROOT / 'results/scaffold_repeat_summary.csv')
    save_scaffold_summary_markdown(summary_df, ROOT / 'results/scaffold_validation_results.md')
    save_scaffold_matlab_script(ROOT / 'results/scaffold_validation_results.m')

    split_df = grouped_scaffold_split(df, random_state=11)
    features = prepare_feature_sets(bundle, split_df)

    catboost_rich = train_catboost(
        features['X_rich_train'],
        features['y_train'],
        features['X_rich_valid'],
        features['y_valid'],
        use_gpu=True,
        depth=8,
        learning_rate=0.03,
        iterations=2000,
        l2_leaf_reg=5.0,
    )
    catboost_pred = catboost_rich.predict(features['X_rich_test'])

    test_idx = features['test_idx']
    case_df = split_df.iloc[test_idx][['chembl_id', 'canonical_smiles', 'target']].copy()
    case_df = case_df.rename(columns={'target': 'observed'}).reset_index(drop=True)
    case_df['predicted'] = catboost_pred

    shap_info = save_shap_artifacts(
        model=catboost_rich,
        X_test=features['X_rich_test'],
        y_test=features['y_test'],
        y_pred=catboost_pred,
        feature_names=features['feature_names'],
        case_df=case_df,
        output_dir=ROOT / 'results',
    )

    summary_json = {
        'best_model_by_rmse_mean': summary_df.iloc[0]['model'],
        'best_rmse_mean': float(summary_df.iloc[0]['rmse_mean']),
        'best_r2_mean': float(summary_df.iloc[0]['r2_mean']),
        'strict_validation_note': 'Scaffold-aware repeated validation is stricter than random split; lower metrics are expected but credibility is higher.',
        'repeat0_shap_model': 'CatBoost_rich_scaffold',
        'top_shap_features': shap_info['top_features'].head(10).to_dict(orient='records'),
    }
    with open(ROOT / 'results/scaffold_validation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    catboost_rich.save_model(str(ROOT / 'models/catboost_rich_scaffold_repeat0.cbm'))
    print('Saved scaffold SHAP artifacts and summary files.')


if __name__ == '__main__':
    main()
