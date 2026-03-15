"""
Statistical Analysis Module for QSAR Research
统计分析模块 - 用于高级统计分析和结果解释
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


def descriptive_statistics(
    data: Union[pd.Series, np.ndarray],
    name: str = "Variable",
) -> Dict[str, float]:
    """
    Calculate comprehensive descriptive statistics.

    Args:
        data: Input data
        name: Variable name for reporting

    Returns:
        Dictionary of statistics
    """
    data = np.asarray(data)
    return {
        'name': name,
        'n': len(data),
        'mean': float(np.mean(data)),
        'std': float(np.std(data, ddof=1)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data)),
        'cv': float(np.std(data, ddof=1) / np.mean(data)) if np.mean(data) != 0 else np.nan,
    }


def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate confidence interval for the mean.

    Args:
        data: Input data
        confidence: Confidence level (default 0.95)

    Returns:
        (lower, upper) bounds
    """
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    ci = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(mean - ci), float(mean + ci)


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate bootstrap confidence intervals for metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        Dictionary with mean and CI for each metric
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    r2_samples, rmse_samples, mae_samples = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        r2_samples.append(stats.pearsonr(y_true_boot, y_pred_boot)[0] ** 2)
        rmse_samples.append(np.sqrt(np.mean((y_true_boot - y_pred_boot) ** 2)))
        mae_samples.append(np.mean(np.abs(y_true_boot - y_pred_boot)))

    results = {}
    for name, samples in [('r2', r2_samples), ('rmse', rmse_samples), ('mae', mae_samples)]:
        results[name] = {
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'ci_lower': float(np.percentile(samples, 2.5)),
            'ci_upper': float(np.percentile(samples, 97.5)),
        }

    return results


def normality_test(
    data: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, any]:
    """
    Perform normality tests.

    Args:
        data: Input data
        alpha: Significance level

    Returns:
        Test results
    """
    shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Shapiro limit
    k2, normaltest_p = stats.normaltest(data)

    return {
        'shapiro_statistic': float(shapiro_stat),
        'shapiro_pvalue': float(shapiro_p),
        'shapiro_normal': shapiro_p > alpha,
        'dagostino_statistic': float(k2),
        'dagostino_pvalue': float(normaltest_p),
        'dagostino_normal': normaltest_p > alpha,
        'n': len(data),
    }


def correlation_analysis(
    df: pd.DataFrame,
    target: str,
    features: List[str],
) -> pd.DataFrame:
    """
    Analyze correlations between features and target.

    Args:
        df: DataFrame containing features and target
        target: Target column name
        features: List of feature column names

    Returns:
        DataFrame with correlation statistics
    """
    results = []
    for feat in features:
        x = df[feat].values
        y = df[target].values

        # Remove NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]

        if len(x_clean) < 10:
            continue

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)

        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)

        results.append({
            'feature': feat,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n': len(x_clean),
        })

    return pd.DataFrame(results)


def pairwise_model_comparison(
    metrics_dict: Dict[str, Dict[str, Union[np.ndarray, float]]],
    metric: str = 'rmse',
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Perform pairwise statistical comparison between models.

    Args:
        metrics_dict: Dictionary of model predictions {model_name: {'y_true': ..., 'y_pred': ...}}
        metric: Metric to compare ('rmse', 'mae', 'r2')
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        DataFrame with pairwise comparisons
    """
    models = list(metrics_dict.keys())
    results = []

    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            y_true = metrics_dict[model1]['y_true']
            y_pred1 = metrics_dict[model1]['y_pred']
            y_pred2 = metrics_dict[model2]['y_pred']

            if metric == 'rmse':
                errors1 = (y_true - y_pred1) ** 2
                errors2 = (y_true - y_pred2) ** 2
            elif metric == 'mae':
                errors1 = np.abs(y_true - y_pred1)
                errors2 = np.abs(y_true - y_pred2)
            elif metric == 'r2':
                errors1 = 1 - stats.pearsonr(y_true, y_pred1)[0] ** 2
                errors2 = 1 - stats.pearsonr(y_true, y_pred2)[0] ** 2
            else:
                raise ValueError(f"Unknown metric: {metric}")

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(errors1, errors2)

            # Bootstrap
            rng = np.random.RandomState(random_state)
            n = len(errors1)
            bootstrap_diffs = []

            for _ in range(n_bootstrap):
                idx = rng.choice(n, n, replace=True)
                diff = np.mean(errors1[idx]) - np.mean(errors2[idx])
                bootstrap_diffs.append(diff)

            ci_lower = np.percentile(bootstrap_diffs, 2.5)
            ci_upper = np.percentile(bootstrap_diffs, 97.5)

            results.append({
                'model1': model1,
                'model2': model2,
                'difference': float(np.mean(errors1) - np.mean(errors2)),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
            })

    return pd.DataFrame(results)


def effect_size(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate effect size between two models.

    Args:
        y_true: True values
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2

    Returns:
        Effect size metrics
    """
    errors1 = (y_true - y_pred1) ** 2
    errors2 = (y_true - y_pred2) ** 2

    # Cohen's d (difference in MSE)
    diff = np.mean(errors1) - np.mean(errors2)
    pooled_std = np.sqrt((np.var(errors1, ddof=1) + np.var(errors2, ddof=1)) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0

    # Relative improvement
    rel_improvement = diff / np.mean(errors1) * 100

    return {
        'difference': float(diff),
        'cohens_d': float(cohens_d),
        'relative_improvement_percent': float(rel_improvement),
    }


def save_statistics_report(
    stats_dict: Dict,
    path: Union[str, Path],
) -> None:
    """
    Save statistics report as markdown.

    Args:
        stats_dict: Statistics dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Statistical Analysis Report\n"]

    for section, content in stats_dict.items():
        lines.append(f"\n## {section}\n")

        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.4f}")
                else:
                    lines.append(f"- {key}: {value}")
        elif isinstance(content, pd.DataFrame):
            lines.append(content.to_string(index=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
