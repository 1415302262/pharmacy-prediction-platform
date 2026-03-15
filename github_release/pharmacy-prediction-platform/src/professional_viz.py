"""
Professional Visualization Module for QSAR Research
科研级可视化模块 - 用于发表级图表生成
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# Professional Color Palettes (ColorBrewer Scientific)
SCIENTIFIC_COLORS = {
    "primary": "#3B4252",
    "secondary": "#4C566A",
    "blue": "#5E81AC",
    "green": "#A3BE8C",
    "red": "#BF616A",
    "yellow": "#EBCB8B",
    "purple": "#B48EAD",
    "orange": "#D08770",
    "cyan": "#88C0D0",
}

MODEL_COLORS = {
    "XGB": SCIENTIFIC_COLORS["blue"],
    "CatBoost": SCIENTIFIC_COLORS["green"],
    "PyTorch": SCIENTIFIC_COLORS["purple"],
    "Ensemble": SCIENTIFIC_COLORS["red"],
    "RF": SCIENTIFIC_COLORS["orange"],
    "SVR": SCIENTIFIC_COLORS["cyan"],
}

# Professional settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 12,
    'text.usetex': False,  # Set to True if LaTeX is available
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'patch.linewidth': 1.0,
})


def set_journal_style(journal: str = 'nature') -> None:
    """
    Set visualization style for different journals.

    Args:
        journal: 'nature', 'science', 'cell', 'acs', 'default'
    """
    styles = {
        'nature': {'figsize': (3.5, 2.5), 'font_size': 7},
        'science': {'figsize': (4, 3), 'font_size': 8},
        'cell': {'figsize': (5, 4), 'font_size': 8},
        'acs': {'figsize': (4, 3), 'font_size': 9},
        'default': {'figsize': (6, 4.5), 'font_size': 10},
    }

    if journal in styles:
        plt.rcParams['font.size'] = styles[journal]['font_size']
        plt.rcParams['axes.labelsize'] = styles[journal]['font_size']
        plt.rcParams['axes.titlesize'] = styles[journal]['font_size'] + 1
        plt.rcParams['xtick.labelsize'] = styles[journal]['font_size']
        plt.rcParams['ytick.labelsize'] = styles[journal]['font_size']


def figure_1_dataset_overview(
    df: pd.DataFrame,
    path: Union[str, Path],
    figsize: Tuple[float, float] = (12, 4),
) -> None:
    """
    Figure 1: Dataset Overview with Distribution and Statistics

    Shows target distribution across splits with statistical annotations.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)

    # Panel A: Distribution
    ax = axes[0]
    for split in ['train', 'valid', 'test']:
        data = df[df['split'] == split]['target']
        color = SCIENTIFIC_COLORS[list(SCIENTIFIC_COLORS.keys())[['train', 'valid', 'test'].index(split)]]
        ax.hist(data, bins=25, alpha=0.6, label=f'{split.capitalize()} (n={len(data)})',
                color=color, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Experimental Lipophilicity (logD)')
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')
    ax.set_title('(a) Target Distribution', loc='left', weight='bold')

    # Panel B: Box plot
    ax = axes[1]
    plot_data = [df[df['split'] == split]['target'].values for split in ['train', 'valid', 'test']]
    bp = ax.boxplot(plot_data, labels=['Train', 'Validation', 'Test'],
                    patch_artist=True, widths=0.6,
                    boxprops=dict(facecolor='white', edgecolor=SCIENTIFIC_COLORS['primary']),
                    medianprops=dict(color=SCIENTIFIC_COLORS['red'], linewidth=2),
                    whiskerprops=dict(color=SCIENTIFIC_COLORS['primary']),
                    capprops=dict(color=SCIENTIFIC_COLORS['primary']))
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor(SCIENTIFIC_COLORS['primary'])
    ax.set_ylabel('Experimental Lipophilicity (logD)')
    ax.set_title('(b) Statistical Summary', loc='left', weight='bold')

    # Panel C: Sample size
    ax = axes[2]
    split_counts = df['split'].value_counts().reindex(['train', 'valid', 'test'])
    colors = [SCIENTIFIC_COLORS['blue'], SCIENTIFIC_COLORS['green'], SCIENTIFIC_COLORS['red']]
    bars = ax.bar(range(3), split_counts.values, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Train', 'Validation', 'Test'])
    ax.set_ylabel('Number of Molecules')
    ax.set_title('(c) Dataset Splits', loc='left', weight='bold')

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, split_counts.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=9, weight='bold')

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def figure_2_chemical_space(
    X: np.ndarray,
    y: np.ndarray,
    split: pd.Series,
    path: Union[str, Path],
    figsize: Tuple[float, float] = (10, 4),
) -> None:
    """
    Figure 2: Chemical Space Analysis

    Shows PCA projection with statistical annotations.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel A: By split
    ax = axes[0]
    for split_name, color in [('train', SCIENTIFIC_COLORS['blue']),
                               ('valid', SCIENTIFIC_COLORS['green']),
                               ('test', SCIENTIFIC_COLORS['red'])]:
        mask = split == split_name
        ax.scatter(coords[mask, 0], coords[mask, 1],
                 c=color, alpha=0.5, s=20, label=f'{split_name.capitalize()}',
                 edgecolors='none')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, markerscale=1.5)
    ax.set_title('(a) Chemical Space by Split', loc='left', weight='bold')

    # Panel B: By target value
    ax = axes[1]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y,
                      cmap='viridis', alpha=0.6, s=25,
                      vmin=np.percentile(y, 5), vmax=np.percentile(y, 95))
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('(b) Chemical Space Colored by Lipophilicity', loc='left', weight='bold')
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('logD', rotation=0, labelpad=10)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def figure_3_model_performance(
    metrics_df: pd.DataFrame,
    path: Union[str, Path],
    figsize: Tuple[float, float] = (14, 4.5),
) -> None:
    """
    Figure 3: Model Performance Comparison

    Comprehensive model comparison with multiple metrics.
    """
    test_df = metrics_df[metrics_df['split'] == 'test'].sort_values('rmse')

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel A: RMSE
    ax = axes[0]
    colors = [MODEL_COLORS.get(m.replace('_desc_fp_maccs', '').replace('_rich', '').split('_')[0],
                               SCIENTIFIC_COLORS['secondary']) for m in test_df['model']]
    bars = ax.barh(range(len(test_df)), test_df['rmse'], color=colors,
                   edgecolor='white', linewidth=1)
    ax.set_yticks(range(len(test_df)))
    ax.set_yticklabels([m.replace('_desc_fp_maccs', '').replace('_rich', '')
                      for m in test_df['model']], fontsize=9)
    ax.set_xlabel('RMSE $\\downarrow$')
    ax.set_title('(a) Root Mean Square Error', loc='left', weight='bold')
    ax.invert_xaxis()

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, test_df['rmse'])):
        ax.text(value + 0.01, i, f'{value:.3f}',
               va='center', ha='left', fontsize=8)

    # Panel B: R²
    ax = axes[1]
    colors = [MODEL_COLORS.get(m.replace('_desc_fp_maccs', '').replace('_rich', '').split('_')[0],
                               SCIENTIFIC_COLORS['secondary']) for m in test_df['model']]
    bars = ax.barh(range(len(test_df)), test_df['r2'], color=colors,
                   edgecolor='white', linewidth=1)
    ax.set_yticks(range(len(test_df)))
    ax.set_yticklabels([m.replace('_desc_fp_maccs', '').replace('_rich', '')
                      for m in test_df['model']], fontsize=9)
    ax.set_xlabel('$R^2$ $\\uparrow$')
    ax.set_title('(b) Coefficient of Determination', loc='left', weight='bold')

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, test_df['r2'])):
        ax.text(value - 0.01, i, f'{value:.3f}',
               va='center', ha='right', fontsize=8, color='white', weight='bold')

    # Panel C: MAE
    ax = axes[2]
    colors = [MODEL_COLORS.get(m.replace('_desc_fp_maccs', '').replace('_rich', '').split('_')[0],
                               SCIENTIFIC_COLORS['secondary']) for m in test_df['model']]
    bars = ax.barh(range(len(test_df)), test_df['mae'], color=colors,
                   edgecolor='white', linewidth=1)
    ax.set_yticks(range(len(test_df)))
    ax.set_yticklabels([m.replace('_desc_fp_maccs', '').replace('_rich', '')
                      for m in test_df['model']], fontsize=9)
    ax.set_xlabel('MAE $\\downarrow$')
    ax.set_title('(c) Mean Absolute Error', loc='left', weight='bold')
    ax.invert_xaxis()

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, test_df['mae'])):
        ax.text(value + 0.005, i, f'{value:.3f}',
               va='center', ha='left', fontsize=8)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def figure_4_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    metrics: Dict[str, float],
    path: Union[str, Path],
    figsize: Tuple[float, float] = (5, 4.5),
    color: str = SCIENTIFIC_COLORS['blue'],
) -> None:
    """
    Figure 4: Prediction vs Observation Scatter Plot

    Professional scatter plot with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate confidence interval
    residuals = y_true - y_pred
    ci_95 = 1.96 * np.std(residuals)

    # Scatter plot
    ax.scatter(y_true, y_pred, s=35, alpha=0.7, color=color,
              edgecolors='white', linewidth=0.5)

    # Perfect prediction line
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], '--', color=SCIENTIFIC_COLORS['red'],
            linewidth=2, alpha=0.8, label='Perfect Prediction')

    # 95% CI band
    ax.fill_between([lo, hi], [lo-ci_95, hi-ci_95], [lo+ci_95, hi+ci_95],
                   alpha=0.2, color=color, label='95% CI')

    # Metrics text box
    textstr = f'$R^2 = {metrics["r2"]:.3f}$\n$RMSE = {metrics["rmse"]:.3f}$\n$MAE = {metrics["mae"]:.3f}$'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.set_xlabel('Observed logD')
    ax.set_ylabel('Predicted logD')
    ax.set_title(f'{model_name}', weight='bold')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def figure_5_residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    path: Union[str, Path],
    figsize: Tuple[float, float] = (12, 3.5),
) -> None:
    """
    Figure 5: Residual Analysis

    Comprehensive residual diagnostics.
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel A: Residuals vs Predicted
    ax = axes[0]
    ax.scatter(y_pred, residuals, s=30, alpha=0.6,
              color=SCIENTIFIC_COLORS['blue'], edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color=SCIENTIFIC_COLORS['red'], linestyle='--', linewidth=1.5)
    # Add LOESS smooth trend
    from scipy.interpolate import UnivariateSpline
    try:
        sort_idx = np.argsort(y_pred)
        spline = UnivariateSpline(y_pred[sort_idx], residuals[sort_idx], s=len(y_pred)*0.5)
        ax.plot(y_pred[sort_idx], spline(y_pred[sort_idx]), '-',
                color=SCIENTIFIC_COLORS['red'], linewidth=1.5, alpha=0.7, label='Trend')
    except:
        pass
    ax.set_xlabel('Predicted logD')
    ax.set_ylabel('Residuals')
    ax.set_title('(a) Residuals vs Predicted', loc='left', weight='bold')
    ax.legend(loc='upper right', fontsize=8)

    # Panel B: Q-Q plot
    ax = axes[1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(SCIENTIFIC_COLORS['blue'])
    ax.get_lines()[0].set_markeredgecolor('white')
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color(SCIENTIFIC_COLORS['red'])
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('(b) Normal Q-Q Plot', loc='left', weight='bold')
    ax.grid(True, alpha=0.3)

    # Panel C: Residual distribution
    ax = axes[2]
    ax.hist(residuals, bins=30, density=True, alpha=0.7,
            color=SCIENTIFIC_COLORS['blue'], edgecolor='white', linewidth=1)
    # Fit normal distribution
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), '--',
            color=SCIENTIFIC_COLORS['red'], linewidth=2, label='Normal Fit')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title('(c) Residual Distribution', loc='left', weight='bold')
    ax.legend(fontsize=8)
    ax.text(0.05, 0.95, f'$\\mu={mu:.3f}$\n$\\sigma={sigma:.3f}$',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def figure_6_feature_importance(
    importance: pd.Series,
    path: Union[str, Path],
    top_n: int = 15,
    figsize: Tuple[float, float] = (6, 4.5),
) -> None:
    """
    Figure 6: Feature Importance Analysis

    Top features with clear labeling.
    """
    top_features = importance.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.magma(np.linspace(0.8, 0.3, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_features.values, color=colors,
                   edgecolor='white', linewidth=1)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=9)
    ax.set_xlabel('Relative Importance')
    ax.set_title('Top Feature Importance', weight='bold')

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_features.values)):
        ax.text(value + 0.0005, i, f'{value:.3f}',
               va='center', ha='left', fontsize=8)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def figure_7_ensemble_weights(
    weights: Dict[str, float],
    path: Union[str, Path],
    figsize: Tuple[float, float] = (6, 4),
) -> None:
    """
    Figure 7: Ensemble Weights Visualization

    Clear display of model contributions.
    """
    plot_df = pd.DataFrame({'Model': list(weights.keys()), 'Weight': list(weights.values())})
    plot_df['Model'] = plot_df['Model'].str.replace('_desc_fp_maccs', '').str.replace('_rich', '')
    plot_df = plot_df.sort_values('Weight', ascending=False)

    colors = [MODEL_COLORS.get(m, SCIENTIFIC_COLORS['secondary']) for m in plot_df['Model']]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(plot_df)), plot_df['Weight'], color=colors,
                 edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df['Model'], rotation=20, ha='right')
    ax.set_ylabel('Weight')
    ax.set_title('Ensemble Model Weights', weight='bold')
    ax.set_ylim(0, max(plot_df['Weight']) * 1.15)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, plot_df['Weight'])):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def figure_8_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str],
    path: Union[str, Path],
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """
    Figure 8: Feature Correlation Heatmap

    Scientific correlation visualization with hierarchical clustering.
    """
    # Select features and compute correlation
    corr_data = df[features].corr()

    # Hierarchical clustering
    from scipy.cluster import hierarchy
    linkage = hierarchy.linkage(corr_data, method='average')
    order = hierarchy.leaves_list(linkage)
    corr_ordered = corr_data.iloc[order, order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_ordered, annot=False, cmap='RdBu_r',
                center=0, square=True, linewidths=0.5,
                cbar_kws={'label': 'Pearson Correlation'}, ax=ax)
    ax.set_title('Feature Correlation Matrix', weight='bold')

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def figure_9_scaffold_analysis(
    scaffold_counts: pd.DataFrame,
    path: Union[str, Path],
    figsize: Tuple[float, float] = (10, 4),
) -> None:
    """
    Figure 9: Scaffold Analysis

    Top scaffolds with structural information.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel A: Scaffold frequency
    ax = axes[0]
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(scaffold_counts)))
    bars = ax.barh(range(len(scaffold_counts)), scaffold_counts['count'].values,
                   color=colors, edgecolor='white', linewidth=1)
    ax.set_yticks(range(len(scaffold_counts)))
    ax.set_yticklabels([f'S{i+1}' for i in range(len(scaffold_counts))])
    ax.set_xlabel('Frequency')
    ax.set_title('(a) Scaffold Frequency', loc='left', weight='bold')

    # Panel B: Cumulative coverage
    ax = axes[1]
    cumulative = scaffold_counts['count'].cumsum() / scaffold_counts['count'].sum() * 100
    ax.plot(range(1, len(cumulative)+1), cumulative, 'o-',
            color=SCIENTIFIC_COLORS['blue'], linewidth=2, markersize=6)
    ax.axhline(y=80, color=SCIENTIFIC_COLORS['red'], linestyle='--',
               alpha=0.7, label='80% coverage')
    ax.fill_between(range(1, len(cumulative)+1), cumulative, 0,
                  alpha=0.2, color=SCIENTIFIC_COLORS['blue'])
    ax.set_xlabel('Number of Top Scaffolds')
    ax.set_ylabel('Cumulative Coverage (%)')
    ax.set_title('(b) Coverage Analysis', loc='left', weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def create_supplementary_figure_performance_table(
    metrics_df: pd.DataFrame,
    path: Union[str, Path],
) -> None:
    """
    Create supplementary table for model performance.
    """
    test_df = metrics_df[metrics_df['split'] == 'test'].sort_values('rmse')

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table_data = [
        ['Model', 'R²', 'RMSE', 'MAE', 'Rank']
    ]
    for i, row in enumerate(test_df.itertuples(), 1):
        model_short = row.model.replace('_desc_fp_maccs', '').replace('_rich', '')
        table_data.append([
            model_short,
            f'{row.r2:.4f}',
            f'{row.rmse:.4f}',
            f'{row.mae:.4f}',
            str(i)
        ])

    table = ax.table(cellText=table_data, cellLoc='center',
                    loc='center', colWidths=[0.4, 0.15, 0.15, 0.15, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor(SCIENTIFIC_COLORS['primary'])
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f9fa')

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)
