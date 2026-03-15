from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA


def add_scaffold_column(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["scaffold"] = work["canonical_smiles"].map(lambda smiles: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles)))
    return work


def plot_chemical_space(feature_matrix: np.ndarray, target: np.ndarray, split: pd.Series, path: str | Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(feature_matrix)
    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "target": target,
        "split": split.values,
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    scatter = ax.scatter(
        plot_df["PC1"],
        plot_df["PC2"],
        c=plot_df["target"],
        cmap="viridis",
        alpha=0.75,
        s=22,
        edgecolors="none",
    )
    ax.set_title("Chemical space projection of enriched structural features")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Experimental lipophilicity")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_top_scaffolds(df: pd.DataFrame, path: str | Path, top_n: int = 12) -> pd.DataFrame:
    scaffold_df = add_scaffold_column(df)
    top_scaffolds = (
        scaffold_df.loc[scaffold_df["scaffold"] != "", "scaffold"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    top_scaffolds.columns = ["scaffold", "count"]
    top_scaffolds["label"] = [f"S{i + 1}" for i in range(len(top_scaffolds))]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.barplot(data=top_scaffolds, x="label", y="count", hue="label", dodge=False, legend=False, palette="cubehelix", ax=ax)
    ax.set_title("Top Bemis–Murcko scaffolds")
    ax.set_xlabel("Scaffold ID")
    ax.set_ylabel("Molecule count")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return top_scaffolds


def save_top_scaffold_grid(df: pd.DataFrame, path: str | Path, top_n: int = 9) -> None:
    scaffold_df = add_scaffold_column(df)
    top_scaffolds = scaffold_df.loc[scaffold_df["scaffold"] != "", "scaffold"].value_counts().head(top_n)

    scaffold_mols = [Chem.MolFromSmiles(scaffold) for scaffold in top_scaffolds.index]
    legends = [f"Scaffold {idx + 1}\nn={count}" for idx, count in enumerate(top_scaffolds.values)]
    image = Draw.MolsToGridImage(scaffold_mols, molsPerRow=3, subImgSize=(280, 220), legends=legends, useSVG=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))


def save_extreme_molecule_grid(df: pd.DataFrame, path: str | Path, n_each: int = 6) -> None:
    ordered = df.sort_values("target")
    low_df = ordered.head(n_each)
    high_df = ordered.tail(n_each)
    display_df = pd.concat([low_df, high_df], axis=0)
    mols = [Chem.MolFromSmiles(smiles) for smiles in display_df["canonical_smiles"]]
    legends = [f"exp={value:.2f}" for value in display_df["target"]]
    image = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(280, 220), legends=legends, useSVG=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))
