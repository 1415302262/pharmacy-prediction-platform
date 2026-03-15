from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict
import json
import urllib.request

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


DATASET_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
RAW_DATA_PATH = Path("data/raw/Lipophilicity.csv")
PROCESSED_DATA_PATH = Path("data/processed/lipophilicity_clean.csv")
SPLIT_DATA_PATH = Path("data/processed/lipophilicity_split.csv")
SUMMARY_PATH = Path("results/dataset_summary.json")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_dataset(url: str = DATASET_URL, dest: Path = RAW_DATA_PATH, force: bool = False) -> Path:
    ensure_parent(dest)
    if dest.exists() and not force:
        return dest
    urllib.request.urlretrieve(url, dest)
    return dest


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work = work.rename(columns={"exp": "target", "CMPD_CHEMBLID": "chembl_id"})
    work["canonical_smiles"] = work["smiles"].map(canonicalize_smiles)
    work = work.dropna(subset=["target", "canonical_smiles"]).copy()
    work["target"] = work["target"].astype(float)
    work = (
        work.groupby("canonical_smiles", as_index=False)
        .agg(
            chembl_id=("chembl_id", "first"),
            smiles=("smiles", "first"),
            target=("target", "mean"),
            n_records=("target", "size"),
        )
    )
    work = work.sort_values("target").reset_index(drop=True)
    return work


def load_and_prepare_dataset(raw_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    clean = clean_dataset(df)
    ensure_parent(PROCESSED_DATA_PATH)
    clean.to_csv(PROCESSED_DATA_PATH, index=False)
    return clean


def stratified_random_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
    n_bins: int = 10,
) -> pd.DataFrame:
    if not np.isclose(train_frac + valid_frac + test_frac, 1.0):
        raise ValueError("split fractions must sum to 1.0")

    work = df.copy().reset_index(drop=True)
    bins = pd.qcut(work["target"], q=n_bins, labels=False, duplicates="drop")
    all_idx = np.arange(len(work))
    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=(1.0 - train_frac),
        random_state=random_state,
        stratify=bins,
    )
    temp_bins = bins.iloc[temp_idx]
    valid_ratio_in_temp = valid_frac / (valid_frac + test_frac)
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - valid_ratio_in_temp),
        random_state=random_state,
        stratify=temp_bins,
    )

    split = np.array(["unassigned"] * len(work), dtype=object)
    split[train_idx] = "train"
    split[valid_idx] = "valid"
    split[test_idx] = "test"
    work["split"] = split
    work.to_csv(SPLIT_DATA_PATH, index=False)
    return work


def scaffold_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    if not np.isclose(train_frac + valid_frac + test_frac, 1.0):
        raise ValueError("split fractions must sum to 1.0")

    work = df.copy().reset_index(drop=True)
    work["scaffold"] = work["canonical_smiles"].map(scaffold_from_smiles)

    scaffold_groups: Dict[str, list[int]] = defaultdict(list)
    for idx, scaffold in enumerate(work["scaffold"]):
        scaffold_groups[scaffold].append(idx)

    grouped_indices = sorted(scaffold_groups.values(), key=len, reverse=True)
    n_total = len(work)
    train_cutoff = int(n_total * train_frac)
    valid_cutoff = int(n_total * (train_frac + valid_frac))

    split_labels = np.array(["unassigned"] * n_total, dtype=object)
    count_train = 0
    count_valid = 0

    for indices in grouped_indices:
        if count_train + len(indices) <= train_cutoff:
            split = "train"
            count_train += len(indices)
        elif count_train + count_valid + len(indices) <= valid_cutoff:
            split = "valid"
            count_valid += len(indices)
        else:
            split = "test"
        split_labels[indices] = split

    work["split"] = split_labels
    return work


def summarize_dataset(df: pd.DataFrame) -> Dict[str, object]:
    summary = {
        "n_molecules": int(len(df)),
        "target_name": "experimental lipophilicity (logD)",
        "target_min": float(df["target"].min()),
        "target_max": float(df["target"].max()),
        "target_mean": float(df["target"].mean()),
        "target_std": float(df["target"].std()),
    }
    if "split" in df.columns:
        split_stats = (
            df.groupby("split")["target"]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4)
            .to_dict(orient="index")
        )
        summary["split_stats"] = split_stats
    ensure_parent(SUMMARY_PATH)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
