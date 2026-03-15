from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator, rdMolDescriptors
from sklearn.preprocessing import StandardScaler


BASIC_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotBonds",
    "RingCount",
    "AromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "LabuteASA",
    "BalabanJ",
    "QED",
]

FULL_DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
FULL_DESCRIPTOR_FUNCS: list[Callable] = [func for _, func in Descriptors.descList]

MORGAN_BITS = 2048
MACCS_BITS = 167
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_BITS)


@dataclass
class FeatureBundle:
    X_desc_basic: np.ndarray
    X_desc_full: pd.DataFrame
    X_fp: np.ndarray
    X_maccs: np.ndarray
    y: np.ndarray
    mols: list
    basic_feature_names: List[str]
    full_feature_names: List[str]


class DescriptorProcessor:
    def __init__(self, corr_threshold: float = 0.98):
        self.corr_threshold = corr_threshold
        self.keep_columns_: list[str] = []
        self.medians_: pd.Series | None = None
        self.scaler_ = StandardScaler()

    def fit(self, descriptor_frame: pd.DataFrame) -> "DescriptorProcessor":
        work = descriptor_frame.replace([np.inf, -np.inf], np.nan).copy()
        non_missing_mask = work.notna().all(axis=0)
        variance_mask = work.nunique(dropna=False) > 1
        work = work.loc[:, non_missing_mask & variance_mask]

        corr = work.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
        work = work.drop(columns=to_drop)

        self.keep_columns_ = work.columns.tolist()
        self.medians_ = work.median(axis=0)
        self.scaler_.fit(work.to_numpy(dtype=np.float32))
        return self

    def transform(self, descriptor_frame: pd.DataFrame) -> np.ndarray:
        if not self.keep_columns_ or self.medians_ is None:
            raise RuntimeError("DescriptorProcessor must be fitted before transform().")
        work = descriptor_frame.replace([np.inf, -np.inf], np.nan).copy()
        work = work.reindex(columns=self.keep_columns_)
        work = work.fillna(self.medians_)
        transformed = self.scaler_.transform(work.to_numpy(dtype=np.float32))
        return transformed.astype(np.float32)

    def fit_transform(self, descriptor_frame: pd.DataFrame) -> np.ndarray:
        return self.fit(descriptor_frame).transform(descriptor_frame)

    def get_feature_names_out(self) -> list[str]:
        return list(self.keep_columns_)


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid smiles: {smiles}")
    return mol


def compute_basic_descriptors(mol) -> List[float]:
    return [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        Lipinski.FractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcLabuteASA(mol),
        Descriptors.BalabanJ(mol),
        QED.qed(mol),
    ]


def compute_full_descriptors(mol) -> list[float]:
    values: list[float] = []
    for func in FULL_DESCRIPTOR_FUNCS:
        try:
            values.append(float(func(mol)))
        except Exception:
            values.append(np.nan)
    return values


def compute_morgan_fingerprint(mol, n_bits: int = MORGAN_BITS) -> np.ndarray:
    fp = _MORGAN_GENERATOR.GetFingerprint(mol)
    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def compute_maccs_keys(mol, n_bits: int = MACCS_BITS) -> np.ndarray:
    fp = MACCSkeys.GenMACCSKeys(mol)
    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def featurize_dataframe(df: pd.DataFrame) -> FeatureBundle:
    mols = [mol_from_smiles(smiles) for smiles in df["canonical_smiles"]]
    X_desc_basic = np.asarray([compute_basic_descriptors(mol) for mol in mols], dtype=np.float32)
    X_desc_full = pd.DataFrame([compute_full_descriptors(mol) for mol in mols], columns=FULL_DESCRIPTOR_NAMES)
    X_fp = np.asarray([compute_morgan_fingerprint(mol) for mol in mols], dtype=np.float32)
    X_maccs = np.asarray([compute_maccs_keys(mol) for mol in mols], dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.float32)
    return FeatureBundle(
        X_desc_basic=X_desc_basic,
        X_desc_full=X_desc_full,
        X_fp=X_fp,
        X_maccs=X_maccs,
        y=y,
        mols=mols,
        basic_feature_names=BASIC_DESCRIPTOR_NAMES,
        full_feature_names=FULL_DESCRIPTOR_NAMES,
    )
