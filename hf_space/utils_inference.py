from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor, Pool
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from xgboost import XGBRegressor

try:
    from google import genai
except Exception:
    genai = None

from featurization import (
    compute_full_descriptors,
    compute_maccs_keys,
    compute_morgan_fingerprint,
    mol_from_smiles,
)


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODELS_DIR = ASSETS_DIR / "models"
RESULTS_DIR = ASSETS_DIR / "results"
FIGURES_DIR = ASSETS_DIR / "figures"


class FingerprintMLP(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(1024, 384),
            torch.nn.BatchNorm1d(384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.18),
            torch.nn.Linear(384, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


@dataclass
class PredictionOutput:
    canonical_smiles: str
    ensemble_prediction: float
    catboost_prediction: float
    interpretation: str
    top_positive: pd.DataFrame
    top_negative: pd.DataFrame


def _response_text(response) -> str:
    text = getattr(response, "text", "")
    if text:
        return text.strip()
    try:
        candidates = getattr(response, "candidates", [])
        parts = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []):
                part_text = getattr(part, "text", "")
                if part_text:
                    parts.append(part_text)
        return "\n".join(parts).strip()
    except Exception:
        return ""


class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("ECOSORT_GEMINI_API_KEY", "").strip()
        self.model_name = os.getenv("ECOSORT_GEMINI_MODEL", "gemini-2.5-flash").strip()
        self.client = None
        if self.api_key and genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.client is not None)

    def explain_prediction(self, prediction_payload: dict, user_focus: str = "药化改造") -> dict:
        if not self.available:
            raise RuntimeError(
                "Gemini 尚未配置。请在 Hugging Face Space Settings -> Secrets 中添加 ECOSORT_GEMINI_API_KEY，"
                "并在 Variables 中添加 ECOSORT_GEMINI_MODEL（可选）。"
            )

        positive = prediction_payload.get("top_positive_features", [])
        negative = prediction_payload.get("top_negative_features", [])

        prompt = f"""
你是一个帮助药学和计算机交叉团队理解 QSAR 结果的药物化学助理。

请根据下面这个分子的模型输出，写一段结构化中文解释，面向药学复试和项目展示。

要求：
1. 用中文输出。
2. 不要编造实验结论，明确说明这是模型辅助分析，不替代实验。
3. 输出分成 4 个小节，每节标题前加 ## ：
   - ## 模型结论
   - ## 可能的结构原因
   - ## 药化优化建议
   - ## 使用风险提醒
4. “药化优化建议”请尽量具体，说明如果想提高/降低脂溶性，可以优先考虑哪些结构方向。
5. 结合正负 SHAP 特征解释，不要只重复预测值。
6. 语言要适合直接展示在网页中。

当前关注点：{user_focus}

分子信息：
- canonical_smiles: {prediction_payload.get('canonical_smiles', '')}
- ensemble_prediction: {prediction_payload.get('ensemble_prediction', '')}
- catboost_prediction: {prediction_payload.get('catboost_prediction', '')}
- interpretation: {prediction_payload.get('interpretation', '')}

提高预测值的主要特征：
{json.dumps(positive, ensure_ascii=False, indent=2)}

降低预测值的主要特征：
{json.dumps(negative, ensure_ascii=False, indent=2)}
""".strip()

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        text = _response_text(response)
        if not text:
            raise RuntimeError("Gemini 返回为空，请稍后重试。")

        return {
            "gemini_model": self.model_name,
            "analysis_markdown": text,
        }


def render_molecule(smiles: str, size: tuple[int, int] = (420, 280)) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效 SMILES，无法绘制结构。")
    return Draw.MolToImage(mol, size=size)


def summarize_lipophilicity(value: float) -> str:
    if value < 0.0:
        return "较低脂溶性：分子更偏亲水，可能更易溶于水，但膜通透性可能受限。"
    if value < 2.0:
        return "中低脂溶性：在水溶性和膜通透性之间较平衡。"
    if value < 3.5:
        return "中等脂溶性：常见于较多药物样分子，兼顾一定疏水性与可分布性。"
    return "较高脂溶性：更偏疏水，可能有利于膜通透，但也需要关注溶解性与暴露风险。"


class DemoService:
    def __init__(self):
        import pickle

        with open(MODELS_DIR / "rich_descriptor_processor.pkl", "rb") as f:
            self.processor = pickle.load(f)

        self.feature_names = (
            self.processor.get_feature_names_out()
            + [f"Morgan_{idx:04d}" for idx in range(2048)]
            + [f"MACCS_{idx:03d}" for idx in range(167)]
        )

        self.xgb = XGBRegressor()
        self.xgb.load_model(str(MODELS_DIR / "xgb_rich_desc_fp_maccs.json"))

        self.catboost = CatBoostRegressor()
        self.catboost.load_model(str(MODELS_DIR / "catboost_rich_desc_fp_maccs.cbm"))

        checkpoint = torch.load(MODELS_DIR / "pytorch_mlp_rich_desc_fp_maccs.pt", map_location="cpu")
        input_dim = int(checkpoint["metadata"]["input_dim"])
        self.mlp = FingerprintMLP(input_dim=input_dim)
        self.mlp.load_state_dict(checkpoint["state_dict"])
        self.mlp.eval()

        with open(RESULTS_DIR / "run_summary.json", "r", encoding="utf-8") as f:
            self.run_summary = json.load(f)
        self.ensemble_weights = self.run_summary["ensemble_weights"]

        self.shap_top20 = pd.read_csv(RESULTS_DIR / "shap_top20_features.csv")
        self.scaffold_summary = pd.read_csv(RESULTS_DIR / "scaffold_repeat_summary.csv")
        self.baseline_vs_enhanced = pd.read_csv(RESULTS_DIR / "baseline_vs_enhanced.csv")
        self.gemini = GeminiService()

    def validate_smiles(self, smiles: str) -> str:
        if not smiles or not smiles.strip():
            raise ValueError("请输入 SMILES。")
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            raise ValueError("SMILES 无法被 RDKit 解析，请检查输入格式。")
        return Chem.MolToSmiles(mol)

    def featurize_single(self, smiles: str) -> np.ndarray:
        canonical_smiles = self.validate_smiles(smiles)
        mol = mol_from_smiles(canonical_smiles)
        desc_df = pd.DataFrame([compute_full_descriptors(mol)], columns=[name for name, _ in Descriptors.descList])
        x_desc = self.processor.transform(desc_df)
        x_fp = compute_morgan_fingerprint(mol).reshape(1, -1)
        x_maccs = compute_maccs_keys(mol).reshape(1, -1)
        return canonical_smiles, np.concatenate([x_desc, x_fp, x_maccs], axis=1).astype(np.float32)

    def mlp_predict(self, X: np.ndarray) -> float:
        with torch.no_grad():
            tensor = torch.as_tensor(np.array(X, copy=True), dtype=torch.float32)
            return float(self.mlp(tensor).cpu().numpy()[0])

    def local_shap(self, X: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        pool = Pool(X)
        shap_values = self.catboost.get_feature_importance(pool, type="ShapValues")[:, :-1][0]
        plot_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "shap_value": shap_values,
                "abs_shap": np.abs(shap_values),
                "direction": np.where(shap_values >= 0, "提高预测值", "降低预测值"),
            }
        ).sort_values("abs_shap", ascending=False)
        positive = plot_df[plot_df["shap_value"] > 0].head(6)[["feature", "shap_value", "direction"]]
        negative = plot_df[plot_df["shap_value"] < 0].head(6)[["feature", "shap_value", "direction"]]
        return positive, negative

    def predict(self, smiles: str) -> PredictionOutput:
        canonical_smiles, X = self.featurize_single(smiles)
        xgb_pred = float(self.xgb.predict(X)[0])
        catboost_pred = float(self.catboost.predict(X)[0])
        mlp_pred = self.mlp_predict(X)

        ensemble_pred = (
            self.ensemble_weights["XGB_rich_desc_fp_maccs"] * xgb_pred
            + self.ensemble_weights["CatBoost_rich_desc_fp_maccs"] * catboost_pred
            + self.ensemble_weights["PyTorch_MLP_rich_desc_fp_maccs"] * mlp_pred
        )

        positive, negative = self.local_shap(X)
        interpretation = summarize_lipophilicity(ensemble_pred)
        return PredictionOutput(
            canonical_smiles=canonical_smiles,
            ensemble_prediction=float(ensemble_pred),
            catboost_prediction=catboost_pred,
            interpretation=interpretation,
            top_positive=positive,
            top_negative=negative,
        )

    def predict_for_ui(self, smiles: str):
        result = self.predict(smiles)
        image = render_molecule(result.canonical_smiles)
        summary_md = (
            f"### 预测结果\n"
            f"- 标准化 SMILES：`{result.canonical_smiles}`\n"
            f"- 集成模型预测脂溶性：`{result.ensemble_prediction:.4f}`\n"
            f"- 解释模型（CatBoost）预测：`{result.catboost_prediction:.4f}`\n"
            f"- 解读：{result.interpretation}"
        )
        return image, summary_md, result.top_positive, result.top_negative

    def predict_for_api(self, smiles: str) -> dict:
        result = self.predict(smiles)
        return {
            "canonical_smiles": result.canonical_smiles,
            "ensemble_prediction": round(result.ensemble_prediction, 6),
            "catboost_prediction": round(result.catboost_prediction, 6),
            "interpretation": result.interpretation,
            "top_positive_features": result.top_positive.to_dict(orient="records"),
            "top_negative_features": result.top_negative.to_dict(orient="records"),
        }

    def gemini_explain_for_api(self, smiles: str, focus: str = "药化改造") -> dict:
        prediction = self.predict_for_api(smiles)
        gemini_result = self.gemini.explain_prediction(prediction, user_focus=focus)
        return {
            **prediction,
            **gemini_result,
        }


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""
