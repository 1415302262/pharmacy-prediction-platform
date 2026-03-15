from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from utils_inference import ASSETS_DIR, DemoService, render_molecule


APP_DIR = Path(__file__).resolve().parent
service = DemoService()
app = Flask(__name__, template_folder="templates", static_folder="static")


PROJECT_CONTEXT = {
    "best_random": service.run_summary["test_metrics"]["Weighted_Ensemble_rich"],
    "strict_summary": service.scaffold_summary.to_dict(orient="records"),
}


def image_to_base64(smiles: str) -> str:
    image = render_molecule(smiles)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.get("/")
def index():
    return render_template(
        "index.html",
        best_random=PROJECT_CONTEXT["best_random"],
        strict_summary=PROJECT_CONTEXT["strict_summary"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/files/<path:subpath>")
def project_files(subpath: str):
    return send_from_directory(ASSETS_DIR, subpath)


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    smiles = payload.get("smiles", "")
    try:
        result = service.predict_for_api(smiles)
        result["molecule_image"] = image_to_base64(result["canonical_smiles"])
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/gemini-explain")
def api_gemini_explain():
    payload = request.get_json(silent=True) or {}
    smiles = payload.get("smiles", "")
    focus = payload.get("focus", "药化改造")
    try:
        result = service.gemini_explain_for_api(smiles, focus=focus)
        result["molecule_image"] = image_to_base64(result["canonical_smiles"])
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
