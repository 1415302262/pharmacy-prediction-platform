from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "hf_space"
ASSETS_DIR = SPACE_DIR / "assets"


FILES_TO_COPY = {
    "models": [
        "xgb_rich_desc_fp_maccs.json",
        "catboost_rich_desc_fp_maccs.cbm",
        "pytorch_mlp_rich_desc_fp_maccs.pt",
        "rich_descriptor_processor.pkl",
    ],
    "results": [
        "run_summary.json",
        "baseline_vs_enhanced.csv",
        "scaffold_repeat_summary.csv",
        "scaffold_validation_summary.json",
        "shap_top20_features.csv",
        "shap_case_studies.md",
    ],
    "figures": [
        "08_chemical_space.png",
        "14_baseline_vs_enhanced.png",
        "16_scaffold_repeat_summary.png",
        "19_shap_top20_bar.png",
        "22_shap_case_molecules.png",
    ],
}


def copy_file_group(group_name: str, filenames: list[str]) -> None:
    source_dir = ROOT / group_name
    target_dir = ASSETS_DIR / group_name
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        source = source_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing required asset: {source}")
        shutil.copy2(source, target_dir / filename)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for group_name, filenames in FILES_TO_COPY.items():
        copy_file_group(group_name, filenames)

    shutil.copy2(ROOT / "src" / "featurization.py", SPACE_DIR / "featurization.py")
    print(f"Hugging Face Space bundle refreshed at: {SPACE_DIR}")


if __name__ == "__main__":
    main()
