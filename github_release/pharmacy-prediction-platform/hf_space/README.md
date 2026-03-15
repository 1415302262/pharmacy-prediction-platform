---
title: Drug Lipophilicity QSAR XAI Demo
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Drug Lipophilicity QSAR XAI Demo

This Hugging Face Space packages the project as a public Flask web app with JSON API support.

Main features:

- single-molecule lipophilicity prediction from SMILES
- ensemble model output
- scaffold-based strict validation summary
- SHAP-inspired explainability view
- optional Gemini-assisted medicinal chemistry explanation
- project result gallery

## Gemini Configuration

If you want Gemini-assisted explanation in the Space, configure:

- **Secrets**: `ECOSORT_GEMINI_API_KEY`
- **Variables**: `ECOSORT_GEMINI_MODEL=gemini-2.5-flash`

After that, the platform will expose:

- `POST /api/gemini-explain`

Example request:

```json
{
  "smiles": "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14",
  "focus": "药化改造"
}
```

Before deploying this folder as a Space, refresh bundled assets from the main project:

```bash
python scripts/build_hf_space_bundle.py
```
