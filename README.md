# ADEGuard

AI-powered adverse drug event (ADE) detection, severity classification, and age-aware clustering for VAERS narratives.

## Tech Stack
- Python, FastAPI, Transformers, BioBERT
- Sentence-BERT + HDBSCAN + UMAP
- Next.js + TypeScript + Tailwind CSS
- SHAP, LIME

## Setup
Backend:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r /Users/deakshshetty/Documents/ADE-Guard/requirements.txt
```

Frontend:
```
cd /Users/deakshshetty/Documents/ADE-Guard/frontend
npm install
```

## Run Locally (safe ports)
You mentioned ports `3000`, `4000`, and `8000` are already in use. Use these instead:

Backend (port 8100):
```
COVID_ONLY=1 uvicorn backend.app:app --reload --port 8100
```

Frontend (port 3100):
```
cd /Users/deakshshetty/Documents/ADE-Guard/frontend
NEXT_PUBLIC_API_BASE=http://localhost:8100 npm run dev -- -p 3100
```

## Usage
API base: `http://localhost:8100`
- `POST /api/v1/ner` → entity spans
- `POST /api/v1/severity` → severity label + confidence
- `POST /api/v1/analyze` → ADE spans with severity
- `GET /api/v1/clusters` → age-aware clusters + UMAP points
- `POST /api/v1/explain/severity` → LIME + SHAP token attributions

Example (CLI):
```
python3 /Users/deakshshetty/Documents/ADE-Guard/backend/scripts/infer_pipeline.py \
  --text "Patient developed severe chest pain after Pfizer vaccine."
```

## Label Schema
- `ADE`: adverse drug event mentions
- `DRUG`: drug or vaccine mentions
- `AGE`: age-related spans (used in gold data)

## Architecture
- Data prep: `/Users/deakshshetty/Documents/ADE-Guard/backend/data_preparation.py`
- NER model: `/Users/deakshshetty/Documents/ADE-Guard/backend/train_ner.py`
- Severity model: `/Users/deakshshetty/Documents/ADE-Guard/backend/train_severity.py`
- Inference API: `/Users/deakshshetty/Documents/ADE-Guard/backend/app.py`
- UI: `/Users/deakshshetty/Documents/ADE-Guard/frontend`

## Reproducibility Notes
- Use the same filtered dataset for train/eval/test (`COVID_ONLY=1`).
- Keep tokenizer + label mappings with saved model checkpoints.
- Record exact model paths and commit hashes used for evaluation.
