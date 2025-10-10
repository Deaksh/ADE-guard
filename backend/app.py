# backend/app.py
import os
import re
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI
from backend.ner_utils import extract_entities
from backend.llm_utils import classify_severity
from backend.clustering_utils import cluster_ades  # the utility we wrote
import shap
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel
from backend.ade_pipeline import extract_ade_with_severity
from backend.severity_utils import load_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adeguard")

app = FastAPI(title="ADEGuard API")

# CORS - allow local frontend to call
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # restrict this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import json
import logging

logger = logging.getLogger("adeguard")

GOLD_PATH = "/content/drive/MyDrive/ADE-gUARD/gold_data.json"
WEAK_PATH = "/content/drive/MyDrive/ADE-gUARD/weak_labels.json"

# Load data safely
with open(GOLD_PATH, "r") as f:
    data = json.load(f)

gold_data = {}
weak_data = {}

if os.path.exists(GOLD_PATH):
    with open(GOLD_PATH, "r") as f:
        # Build dictionary keyed by VAERS_ID (case-insensitive)
        gold_data = {}
        for entry in data:
            # Handle both "vaers_id" and "VAERS_ID" cases
            vaers_id = entry.get("vaers_id") or entry.get("VAERS_ID")
            if not vaers_id:
                continue  # Skip any malformed entries
            gold_data[str(vaers_id)] = entry
else:
    logger.warning(f"Gold data not found at {GOLD_PATH}")

if os.path.exists(WEAK_PATH):
    with open(WEAK_PATH, "r") as f:
        for entry in json.load(f).get("weak_labels", []):
            weak_data[entry["VAERS_ID"]] = entry
else:
    logger.warning(f"Weak labels not found at {WEAK_PATH}")

# --- Load dataset once at startup ---
BASE_DIR = "/content/drive/MyDrive/ADE-gUARD"

# Construct full dataset path
DATA_PATH = os.path.join(BASE_DIR,"merged_2025.csv")

if not os.path.exists(DATA_PATH):
    logger.error(f"Dataset not found at: {DATA_PATH}")
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

logger.info(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)
logger.info(f"Raw dataframe shape: {df.shape}")

# Normalize column names (strip)
df.columns = [c.strip() for c in df.columns]
#-------------Severity---
# Data models for request and response
from fastapi import Body, HTTPException
from pydantic import BaseModel
class SeverityRequest(BaseModel):
    text: str

class SeverityResponse(BaseModel):
    label: str
    confidence: float

import time

# Preload classifier once at module level
_clf = load_classifier()

def classify_severity(text: str):
    global _clf
    if _clf is None:
        return "Unknown", 0.0
    try:
        out = _clf(text, top_k=None)
        best_pred = max(out, key=lambda x: x["score"])
        return best_pred["label"], best_pred["score"]
    except Exception as e:
        logger.error(f"Error in classify_severity for text: {text[:50]}...: {e}")
        return "Unknown", 0.0
#-------------Severity---

# Ensure RECVDATE parsed
if "RECVDATE" in df.columns:
    df["RECVDATE_PARSED"] = pd.to_datetime(df["RECVDATE"], errors="coerce")
else:
    df["RECVDATE_PARSED"] = pd.NaT

# Fill NaN for safe operations
df["ALL_SYMPTOMS"] = df.get("ALL_SYMPTOMS", "").fillna("")
df["SYMPTOM_TEXT"] = df.get("SYMPTOM_TEXT", "").fillna("")
df["VAX_NAME"] = df.get("VAX_NAME", "").fillna("")
df["DIED"] = df.get("DIED", "").fillna("")
df["HOSPITAL"] = df.get("HOSPITAL", "").fillna("")


# Utility: parse ALL_SYMPTOMS into a set (split on | , ;)
_sym_splitter = re.compile(r"\s*\|\s*|;|,")
def extract_symptom_tokens(s: str):
    tokens = []
    if not s:
        return tokens
    parts = _sym_splitter.split(str(s))
    for p in parts:
        p = p.strip()
        if p:
            tokens.append(p)
    return tokens

# Precompute unique symptom set (fast)
_unique_symptoms = set()
for v in df["ALL_SYMPTOMS"].dropna():
    for token in extract_symptom_tokens(v):
        _unique_symptoms.add(token)
logger.info(f"Unique symptoms precomputed: {len(_unique_symptoms)}")

# ---------------------------
# API v1 endpoints (frontend expects /api/v1/...)
# ---------------------------
@app.get("/")
def root():
    return {"message": "ADEGuard API is running 🚀"}

@app.get("/api/v1")
def api_root():
    return {
        "message": "ADEGuard API v1 is running 🚀",
        "endpoints": ["/api/v1/summary", "/api/v1/trends", "/api/v1/alerts", "/api/v1/search"]
    }

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze(input: TextInput):
    results = extract_ade_with_severity(input.text)
    return {"results": results}

@app.get("/api/v1/summary")
def get_summary():
    total_reports = int(len(df))
    unique_symptoms_count = int(len(_unique_symptoms))
    vaccines_tracked = int(df["VAX_NAME"].nunique()) if "VAX_NAME" in df.columns else 0
    latest_date = None
    if not df["RECVDATE_PARSED"].isna().all():
        latest_date = df["RECVDATE_PARSED"].max().strftime("%Y-%m-%d")
    return {
        "total_reports": total_reports,
        "unique_symptoms": unique_symptoms_count,
        "vaccines_tracked": vaccines_tracked,
        "latest_report_date": latest_date or "N/A",
    }

@app.get("/api/v1/report/{vaers_id}")
def get_full_report(vaers_id: int):
    row = df.loc[df["VAERS_ID"] == vaers_id]
    if row.empty:
        return {
            "VAERS_ID": vaers_id, "text": "Not found", "gold_entities": [],
            "weak_severity": None, "ner_entities": [], "predicted_severity": "Unknown"
        }

    text = row.iloc[0].get("SYMPTOM_TEXT", "No text")
    gold_entry = gold_data.get(vaers_id, {})
    gold_entities = gold_entry.get("entities", [])
    weak_entry = weak_data.get(vaers_id, {})
    weak_severity = weak_entry.get("WeakSeverity")

    try:
        ner_entities = extract_entities(text)
    except Exception as e:
        logger.error(f"NER error for VAERS_ID {vaers_id}: {e}")
        ner_entities = []

    try:
        predicted_severity = classify_severity(text)
    except Exception as e:
        logger.error(f"Severity error for VAERS_ID {vaers_id}: {e}")
        predicted_severity = "Unknown"

    return {
        "VAERS_ID": vaers_id,
        "text": text,
        "gold_entities": gold_entities,
        "weak_severity": weak_severity,
        "ner_entities": ner_entities,
        "predicted_severity": predicted_severity
    }


@app.get("/api/v1/trends")
def get_trends(days: int = Query(30, description="Number of days to return (most recent)")):
    """
    Returns:
      {
        "dates": ["2025-01-01", ...],
        "report_counts": [120, ...],
        "ade_signals": [3, ...]   -- here ADE signals = count of severe (DIED or HOSPITAL)
      }
    """
    if df["RECVDATE_PARSED"].isna().all():
        return {"dates": [], "report_counts": [], "ade_signals": []}

    # Group by date (day-level)
    per_day = (
        df.groupby(df["RECVDATE_PARSED"].dt.date)
        .agg(
            reports=("VAERS_ID", "count"),
            died_count=("DIED", lambda s: (s.astype(str) == "Y").sum()),
            hosp_count=("HOSPITAL", lambda s: (s.astype(str) == "Y").sum()),
        )
        .reset_index()
        .rename(columns={"RECVDATE_PARSED": "date"})
    )

    # Keep only the last `days` rows
    per_day = per_day.tail(days)

    # Create lists for frontend
    date_list = [str(d) for d in per_day["date"]]
    report_counts = per_day["reports"].astype(int).tolist()
    ade_signals = ((per_day["died_count"] + per_day["hosp_count"]).astype(int)).tolist()

    return {
        "dates": date_list,
        "report_counts": report_counts,
        "ade_signals": ade_signals
    }


@app.get("/api/v1/alerts")
def get_alerts(top_n: int = Query(10, description="Top N symptom-vaccine pairs")):
    """
    Returns {"alerts": [ {symptom, vaccine, count, signal_strength}, ... ]}
    """
    if "ALL_SYMPTOMS" not in df.columns or "VAX_NAME" not in df.columns:
        return {"alerts": []}

    # Explode the ALL_SYMPTOMS into rows, then group by symptom and VAX_NAME
    exploded_rows = []
    for _, row in df[["ALL_SYMPTOMS", "VAX_NAME"]].dropna(how="all").iterrows():
        all_sym = row["ALL_SYMPTOMS"]
        vax = row["VAX_NAME"]
        if pd.isna(all_sym) or not all_sym:
            continue
        for token in extract_symptom_tokens(all_sym):
            exploded_rows.append({"SYMPTOM": token, "VAX_NAME": vax})

    if not exploded_rows:
        return {"alerts": []}

    exploded_df = pd.DataFrame(exploded_rows)

    # Clean missing / invalid values
    exploded_df["SYMPTOM"] = (
        exploded_df["SYMPTOM"]
        .fillna("Unknown")
        .astype(str)  # ensure string
        .replace("nan", "Unknown")
    )

    exploded_df["VAX_NAME"] = (
        exploded_df["VAX_NAME"]
        .fillna("Unknown")
        .astype(str)
        .replace("nan", "Unknown")
    )

    # Filter out numeric "SYMPTOM" values (like 27.1, 28.0)
    exploded_df["SYMPTOM"] = exploded_df["SYMPTOM"].apply(
        lambda x: x if not x.replace(".", "", 1).isdigit() else "Unknown"
    )

    grouped = (
        exploded_df.groupby(["SYMPTOM", "VAX_NAME"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )

    alerts = []
    for _, r in grouped.iterrows():
        cnt = int(r["count"])
        if cnt > 50:
            sig = "high"
        elif cnt > 15:
            sig = "medium"
        else:
            sig = "low"
        alerts.append({
            "symptom": r["SYMPTOM"],
            "vaccine": r["VAX_NAME"],
            "count": cnt,
            "signal_strength": sig
        })

    return {"alerts": alerts}

import time

@app.get("/api/v1/ai_insights")
def get_ai_insights(sample_size: int = 5):
    start = time.time()
    insights = []
    if df.empty:
        return {"insights": []}

    sample = df.sample(min(sample_size, len(df)), random_state=42)

    for _, row in sample.iterrows():
        try:
            vaers_id = int(row.get("VAERS_ID", -1))
            text = str(row.get("SYMPTOM_TEXT", "")).strip() or "No symptom text available"
            severity_label, severity_confidence = classify_severity(text)
            insights.append({
                "VAERS_ID": vaers_id,
                "Text": text,
                "Severity": severity_label,
                "Confidence": severity_confidence
            })
        except Exception as ex:
            logger.error(f"Skipping row due to error: {ex}")
            continue
    logger.info(f"Processed {len(insights)} AI insights in {time.time()-start:.2f} seconds")
    return {"insights": insights}

@app.get("/api/v1/search")
def search_symptom(symptom: str = Query(..., description="symptom to search (case-insensitive)"), limit: int = 50):
    """Return sample reports matching the symptom token (searches ALL_SYMPTOMS and SYMPTOM_TEXT)."""
    symptom_lower = symptom.strip().lower()
    mask = df["ALL_SYMPTOMS"].fillna("").str.lower().str.contains(symptom_lower) | df["SYMPTOM_TEXT"].fillna("").str.lower().str.contains(symptom_lower)
    matches = df[mask].head(limit)
    results = []
    for _, row in matches.iterrows():
        results.append({
            "vaers_id": int(row["VAERS_ID"]),
            "date": str(row["RECVDATE"]),
            "vaccine": row.get("VAX_NAME", ""),
            "description": row.get("SYMPTOM_TEXT", "")
        })
    return {"symptom": symptom, "reports": results}

# ---------------------------
# routes/ endpoints used by AI UI
# ---------------------------
@app.get("/routes/data")
def get_sample_data(limit: int = 10):
    return df.head(limit).to_dict(orient="records")


@app.get("/routes/ner/{vaers_id}")
def get_ner_results(vaers_id: int):
    row = df.loc[df["VAERS_ID"] == vaers_id]
    if row.empty:
        return {"VAERS_ID": vaers_id, "text": "Not found", "entities": []}
    text = row.iloc[0].get("SYMPTOM_TEXT", "") or "No text"
    entities = extract_entities(text)
    return {"VAERS_ID": vaers_id, "text": text, "entities": entities}

@app.get("/api/v1/sample_vaers_ids")
def get_sample_vaers_ids(sample_size: int = 5):
    if df.empty:
        return {"vaers_ids": []}
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    vaers_ids = sample["VAERS_ID"].tolist()
    return {"vaers_ids": vaers_ids}


@app.get("/routes/severity/{vaers_id}")
def get_severity(vaers_id: int):
    row = df.loc[df["VAERS_ID"] == vaers_id]
    if row.empty:
        return {"VAERS_ID": vaers_id, "severity": "Unknown", "probabilities": {}}
    text = row.iloc[0].get("SYMPTOM_TEXT", "")
    severity = classify_severity(text, vaers_id)
    return {"VAERS_ID": vaers_id, "severity": severity, "probabilities": {severity: 1.0}}


@app.get("/routes/cluster")
def get_clusters(limit: int = 50):
    logger.info(f"Cluster request received with limit={limit}")
    try:
        clusters = cluster_ades(df, max_clusters=limit)
        logger.info(f"Generated {len(clusters)} clusters")
        return {"clusters": clusters}
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        # Fallback stub data for frontend
        fallback = [
            {"cluster_id": 1, "age_group": "18-30", "modifier": "Unknown", "symptoms": ["fever", "chills", "temperature rise"]},
            {"cluster_id": 2, "age_group": "31-50", "modifier": "Unknown", "symptoms": ["headache", "migraine", "pain in head"]}
        ]
        logger.info("Returning fallback clusters")
        return {"clusters": fallback}


# API endpoint
@app.post("/api/v1/severity_classify", response_model=SeverityResponse)
def severity_classify_api(request: SeverityRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    label, confidence = classify_severity(text)
    if label is None:
        label = "Unknown"
    return SeverityResponse(label=label, confidence=confidence)

@app.post("/api/v1/analyze")
def analyze_text(input: TextInput):
    """
    Analyze text using BioBERT NER and severity classifier.
    Returns entities with severity per ADE.
    """
    try:
        results = extract_ade_with_severity(input.text)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# SHAP explainability endpoint
# ---------------------------

# Lazy load model + tokenizer to avoid startup slowdown
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        try:
            logger.info("Loading BioBERT for SHAP explainability...")
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        except Exception as e:
            logger.error(f"Could not load BioBERT: {e}")
            raise HTTPException(status_code=500, detail="BioBERT model load failed")
    return _tokenizer, _model


@app.post("/api/v1/explain_shap")
def explain_shap(payload: Dict[str, Any]):
    """
    Returns token-level SHAP values for a given text.
    Example request:
    {
        "text": "Patient experienced mild fever after Pfizer dose."
    }
    """
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in request")

    tokenizer, model = load_model()

    def f(x):
        tokens = tokenizer(list(x), return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**tokens)
        return torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

    try:
        explainer = shap.Explainer(f, tokenizer)
        shap_values = explainer([text])
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        raise HTTPException(status_code=500, detail="SHAP explanation failed")

    return {
        "tokens": shap_values.data[0].tolist(),
        "shap_values": shap_values.values[0].tolist()
    }

