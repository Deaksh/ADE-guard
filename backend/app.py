import os
# Must be set before importing numba/umap/shap to avoid threading layer issues
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_DISABLE_JIT", "0")
import re
import json
import logging
from typing import Any, Dict, List
from functools import lru_cache
import threading

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.ade_pipeline import extract_ade_with_severity
from backend.clustering_utils import cluster_ades
from backend.data_preparation import load_and_merge_vaers, load_and_merge_vaers_year
from backend.ner_utils import extract_entities
from backend.severity_utils import load_classifier, classify_severity
from dotenv import load_dotenv
from backend.services.insights_service import build_clinical_insights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adeguard")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
REPO_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
VAERS_DATA_ROOT = os.environ.get("VAERS_DATA_ROOT") or os.path.join(REPO_DIR, "Input_files", "data")
GOLD_PATH = os.path.join(DATA_DIR, "gold_data.json")
GOLD_COVID_PATH = os.path.join(DATA_DIR, "gold_data_covid.json")
WEAK_PATH = os.path.join(DATA_DIR, "weak_labels.json")
MERGED_PATH = os.path.join(DATA_DIR, "merged_2025.csv")
MERGED_COVID_PATH = os.path.join(DATA_DIR, "merged_covid_2025.csv")

app = FastAPI(title="ADEGuard API")
CLUSTER_LOCK = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3100",
        "http://127.0.0.1:3100",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


class SeverityResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]


class ExplainRequest(BaseModel):
    text: str


def _load_gold_data() -> Dict[str, Dict[str, Any]]:
    covid_only = os.environ.get("COVID_ONLY", "0") == "1"
    target = GOLD_COVID_PATH if covid_only else GOLD_PATH
    if not os.path.exists(target):
        logger.warning("Gold data not found at %s", target)
        return {}
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for entry in data:
        vaers_id = entry.get("vaers_id") or entry.get("VAERS_ID")
        if vaers_id is None:
            continue
        out[str(vaers_id)] = entry
    return out


def _load_weak_data() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(WEAK_PATH):
        logger.warning("Weak labels not found at %s", WEAK_PATH)
        return {}
    with open(WEAK_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for entry in raw.get("weak_labels", []):
        vaers_id = entry.get("VAERS_ID") or entry.get("vaers_id")
        if vaers_id is None:
            continue
        out[str(vaers_id)] = entry
    return out


def _load_dataset() -> pd.DataFrame:
    covid_only = os.environ.get("COVID_ONLY", "0") == "1"
    target_path = MERGED_COVID_PATH if covid_only else MERGED_PATH

    if not os.path.exists(target_path):
        if covid_only:
            logger.info("COVID-only dataset missing, building from raw CSVs...")
            load_and_merge_vaers(DATA_DIR, output_name="merged_covid_2025.csv", covid_only=True)
        else:
            logger.info("Merged dataset missing, building from raw CSVs...")
            load_and_merge_vaers(DATA_DIR)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Dataset not found at: {target_path}")

    logger.info("Loading dataset from: %s", target_path)
    df = pd.read_csv(target_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    if "RECVDATE" in df.columns:
        df["RECVDATE_PARSED"] = pd.to_datetime(df["RECVDATE"], errors="coerce")
    else:
        df["RECVDATE_PARSED"] = pd.NaT

    for col in ["ALL_SYMPTOMS", "SYMPTOM_TEXT", "VAX_NAME", "DIED", "HOSPITAL"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    return df


def _extract_symptom_tokens(s: str) -> List[str]:
    if not s:
        return []
    splitter = re.compile(r"\s*\|\s*|;|,")
    return [p.strip() for p in splitter.split(str(s)) if p.strip()]


@lru_cache(maxsize=1)
def get_gold_data():
    return _load_gold_data()


@lru_cache(maxsize=1)
def get_weak_data():
    return _load_weak_data()


@lru_cache(maxsize=8)
def get_df(year: int | None = None):
    if year is None:
        return _load_dataset()
    return _load_dataset_year(year)


def _load_dataset_year(year: int) -> pd.DataFrame:
    covid_only = os.environ.get("COVID_ONLY", "0") == "1"
    output_dir = os.path.join(DATA_DIR, "year_cache")
    os.makedirs(output_dir, exist_ok=True)
    cached_path = os.path.join(output_dir, f"merged_{year}.csv")
    if os.path.exists(cached_path):
        df = pd.read_csv(cached_path, low_memory=False)
    else:
        df = load_and_merge_vaers_year(year, data_root=VAERS_DATA_ROOT, output_dir=output_dir, covid_only=covid_only)

    df.columns = [c.strip() for c in df.columns]
    if "RECVDATE" in df.columns:
        df["RECVDATE_PARSED"] = pd.to_datetime(df["RECVDATE"], errors="coerce")
    else:
        df["RECVDATE_PARSED"] = pd.NaT
    for col in ["ALL_SYMPTOMS", "SYMPTOM_TEXT", "VAX_NAME", "DIED", "HOSPITAL"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    return df


def _available_years() -> List[int]:
    if not os.path.isdir(VAERS_DATA_ROOT):
        return []
    years = []
    for name in os.listdir(VAERS_DATA_ROOT):
        if name.endswith("VAERSData.zip"):
            try:
                years.append(int(name[:4]))
            except Exception:
                continue
    return sorted(set(years))


@lru_cache(maxsize=8)
def get_unique_symptoms(year: int | None = None):
    df = get_df(year)
    unique = set()
    for v in df["ALL_SYMPTOMS"].dropna():
        for token in _extract_symptom_tokens(v):
            unique.add(token)
    return unique


@lru_cache(maxsize=1)
def get_severity_pipeline():
    return load_classifier()


def _predict_proba(texts: List[str]) -> List[List[float]]:
    pipeline = get_severity_pipeline()
    if pipeline is None:
        return []
    outputs = pipeline(texts, top_k=None)
    if isinstance(outputs, dict):
        outputs = [outputs]
    labels = list(pipeline.model.config.id2label.values())
    prob_rows = []
    for row in outputs:
        label_map = {r["label"]: float(r["score"]) for r in row}
        prob_rows.append([label_map.get(l, 0.0) for l in labels])
    return prob_rows


@lru_cache(maxsize=2048)
def _classify_severity_cached(text: str) -> Dict[str, Any]:
    # Prefer remote inference if configured in severity_utils
    result = classify_severity(text)
    label = result.get("label", "Unknown")
    confidence = float(result.get("confidence", 0.0))
    probs = result.get("probabilities", {})
    # Rule-based overrides for critical signals
    text_lower = text.lower()
    severe_markers = [
        "death", "fatal", "life-threatening", "anaphylaxis", "stroke", "seizure",
        "myocarditis", "cardiac arrest", "icu", "intensive care", "respiratory failure"
    ]
    moderate_markers = ["hospitalized", "er visit", "emergency room", "er ", "syncope", "fainting"]
    if any(m in text_lower for m in severe_markers):
        label = "Severe"
        confidence = max(confidence, 0.95)
    elif any(m in text_lower for m in moderate_markers) and label == "Mild":
        label = "Moderate"
        confidence = max(confidence, 0.7)
    return {"label": label, "confidence": confidence, "probabilities": probs}


def _classify_severity(text: str) -> SeverityResponse:
    data = _classify_severity_cached(text)
    return SeverityResponse(**data)


@app.get("/")
def root():
    return {"message": "ADEGuard API is running"}


@app.head("/")
def root_head():
    return Response(status_code=200)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/api/v1")
def api_root():
    return {
        "message": "ADEGuard API v1 is running",
        "endpoints": [
            "/api/v1/summary",
            "/api/v1/trends",
            "/api/v1/alerts",
            "/api/v1/search",
            "/api/v1/report/{vaers_id}",
            "/api/v1/ner",
            "/api/v1/severity",
            "/api/v1/analyze",
            "/api/v1/clusters",
            "/api/v1/explain/severity",
        ],
    }


@app.get("/api/v1/years")
def get_years():
    return {"years": _available_years()}


@app.get("/api/v1/summary")
def get_summary(year: int | None = None):
    df = get_df(year)
    total_reports = int(len(df))
    unique_symptoms_count = int(len(get_unique_symptoms(year)))
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


@app.get("/api/v1/trends")
def get_trends(days: int = 30, year: int | None = None):
    df = get_df(year)
    if "RECVDATE_PARSED" not in df.columns or df["RECVDATE_PARSED"].isna().all():
        return {"dates": [], "report_counts": [], "ade_signals": []}

    df = df.dropna(subset=["RECVDATE_PARSED"]).copy()
    df["date"] = df["RECVDATE_PARSED"].dt.date
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    df = df[df["date"] >= cutoff]

    counts = df.groupby("date").size().reset_index(name="report_counts")

    severe_keywords = re.compile(r"death|life-threatening|hospitalized|stroke|seizure", re.I)
    df["is_severe"] = df["SYMPTOM_TEXT"].astype(str).str.contains(severe_keywords)
    signals = df.groupby("date")["is_severe"].sum().reset_index(name="ade_signals")

    merged = counts.merge(signals, on="date", how="left").fillna(0)
    merged = merged.sort_values("date")

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in merged["date"]],
        "report_counts": merged["report_counts"].astype(int).tolist(),
        "ade_signals": merged["ade_signals"].astype(int).tolist(),
    }


@app.get("/api/v1/alerts")
def get_alerts(limit: int = 5, year: int | None = None):
    df = get_df(year).copy()
    df["symptom_tokens"] = df["ALL_SYMPTOMS"].apply(_extract_symptom_tokens)
    flat = []
    for _, row in df.iterrows():
        for sym in row["symptom_tokens"]:
            flat.append({
                "symptom": sym,
                "vaccine": row.get("VAX_NAME", ""),
                "text": row.get("SYMPTOM_TEXT", ""),
            })
    if not flat:
        return {"alerts": []}

    flat_df = pd.DataFrame(flat)
    severe_keywords = re.compile(r"death|life-threatening|hospitalized|stroke|seizure", re.I)
    flat_df["is_severe"] = flat_df["text"].astype(str).str.contains(severe_keywords)

    grouped = flat_df.groupby(["symptom", "vaccine"]).agg(
        count=("symptom", "size"),
        severe_count=("is_severe", "sum"),
    ).reset_index()

    grouped["signal_strength"] = grouped["severe_count"].apply(
        lambda x: "high" if x >= 5 else "medium" if x >= 2 else "low"
    )

    top = grouped.sort_values(["severe_count", "count"], ascending=False).head(limit)
    alerts = [
        {
            "symptom": row["symptom"],
            "vaccine": row["vaccine"],
            "count": int(row["count"]),
            "signal_strength": row["signal_strength"],
        }
        for _, row in top.iterrows()
    ]
    return {"alerts": alerts}


@app.get("/api/v1/search")
def search(symptom: str, limit: int = 20, year: int | None = None):
    if not symptom:
        raise HTTPException(status_code=400, detail="symptom query is required")
    query = symptom.lower()
    df = get_df(year)
    mask = df["SYMPTOM_TEXT"].astype(str).str.lower().str.contains(query) | \
        df["ALL_SYMPTOMS"].astype(str).str.lower().str.contains(query)
    subset = df[mask].head(limit)
    reports = []
    for _, row in subset.iterrows():
        reports.append({
            "vaers_id": int(row.get("VAERS_ID")) if pd.notna(row.get("VAERS_ID")) else None,
            "date": str(row.get("RECVDATE", "")),
            "vaccine": row.get("VAX_NAME", ""),
            "description": row.get("SYMPTOM_TEXT", ""),
        })
    return {"symptom": symptom, "reports": reports}


@app.get("/api/v1/report/{vaers_id}")
def get_full_report(vaers_id: int, year: int | None = None):
    df = get_df(year)
    row = df.loc[df["VAERS_ID"] == vaers_id]
    if row.empty:
        return {
            "VAERS_ID": vaers_id,
            "text": "Not found",
            "gold_entities": [],
            "weak_severity": None,
            "ner_entities": [],
            "predicted_severity": "Unknown",
        }

    text = row.iloc[0].get("SYMPTOM_TEXT", "No text")
    gold_entry = get_gold_data().get(str(vaers_id), {})
    gold_entities = gold_entry.get("entities", [])
    weak_entry = get_weak_data().get(str(vaers_id), {})
    weak_severity = weak_entry.get("WeakSeverity")

    try:
        ner_entities = extract_entities(text)
    except Exception as e:
        logger.error("NER error for VAERS_ID %s: %s", vaers_id, e)
        ner_entities = []

    predicted = _classify_severity(text)

    return {
        "VAERS_ID": vaers_id,
        "text": text,
        "gold_entities": gold_entities,
        "weak_severity": weak_severity,
        "ner_entities": ner_entities,
        "predicted_severity": predicted.label,
    }


@app.post("/api/v1/ner")
def ner_extract(input: TextInput):
    try:
        return {"entities": extract_entities(input.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/severity", response_model=SeverityResponse)
def severity(input: TextInput):
    return _classify_severity(input.text)


@app.post("/api/v1/analyze")
def analyze(input: TextInput):
    results = extract_ade_with_severity(input.text)
    return {"results": results}



@app.get("/api/v1/insights")
def insights(year: int | None = None):
    return build_clinical_insights(get_df(year))

@app.get("/api/v1/export")
def export_reports(limit: int = 500, year: int | None = None):
    subset = get_df(year).head(limit).copy()
    csv_data = subset.to_csv(index=False)
    return Response(content=csv_data, media_type="text/csv")

@app.get("/api/v1/clusters")
def clusters(max_records: int = 500, min_cluster_size: int = 15, include_points: int = 1, year: int | None = None):
    cache_key = (max_records, min_cluster_size)
    if not hasattr(clusters, "_cache"):
        clusters._cache = {}
    cache = clusters._cache
    if cache_key in cache:
        output = cache[cache_key]
    else:
        with CLUSTER_LOCK:
            # Double-check cache in case another thread filled it
            if cache_key in cache:
                output = cache[cache_key]
            else:
                output = cluster_ades(get_df(year), max_records=max_records, min_cluster_size=min_cluster_size)
                cache[cache_key] = output
    if include_points != 1:
        output.pop("points", None)
        return output
    points = output.get("points", [])
    if len(points) > 1000:
        output["points"] = points[:1000]
    return output


@app.post("/api/v1/explain/severity")
def explain_severity(input: ExplainRequest):
    pipeline = get_severity_pipeline()
    if pipeline is None:
        return {"error": "Severity model not loaded"}

    text = input.text
    pred = _classify_severity(text)

    labels = list(pipeline.model.config.id2label.values())
    pred_index = labels.index(pred.label) if pred.label in labels else 0

    lime_payload = None
    shap_payload = None

    try:
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=labels)
        exp = explainer.explain_instance(
            text,
            lambda xs: np.array(_predict_proba(xs)),
            labels=[pred_index],
            num_features=10,
        )
        lime_weights = exp.as_list(label=pred_index)
        lime_payload = {
            "features": [{"token": t, "weight": float(w)} for t, w in lime_weights]
        }
    except Exception as e:
        lime_payload = {"error": str(e)}

    try:
        # Ensure numba doesn't try to spawn extra threads during SHAP
        os.environ["NUMBA_NUM_THREADS"] = "1"
        os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
        import shap

        use_tokenizer = os.environ.get("SHAP_USE_TOKENIZER", "0") == "1"
        masker = shap.maskers.Text(pipeline.tokenizer if use_tokenizer else None)

        def _predict_fn(texts):
            return np.array(_predict_proba(list(texts)))

        explainer = shap.Explainer(_predict_fn, masker, output_names=labels, algorithm="partition")
        shap_values = explainer([text], max_evals=128)
        values = shap_values.values[0][pred_index]
        tokens = shap_values.data[0]
        pairs = sorted(
            [(t, float(v)) for t, v in zip(tokens, values)],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]
        shap_payload = {"features": [{"token": t, "weight": w} for t, w in pairs]}
    except Exception as e:
        shap_payload = {"error": str(e)}

    return {
        "label": pred.label,
        "confidence": pred.confidence,
        "lime": lime_payload,
        "shap": shap_payload,
    }
