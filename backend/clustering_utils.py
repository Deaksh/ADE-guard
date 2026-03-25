import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import hdbscan
import umap

from .ner_utils import extract_entities

MODEL_NAME = os.environ.get("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(MODEL_NAME)
    return _EMBEDDER

ADE_LABELS = {"ADE", "ADR", "ADVERSE_EVENT"}

MODIFIER_RULES = [
    ("Severe", ["severe", "life-threatening", "critical", "fatal", "death", "anaphylaxis"]),
    ("Moderate", ["moderate", "significant", "persistent", "prolonged"]),
    ("Mild", ["mild", "minor", "slight", "low-grade"]),
]

WINDOW_CHARS = 60

STOPWORDS = {
    "and", "or", "the", "a", "an", "to", "of", "in", "after", "for", "on", "at", "with",
    "without", "from", "by", "as", "is", "was", "were", "be", "been", "being", "patient",
    "pt", "he", "she", "they", "it", "this", "that", "these", "those", "day", "days", "year",
    "years", "month", "months", "dose", "dose", "vaccine", "vaccination", "shot", "received",
}



def get_age_group(age) -> str:
    if age is None or pd.isna(age):
        return "Unknown"
    try:
        age = float(age)
    except Exception:
        return "Unknown"
    if age <= 17:
        return "0-17"
    if age <= 30:
        return "18-30"
    if age <= 50:
        return "31-50"
    return "51+"


def detect_modifier(text: str, start: int, end: int) -> str:
    if not text:
        return "Unknown"
    lo = max(0, start - WINDOW_CHARS)
    hi = min(len(text), end + WINDOW_CHARS)
    context = text[lo:hi].lower()
    for label, keys in MODIFIER_RULES:
        if any(k in context for k in keys):
            return label
    return "Unknown"


def _clean_ade(text: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9\-\s]", "", text.lower()).strip()
    if not t or len(t) < 3:
        return ""
    if t in STOPWORDS:
        return ""
    return t


def _extract_ade_mentions(text: str) -> List[Dict[str, object]]:
    mentions = []
    for ent in extract_entities(text):
        label_base = ent["label"].upper().replace("B-", "").replace("I-", "")
        if label_base in ADE_LABELS:
            cleaned = _clean_ade(ent["text"])
            if not cleaned:
                continue
            ent = dict(ent)
            ent["text"] = cleaned
            mentions.append(ent)
    return mentions


def _build_records(df: pd.DataFrame, max_records: int) -> List[Dict[str, object]]:
    records = []
    for _, row in df.iterrows():
        text = str(row.get("SYMPTOM_TEXT", "")).strip()
        if not text:
            continue
        age_group = get_age_group(row.get("AGE_YRS"))
        mentions = _extract_ade_mentions(text)
        for ent in mentions:
            modifier = detect_modifier(text, ent["start"], ent["end"])
            records.append({
                "ade": ent["text"],
                "age_group": age_group,
                "modifier": modifier,
                "text": text,
            })
            if len(records) >= max_records:
                return records
    return records


def _cluster_group(records: List[Dict[str, object]], min_cluster_size: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    texts = [f"{r['modifier']} {r['ade']}".strip() for r in records]
    if not texts:
        return [], []

    embedder = _get_embedder()
    embeddings = embedder.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=32,
    )

    if len(texts) < max(2, min_cluster_size):
        labels = np.full(len(texts), -1)
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(embeddings)

    reducer = umap.UMAP(n_components=2, random_state=42)
    emb2 = reducer.fit_transform(embeddings)

    clusters = defaultdict(list)
    points = []
    for idx, label in enumerate(labels):
        rec = records[idx]
        clusters[label].append(rec)
        points.append({
            "x": float(emb2[idx, 0]),
            "y": float(emb2[idx, 1]),
            "cluster": int(label),
            "age_group": rec["age_group"],
            "modifier": rec["modifier"],
            "ade": rec["ade"],
        })

    cluster_summaries = []
    for label, recs in clusters.items():
        ade_counts = Counter([r["ade"].lower() for r in recs])
        mod_counts = Counter([r["modifier"] for r in recs])
        top_ades = [a for a, _ in ade_counts.most_common(8)]
        top_mod = mod_counts.most_common(1)[0][0] if mod_counts else "Unknown"
        age_group = recs[0]["age_group"] if recs else "Unknown"
        examples = []
        seen = set()
        for r in recs:
            txt = r.get("text", "").strip()
            if not txt or txt in seen:
                continue
            seen.add(txt)
            examples.append(txt[:160])
            if len(examples) >= 2:
                break
        cluster_summaries.append({
            "cluster_id": int(label),
            "age_group": age_group,
            "modifier": top_mod,
            "symptoms": top_ades,
            "count": len(recs),
            "modifier_counts": dict(mod_counts),
            "top_examples": examples,
        })

    return cluster_summaries, points


def cluster_ades(df: pd.DataFrame, max_records: int = 2000, min_cluster_size: int = 15) -> Dict[str, object]:
    if "AGE_YRS" not in df.columns:
        df = df.copy()
        df["AGE_YRS"] = pd.NA

    records = _build_records(df, max_records=max_records)
    if not records:
        return {"clusters": [], "points": []}

    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["age_group"]].append(rec)

    all_clusters = []
    all_points = []
    for age_group, recs in grouped.items():
        clusters, points = _cluster_group(recs, min_cluster_size=min_cluster_size)
        for c in clusters:
            c["age_group"] = age_group
        for p in points:
            p["age_group"] = age_group
        all_clusters.extend(clusters)
        all_points.extend(points)

    # Sort clusters by size (desc), keep noise (-1) last
    all_clusters.sort(key=lambda c: (c["cluster_id"] == -1, -c["count"]))

    return {"clusters": all_clusters, "points": all_points}
