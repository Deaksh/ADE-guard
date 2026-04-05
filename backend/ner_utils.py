import os
from functools import lru_cache
from typing import List, Dict, Any
import requests
from dotenv import load_dotenv

# Load once at module level for performance
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
DEFAULT_DIR = os.path.join(BASE_DIR, "models", "ner_biobert_output")
ENV_PATH = os.environ.get("NER_MODEL_PATH")
USE_HF_INFERENCE = os.environ.get("USE_HF_INFERENCE", "0") == "1" or \
    os.environ.get("NER_INFERENCE", "").lower() in {"hf", "1", "true"}
HF_API_TOKEN = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
HF_API_TIMEOUT = int(os.environ.get("HF_API_TIMEOUT", "20"))


def _resolve_model_path() -> str:
    candidates = []
    if ENV_PATH:
        candidates.append(ENV_PATH)
    candidates.extend([
        DEFAULT_DIR,
        os.path.join(DEFAULT_DIR, "checkpoint-400"),
        os.path.join(DEFAULT_DIR, "checkpoint-200"),
    ])
    for path in candidates:
        # If user provided a HF repo id, use it directly
        if path and not os.path.exists(path) and isinstance(path, str) and "/" in path and not path.startswith("/"):
            return path
        if not path or not os.path.exists(path):
            continue
        # Ensure a model file exists in the directory
        has_model = any(
            os.path.exists(os.path.join(path, fname))
            for fname in ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]
        )
        if has_model:
            return path
    return DEFAULT_DIR


MODEL_PATH = _resolve_model_path()


@lru_cache(maxsize=1)
def _get_pipeline():
    from transformers import pipeline
    return pipeline(
        "ner",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        aggregation_strategy="simple",
    )


def _hf_ner(text: str) -> List[Dict[str, Any]]:
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is required for Hugging Face inference.")
    url = f"https://api-inference.huggingface.co/models/{MODEL_PATH}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    resp = requests.post(url, headers=headers, json=payload, timeout=HF_API_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data["error"])
    if not isinstance(data, list):
        return []
    entities = []
    stopwords = {
        "and", "or", "the", "a", "an", "to", "of", "in", "after", "for", "on", "at", "with",
        "without", "from", "by", "as", "is", "was", "were", "be", "been", "being",
        "developed", "received", "took", "taking", "reported", "patient", "pt"
    }
    for r in data:
        word = r.get("word") or r.get("entity") or ""
        word = word.replace("##", "") if word.startswith("##") else word
        cleaned = word.strip()
        if len(cleaned) < 3 or cleaned.lower() in stopwords:
            continue
        entities.append({
            "text": cleaned,
            "label": r.get("entity_group") or r.get("entity") or "ADE",
            "start": r.get("start", 0),
            "end": r.get("end", 0),
            "score": float(r.get("score", 0.0)),
        })
    return entities


@lru_cache(maxsize=1024)
def _extract_entities_cached(text: str):
    if USE_HF_INFERENCE:
        return _hf_ner(text)
    ner_pipeline = _get_pipeline()
    # Older pipeline versions don't accept truncation params; do manual truncation.
    tokens = ner_pipeline.tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = tokens["input_ids"][0]
    truncated_text = ner_pipeline.tokenizer.decode(input_ids, skip_special_tokens=True)
    results = ner_pipeline(truncated_text)
    entities = []
    stopwords = {"and", "or", "the", "a", "an", "to", "of", "in", "after", "for", "on", "at", "with",
                 "without", "from", "by", "as", "is", "was", "were", "be", "been", "being",
                 "developed", "received", "took", "taking", "reported", "patient", "pt"}
    for r in results:
        word = r["word"].replace("##", "") if r["word"].startswith("##") else r["word"]
        cleaned = word.strip()
        if len(cleaned) < 3 or cleaned.lower() in stopwords:
            continue
        entities.append({
            "text": cleaned,
            "label": r["entity_group"],
            "start": r["start"],
            "end": r["end"],
            "score": float(r["score"]),
        })
    return entities


def extract_entities(text: str):
    return _extract_entities_cached(text)
