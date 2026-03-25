import os
import torch
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCAL_MODEL = os.path.join(BASE, "models", "severity_biobert")
MODEL_ID = os.environ.get("SEVERITY_MODEL") or DEFAULT_LOCAL_MODEL
USE_HF_INFERENCE = os.environ.get("USE_HF_INFERENCE", "0") == "1" or \
    os.environ.get("SEVERITY_INFERENCE", "").lower() in {"hf", "1", "true"}
HF_API_TOKEN = os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Ensure HF cache is writable when pulling from hub
DEFAULT_CACHE = os.path.join(BASE, ".hf_cache")
os.environ["HF_HOME"] = os.environ.get("HF_HOME") or DEFAULT_CACHE
cache_dir = os.environ["HF_HOME"]
try:
    os.makedirs(cache_dir, exist_ok=True)
except OSError:
    cache_dir = DEFAULT_CACHE
    os.environ["HF_HOME"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

_classifier = None


def load_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    if os.path.exists(MODEL_ID):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        device = 0 if torch.cuda.is_available() else -1
        _classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, top_k=None)
        print("Loaded labels:", _classifier.model.config.id2label)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, cache_dir=cache_dir)
            device = 0 if torch.cuda.is_available() else -1
            _classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, top_k=None)
            print("Loaded labels:", _classifier.model.config.id2label)
        except Exception as e:
            print(f"Classifier not loaded: {e}")
            _classifier = None
    return _classifier


def classify_severity_local(text: str):
    clf = load_classifier()
    if clf is None:
        print("Classifier not loaded.")
        return None
    out = clf(text, top_k=None)
    best_pred = max(out, key=lambda x: x['score'])
    label = best_pred['label']
    score = best_pred['score']
    return label, score


def classify_severity_remote(text: str):
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is required for Hugging Face inference.")
    url = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data["error"])
    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, list):
        return {"label": "Unknown", "confidence": 0.0, "probabilities": {}}
    probs = {d["label"]: float(d["score"]) for d in data if "label" in d and "score" in d}
    if not probs:
        return {"label": "Unknown", "confidence": 0.0, "probabilities": {}}
    label = max(probs.items(), key=lambda x: x[1])[0]
    return {"label": label, "confidence": probs[label], "probabilities": probs}


def classify_severity(text: str):
    if USE_HF_INFERENCE:
        return classify_severity_remote(text)
    clf = load_classifier()
    if clf is None:
        return {"label": "Unknown", "confidence": 0.0, "probabilities": {}}
    out = clf(text, top_k=None)
    best_pred = max(out, key=lambda x: x["score"])
    probs = {r["label"]: float(r["score"]) for r in out}
    return {"label": best_pred["label"], "confidence": float(best_pred["score"]), "probabilities": probs}


if __name__ == "__main__":
    text = "The patient has severe pain."
    label, score = classify_severity_local(text)
    print(f"Predicted severity: {label} (confidence: {score:.2f})")
