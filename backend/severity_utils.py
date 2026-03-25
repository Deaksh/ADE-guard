import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCAL_MODEL = os.path.join(BASE, "models", "severity_biobert")
MODEL_ID = os.environ.get("SEVERITY_MODEL") or DEFAULT_LOCAL_MODEL

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


if __name__ == "__main__":
    text = "The patient has severe pain."
    label, score = classify_severity_local(text)
    print(f"Predicted severity: {label} (confidence: {score:.2f})")
