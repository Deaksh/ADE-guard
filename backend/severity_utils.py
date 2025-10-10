import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL = os.path.join(BASE, "models", "severity_roberta")

_classifier = None

def load_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    if os.path.exists(LOCAL_MODEL):
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL)
        device = 0 if torch.cuda.is_available() else -1
        _classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, top_k=None)
        print("Loaded labels:", _classifier.model.config.id2label)
    else:
        _classifier = None
    return _classifier

def classify_severity_local(text: str):
    clf = load_classifier()
    if clf is None:
        print("Classifier not loaded.")
        return None
    out = clf(text, top_k=None)
    print("All class scores:", out)
    best_pred = max(out, key=lambda x: x['score'])
    label = best_pred['label']
    score = best_pred['score']
    return label, score

if __name__ == "__main__":
    text = "The patient has severe pain."
    label, score = classify_severity_local(text)
    print(f"Predicted severity: {label} (confidence: {score:.2f})")
