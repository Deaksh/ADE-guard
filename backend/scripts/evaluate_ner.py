import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

BASE = Path(__file__).resolve().parents[1]
COVID_ONLY = os.environ.get("COVID_ONLY", "0") == "1"
GOLD_PATH = BASE / "data" / ("gold_data_covid.json" if COVID_ONLY else "gold_data.json")
MODEL_DIR = Path(os.environ.get("NER_MODEL_PATH", BASE / "models" / "ner_biobert"))

LABEL_FILTER = os.environ.get("NER_LABELS", "")
ALLOWED_LABELS = {l.strip().upper() for l in LABEL_FILTER.split(",") if l.strip()} if LABEL_FILTER else None

MAX_LEN = 512

with open(GOLD_PATH, "r", encoding="utf-8") as f:
    gold = json.load(f)

# If COVID_ONLY, prefer pre-filtered gold file (no extra text-based filtering)

# Load model
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"NER model not found at {MODEL_DIR}")

model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

id2label = model.config.id2label


def encode_labels(text: str, entities: List[Dict[str, object]]):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_offsets_mapping=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    offsets = encoding["offset_mapping"][0].tolist()

    labels = []
    for start, end in offsets:
        label = "O"
        if start is None or end is None:
            labels.append(label)
            continue
        for ent in entities:
            s, e, lab = ent["start"], ent["end"], ent["label"]
            if ALLOWED_LABELS and str(lab).upper() not in ALLOWED_LABELS:
                continue
            if start >= e or end <= s:
                continue
            label = "B-" + lab if start == s else "I-" + lab
            break
        labels.append(label)
    return encoding, labels


def predict_labels(encoding):
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )
    logits = outputs.logits[0]
    pred_ids = torch.argmax(logits, dim=-1).tolist()
    labels = [id2label.get(i, "O") for i in pred_ids]
    return labels

true_all = []
pred_all = []

for entry in gold:
    text = entry.get("text", "")
    entities = entry.get("entities", [])
    encoding, true_labels = encode_labels(text, entities)
    pred_labels = predict_labels(encoding)

    # Trim to non-pad tokens
    attn = encoding["attention_mask"][0].tolist()
    true_seq = [lab for lab, m in zip(true_labels, attn) if m == 1]
    pred_seq = [lab for lab, m in zip(pred_labels, attn) if m == 1]

    true_all.append(true_seq)
    pred_all.append(pred_seq)

# Compute metrics
precision = precision_score(true_all, pred_all)
recall = recall_score(true_all, pred_all)
f1 = f1_score(true_all, pred_all)

print("NER Evaluation")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
