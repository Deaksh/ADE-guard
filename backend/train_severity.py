# backend/train_severity.py

import json
import os
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

BASE = Path(__file__).resolve().parent
WEAK_PATH = BASE / "data" / "weak_labels.json"
MODEL_NAME = "roberta-base"
OUT_DIR = BASE / "models" / "severity_roberta"
os.makedirs(OUT_DIR, exist_ok=True)

# Load raw data
with open(WEAK_PATH, "r", encoding="utf-8") as f:
    weak = json.load(f)

rows = []
for e in weak.get("weak_labels", []):
    text = e.get("Text") or e.get("text") or ""
    label = e.get("WeakSeverity") or "Mild"
    if not isinstance(text, str) or not text.strip():
        continue
    if not isinstance(label, str):
        label = "Mild"
    rows.append({"text": text, "label": label})

labels = sorted(list({r["label"] for r in rows}))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

ds = Dataset.from_list(rows)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    # Reduced max_length from 256 to 128 for faster training
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

ds = ds.map(tokenize_fn, batched=True)

def encode_labels(example):
    example["labels"] = label2id[example["label"]]
    return example

ds = ds.map(encode_labels)
ds = ds.remove_columns("label")

ds = ds.train_test_split(test_size=0.1, seed=42)
train_ds = ds["train"]
eval_ds = ds["test"]

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,  # Reduced epochs from 3 to 2
    per_device_train_batch_size=4,  # Reduced batch size from 8 to 4
    per_device_eval_batch_size=8,   # Reduced eval batch size from 16 to 8
    logging_dir=str(BASE / "logs_severity"),
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision if supported on Mac MPS
)

def compute_metrics(eval_pred):
    logits, labels_true = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import classification_report
    report = classification_report(labels_true, preds, target_names=labels, output_dict=True)
    return {"classification_report": report}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))
print("Saved severity model ->", OUT_DIR)
