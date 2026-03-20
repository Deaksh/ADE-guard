# backend/train_ner.py

import sys
import os
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from seqeval.metrics import precision_score, recall_score, f1_score
print(sys.executable)
import transformers
print(transformers.__file__)
print(transformers.__version__)

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import torch

BASE = Path(__file__).resolve().parent
COVID_ONLY = os.environ.get("COVID_ONLY", "0") == "1"
GOLD_PATH = BASE / "data" / ("gold_data_covid.json" if COVID_ONLY else "gold_data.json")
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
OUT_DIR = BASE / "models" / "ner_biobert"
os.makedirs(OUT_DIR, exist_ok=True)
MAX_LEN = 512

# Optional: restrict labels for evaluation alignment (e.g., "ADE,DRUG")
LABEL_FILTER = os.environ.get("NER_LABELS", "")
ALLOWED_LABELS = {l.strip().upper() for l in LABEL_FILTER.split(",") if l.strip()} if LABEL_FILTER else None

# Load gold data
with open(GOLD_PATH, "r", encoding="utf-8") as f:
    gold = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def token_label_mapping(text, entities):
    # Tokenize with truncation and padding to max_length for consistent tensor sizes
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_offsets_mapping=True,
        return_attention_mask=True
    )
    offsets = encoding["offset_mapping"]

    # Create label list aligned with tokens, default 'O'
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

    # Pad or truncate labels to MAX_LEN to match tokenizer padding/truncation
    labels = labels[:MAX_LEN] + ["O"] * max(0, MAX_LEN - len(labels))

    encoding["labels"] = labels
    return encoding

records = []
for doc in gold:
    rec = token_label_mapping(doc["text"], doc.get("entities", []))
    rec["id"] = doc.get("VAERS_ID")
    records.append(rec)

ds = Dataset.from_list(records)

# Prepare label mappings
unique_labels = sorted(set(label for labels in ds['labels'] for label in labels))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Map string labels to integer label_ids
def labels_to_ids(example):
    example["labels"] = [label2id[label] for label in example["labels"]]  # override string labels with int labels
    return example

ds = ds.map(labels_to_ids)

# Split dataset into train/eval
ds = ds.train_test_split(test_size=0.2, seed=42)
train_ds = ds["train"]
eval_ds = ds["test"]

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
)

# Data collator that dynamically pads to batch's max length
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    do_train=True,
    do_eval=True,
    logging_dir=str(BASE / "logs"),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",   # Correct argument
    load_best_model_at_end=True,
    push_to_hub=False,
)

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[l] for l in label_row if l != -100]
        for label_row in labels
    ]
    pred_labels = []
    for i, row in enumerate(preds):
        curr_pred = []
        for j, pred_id in enumerate(row):
            if labels[i][j] == -100:
                continue
            curr_pred.append(id2label[pred_id])
        pred_labels.append(curr_pred)

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))
print("Saved NER model ->", OUT_DIR)
