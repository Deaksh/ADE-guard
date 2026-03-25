import json
import os
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch

BASE = Path(__file__).resolve().parent

COVID_ONLY = os.environ.get("COVID_ONLY", "0") == "1"
USE_SNORKEL = os.environ.get("USE_SNORKEL", "0") == "1"

if USE_SNORKEL:
    WEAK_PATH = BASE / "data" / ("weak_labels_snorkel_covid.json" if COVID_ONLY else "weak_labels_snorkel.json")
else:
    WEAK_PATH = BASE / "data" / ("weak_labels_covid.json" if COVID_ONLY else "weak_labels.json")

MODEL_NAME = os.environ.get("SEVERITY_MODEL", "dmis-lab/biobert-base-cased-v1.2")
OUT_DIR = BASE / "models" / "severity_biobert"
os.makedirs(OUT_DIR, exist_ok=True)

# Ensure HF cache is writable
DEFAULT_CACHE = BASE / ".hf_cache"
os.environ["HF_HOME"] = os.environ.get("HF_HOME") or str(DEFAULT_CACHE)
cache_dir = os.environ["HF_HOME"]
try:
    os.makedirs(cache_dir, exist_ok=True)
except OSError:
    cache_dir = str(DEFAULT_CACHE)
    os.environ["HF_HOME"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

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

# Compute class weights (inverse frequency)
label_counts = {l: 0 for l in labels}
for r in rows:
    label_counts[r["label"]] += 1

counts = np.array([label_counts[l] for l in labels], dtype=np.float32)
weights = counts.sum() / (counts + 1e-6)
weights = weights / weights.mean()
class_weights = torch.tensor(weights, dtype=torch.float32)
print("Class weights:", {l: float(w) for l, w in zip(labels, weights)})

ds = Dataset.from_list(rows)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)

def tokenize_fn(examples):
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
    label2id=label2id,
    cache_dir=cache_dir,
)

# Freeze embeddings + early layers, train only last 4 + classifier
for param in model.base_model.parameters():
    param.requires_grad = False

if hasattr(model.base_model, "encoder"):
    encoder_layers = model.base_model.encoder.layer
    for layer in encoder_layers[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    logging_dir=str(BASE / "logs_severity"),
    load_best_model_at_end=True,
    fp16=False,
)


def compute_metrics(eval_pred):
    logits, labels_true = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import classification_report
    report = classification_report(labels_true, preds, target_names=labels, output_dict=True)
    return {"classification_report": report}


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels_tensor = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels_tensor)
        return (loss, outputs) if return_outputs else loss


trainer = WeightedTrainer(
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
