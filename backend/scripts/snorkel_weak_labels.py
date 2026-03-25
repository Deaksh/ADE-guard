import json
import os
from pathlib import Path
import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

# Labels
ABSTAIN = -1
MILD = 0
MODERATE = 1
SEVERE = 2

LABELS = {MILD: "Mild", MODERATE: "Moderate", SEVERE: "Severe"}

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
COVID_ONLY = os.environ.get("COVID_ONLY", "0") == "1"
INPUT_FILE = DATA_DIR / ("merged_covid_2025.csv" if COVID_ONLY else "merged_2025.csv")
OUTPUT_FILE = DATA_DIR / ("weak_labels_snorkel_covid.json" if COVID_ONLY else "weak_labels_snorkel.json")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

# --- Labeling functions ---
@labeling_function()
def lf_severe_keywords(x):
    text = str(x.get("SYMPTOM_TEXT", "")).lower()
    if any(w in text for w in ["death", "fatal", "life-threatening", "anaphylaxis", "stroke", "seizure", "myocarditis"]):
        return SEVERE
    return ABSTAIN

@labeling_function()
def lf_hospitalized(x):
    hosp = str(x.get("HOSPITAL", "")).strip().lower()
    if hosp in {"y", "yes", "1", "true"}:
        return SEVERE
    return ABSTAIN

@labeling_function()
def lf_died(x):
    died = str(x.get("DIED", "")).strip().lower()
    if died in {"y", "yes", "1", "true"}:
        return SEVERE
    return ABSTAIN

@labeling_function()
def lf_moderate_keywords(x):
    text = str(x.get("SYMPTOM_TEXT", "")).lower()
    if any(w in text for w in ["fever", "vomiting", "dizziness", "rash", "swelling", "fainting"]):
        return MODERATE
    return ABSTAIN

@labeling_function()
def lf_mild_keywords(x):
    text = str(x.get("SYMPTOM_TEXT", "")).lower()
    if any(w in text for w in ["pain", "soreness", "redness", "headache", "fatigue", "tired"]):
        return MILD
    return ABSTAIN

lfs = [
    lf_severe_keywords,
    lf_hospitalized,
    lf_died,
    lf_moderate_keywords,
    lf_mild_keywords,
]

applier = PandasLFApplier(lfs=lfs)
L = applier.apply(df=df)

analysis = LFAnalysis(L=L, lfs=lfs).lf_summary()
print(analysis)

label_model = LabelModel(cardinality=3, verbose=True)
label_model.fit(L_train=L, n_epochs=200, log_freq=50, seed=42)

probs = label_model.predict_proba(L)
labels = probs.argmax(axis=1)

weak_labels = []
for idx, row in df.iterrows():
    weak_labels.append({
        "VAERS_ID": row.get("VAERS_ID"),
        "Text": row.get("SYMPTOM_TEXT", ""),
        "WeakSeverity": LABELS.get(int(labels[idx]), "Mild"),
        "Probabilities": {
            "Mild": float(probs[idx][MILD]),
            "Moderate": float(probs[idx][MODERATE]),
            "Severe": float(probs[idx][SEVERE]),
        },
    })

output = {
    "num_samples": len(weak_labels),
    "weak_labels": weak_labels,
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"Saved Snorkel weak labels -> {OUTPUT_FILE}")
