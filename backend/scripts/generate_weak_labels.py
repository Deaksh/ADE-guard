# backend/scripts/generate_weak_labels.py
import json
import logging
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adeguard")

# File paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "merged_2025.csv"
OUTPUT_FILE = DATA_DIR / "weak_labels.json"

logger.info(f"Loading dataset from: {INPUT_FILE}")

# Load dataset
df = pd.read_csv(INPUT_FILE)
logger.info(f"Raw dataframe shape: {df.shape}")
logger.info(f"Unique symptoms precomputed: {df['SYMPTOM_TEXT'].nunique()}")

# --- Weak labeling rules (simplified for demo) ---
def weak_label_severity(text: str) -> str:
    text = str(text).lower()
    if any(w in text for w in ["death", "seizure", "stroke", "hospitalized", "life-threatening"]):
        return "Severe"
    elif any(w in text for w in ["rash", "fever", "swelling", "vomiting", "dizziness"]):
        return "Moderate"
    elif any(w in text for w in ["pain", "redness", "headache", "tired", "sore"]):
        return "Mild"
    return "Unknown"

# Apply weak labels
weak_labels = []
for _, row in df.iterrows():
    text = row.get("SYMPTOM_TEXT", "")
    vid = row.get("VAERS_ID", None)

    label = {
        "VAERS_ID": vid,
        "Text": text,
        "WeakSeverity": weak_label_severity(text),
    }
    weak_labels.append(label)

# Wrap output
weak_output = {
    "num_samples": len(weak_labels),
    "weak_labels": weak_labels[:500]  # limit to 500 for JSON sanity (remove limit if you want all)
}

# Ensure directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Save to JSON
OUTPUT_FILE.write_text(json.dumps(weak_output, indent=2))
logger.info(f"Weak labels saved to: {OUTPUT_FILE}")
