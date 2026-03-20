import json
import os
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"

GOLD_IN = DATA_DIR / "gold_data.json"
MERGED_COVID = DATA_DIR / "merged_covid_2025.csv"
GOLD_OUT = DATA_DIR / "gold_data_covid.json"

if not GOLD_IN.exists():
    raise FileNotFoundError(f"Missing gold file: {GOLD_IN}")
if not MERGED_COVID.exists():
    raise FileNotFoundError(f"Missing COVID merged file: {MERGED_COVID}")

covid_df = pd.read_csv(MERGED_COVID, usecols=["VAERS_ID"])
covid_ids = set(covid_df["VAERS_ID"].dropna().astype(str).tolist())

with open(GOLD_IN, "r", encoding="utf-8") as f:
    gold = json.load(f)

filtered = []
for entry in gold:
    vid = entry.get("vaers_id") or entry.get("VAERS_ID")
    if vid is None:
        continue
    if str(vid) in covid_ids:
        filtered.append(entry)

with open(GOLD_OUT, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2)

print(f"Wrote {len(filtered)} records to {GOLD_OUT}")
