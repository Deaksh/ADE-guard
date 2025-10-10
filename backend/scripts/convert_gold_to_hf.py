# backend/scripts/convert_gold_to_hf.py
import json
from pathlib import Path
IN = Path("/Users/deakshshetty/Documents/ADE-Guard/backend/data/gold_data.json")
OUT = Path("/Users/deakshshetty/Documents/ADE-Guard/backend/data/ner_dataset.jsonl")
data = json.load(open(IN))
out = []
for doc in data:
    text = doc["text"]
    entities = []
    for e in doc.get("entities", []):
        entities.append({"start": e["start"], "end": e["end"], "label": e["label"]})
    out.append({"id": doc.get("VAERS_ID"), "text": text, "entities": entities})
with OUT.open("w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("Wrote", OUT, len(out))
