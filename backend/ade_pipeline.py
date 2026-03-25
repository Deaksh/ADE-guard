# backend/ade_pipeline.py
from backend.ner_utils import extract_entities
from backend.severity_utils import classify_severity

ADE_LABELS = {"ADE", "ADR", "ADVERSE_EVENT"}
SEVERE_MARKERS = {"anaphylaxis", "stroke", "seizure", "myocarditis", "cardiac arrest", "respiratory failure", "death"}

def extract_ade_with_severity(text: str):
    entities = extract_entities(text)
    results = []

    for ent in entities:
        label_base = ent["label"].upper().replace("B-", "").replace("I-", "")
        if label_base in ADE_LABELS:
            ent_text = ent["text"]
            if ent_text.lower() in SEVERE_MARKERS:
                severity = ("Severe", 1.0)
            else:
                out = classify_severity(ent_text)
                severity = (out.get("label", "Unknown"), float(out.get("confidence", 0.0)))
            results.append({
                "entity": ent_text,
                "label": label_base,
                "start": ent["start"],
                "end": ent["end"],
                "severity": severity,
                "score": ent["score"],
            })
    return results
