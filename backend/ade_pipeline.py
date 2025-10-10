# backend/ade_pipeline.py
from backend.ner_utils import extract_entities
from backend.severity_utils import classify_severity_local

ADE_LABELS = {"ADE", "ADR", "ADVERSE_EVENT"}

def extract_ade_with_severity(text: str):
    entities = extract_entities(text)
    results = []

    for ent in entities:
        label_base = ent["label"].upper().replace("B-", "").replace("I-", "")
        if label_base in ADE_LABELS:
            severity = classify_severity_local(ent["text"])
            results.append({
                "entity": ent["text"],
                "label": label_base,
                "start": ent["start"],
                "end": ent["end"],
                "severity": severity,
                "score": ent["score"],
            })
    return results
