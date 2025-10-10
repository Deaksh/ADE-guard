from fastapi import APIRouter
import logging
import pandas as pd

from backend.app import gold_data, weak_data, classify_severity
from backend.ner_utils import extract_entities

router = APIRouter()
logger = logging.getLogger("adeguard")

df = pd.read_csv("/Users/deakshshetty/Documents/ADE-Guard/backend/data/merged_2025.csv")

@router.get("/api/v1/report/{vaers_id}")
def get_full_report(vaers_id: int):
    try:
        row = df.loc[df["VAERS_ID"] == vaers_id]
        if row.empty:
            return {
                "VAERS_ID": vaers_id, "text": "Not found", "gold_entities": [],
                "weak_severity": None, "ner_entities": [], "predicted_severity": "Unknown"
            }
        text = row.iloc[0].get("SYMPTOM_TEXT", "No text")

        gold_entry = gold_data.get(vaers_id, {})
        gold_entities = gold_entry.get("entities", [])
        weak_severity = weak_data.get(vaers_id, {}).get("WeakSeverity")

        try:
            ner_entities = extract_entities(text)
        except Exception as e:
            logger.error(f"NER extraction failed for VAERS_ID {vaers_id}: {e}")
            ner_entities = []

        try:
            predicted_severity = classify_severity(text)
        except Exception as e:
            logger.error(f"Severity classification failed for VAERS_ID {vaers_id}: {e}")
            predicted_severity = "Unknown"

        return {
            "VAERS_ID": vaers_id,
            "text": text,
            "gold_entities": gold_entities,
            "weak_severity": weak_severity,
            "ner_entities": ner_entities,
            "predicted_severity": predicted_severity
        }
    except Exception as ex:
        logger.error(f"Failed to get report for VAERS_ID {vaers_id}: {ex}")
        return {
            "VAERS_ID": vaers_id,
            "text": "Error occurred",
            "gold_entities": [],
            "weak_severity": None,
            "ner_entities": [],
            "predicted_severity": "Unknown"
        }
