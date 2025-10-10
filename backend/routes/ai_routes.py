# backend/routes/ai_routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter()

# Request model for AI endpoint
class ADERequest(BaseModel):
    vaers_id: int
    age: int = None  # optional

# Response model for token highlights
class TokenHighlight(BaseModel):
    token: str
    entity: str = None
    severity: str = None

# Placeholder NER endpoint
@router.get("/ner/{vaers_id}", response_model=List[TokenHighlight])
def ner_extract(vaers_id: int):
    # Here call your BioBERT NER model
    return [
        {"token": "myocarditis", "entity": "ADE", "severity": "High"},
        {"token": "Pfizer COVID-19", "entity": "DRUG"}
    ]

# Severity endpoint
@router.get("/severity/{vaers_id}")
def classify_severity(vaers_id: int):
    return {"VAERS_ID": vaers_id, "severity": "High", "probabilities": {"Mild": 0.1, "Moderate": 0.2, "Severe": 0.7}}

# Clustering endpoint
@router.get("/cluster")
def cluster_ades():
    return {
        "clusters": [
            {"cluster_id": 1, "ades": ["myocarditis", "anaphylaxis"], "age_group": "18-30"},
            {"cluster_id": 2, "ades": ["headache"], "age_group": "31-50"}
        ]
    }
