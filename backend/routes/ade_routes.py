# backend/routes/ade_routes.py
from fastapi import APIRouter
from backend.services.ade_service import get_summary_data, get_trends_data, get_alerts_data, search_symptom_data

router = APIRouter()

@router.get("/summary")
async def get_summary():
    return get_summary_data()

@router.get("/trends")
async def get_trends():
    return get_trends_data()

@router.get("/alerts")
async def get_alerts():
    return get_alerts_data()

@router.get("/search")
async def search(symptom: str):
    return search_symptom_data(symptom)
