# backend/services/ade_service.py

def get_summary_data():
    return {
        "total_reports": 15234,
        "unique_symptoms": 542,
        "vaccines_tracked": 18,
        "latest_report_date": "2025-01-05"
    }

def get_trends_data():
    return {
        "dates": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "report_counts": [120, 135, 142],
        "ade_signals": [3, 5, 6]
    }

def get_alerts_data():
    return {
        "alerts": [
            {
                "symptom": "Anaphylaxis",
                "vaccine": "Pfizer COVID-19",
                "count": 32,
                "signal_strength": "high"
            },
            {
                "symptom": "Myocarditis",
                "vaccine": "Moderna COVID-19",
                "count": 21,
                "signal_strength": "medium"
            }
        ]
    }

def search_symptom_data(symptom: str):
    return {
        "symptom": symptom,
        "reports": [
            {
                "vaers_id": 2818783,
                "date": "2025-01-01",
                "vaccine": "Shingrix",
                "description": "Pain down arm and weakness for a few days after vaccine"
            }
        ]
    }
