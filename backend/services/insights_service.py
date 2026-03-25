import pandas as pd
import re
from typing import Dict, Any


def build_clinical_insights(df: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
    out = {}

    # Top ADE symptoms
    if "ALL_SYMPTOMS" in df.columns:
        symptoms = df["ALL_SYMPTOMS"].astype(str).str.split("\s*\|\s*|;|,")
        flat = []
        for lst in symptoms:
            if isinstance(lst, list):
                flat.extend([s.strip() for s in lst if s.strip()])
        if flat:
            def ok_symptom(s: str) -> bool:
                if not s or s.lower() == "nan":
                    return False
                if s.replace(".", "", 1).isdigit():
                    return False
                return any(c.isalpha() for c in s)
            flat = [s for s in flat if ok_symptom(s)]
            counts = pd.Series(flat).value_counts().head(top_n)
            out["top_symptoms"] = [{"symptom": k, "count": int(v)} for k, v in counts.items()]
        else:
            out["top_symptoms"] = []
    else:
        out["top_symptoms"] = []

    # Severe signal counts
    severe_regex = re.compile(r"death|life-threatening|hospitalized|stroke|seizure|anaphylaxis|icu|intensive care", re.I)
    df["severe_signal"] = df["SYMPTOM_TEXT"].astype(str).str.contains(severe_regex)
    out["severe_signal_count"] = int(df["severe_signal"].sum())

    # Age band distribution
    def age_band(age):
        try:
            age = float(age)
        except Exception:
            return "Unknown"
        if age <= 17:
            return "0-17"
        if age <= 30:
            return "18-30"
        if age <= 50:
            return "31-50"
        return "51+"

    df["AGE_BAND"] = df.get("AGE_YRS", pd.NA).apply(age_band)
    age_counts = df["AGE_BAND"].value_counts().head(top_n)
    out["age_distribution"] = [{"age_band": k, "count": int(v)} for k, v in age_counts.items()]

    # Vaccine counts
    if "VAX_NAME" in df.columns:
        vax_counts = df["VAX_NAME"].astype(str).value_counts().head(top_n)
        out["top_vaccines"] = [{"vaccine": k, "count": int(v)} for k, v in vax_counts.items()]
    else:
        out["top_vaccines"] = []

    return out
