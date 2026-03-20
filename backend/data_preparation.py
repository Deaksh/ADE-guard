import pandas as pd
import os
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _is_covid_row(row: pd.Series) -> bool:
    vax_type = str(row.get("VAX_TYPE", "")).upper()
    vax_name = str(row.get("VAX_NAME", "")).upper()
    if "COVID" in vax_type:
        return True
    return "COVID" in vax_name


def load_and_merge_vaers(
    data_path: Optional[str] = None,
    output_name: str = "merged_2025.csv",
    covid_only: bool = False,
):
    data_path = data_path or os.path.join(BASE_DIR, "data")

    df_data = pd.read_csv(
        os.path.join(data_path, "2025VAERSDATA.csv"),
        low_memory=False,
        encoding="latin-1",
    )
    df_symptoms = pd.read_csv(
        os.path.join(data_path, "2025VAERSSYMPTOMS.csv"),
        low_memory=False,
        encoding="latin-1",
    )
    df_vax = pd.read_csv(
        os.path.join(data_path, "2025VAERSVAX.csv"),
        low_memory=False,
        encoding="latin-1",
    )

    merged = df_data.merge(df_symptoms, on="VAERS_ID", how="left")
    merged = merged.merge(df_vax, on="VAERS_ID", how="left")

    symptom_cols = [col for col in df_symptoms.columns if col.startswith("SYMPTOM")]
    merged["ALL_SYMPTOMS"] = merged[symptom_cols].astype(str).apply(lambda x: " | ".join(x), axis=1)

    keep_cols = [
        "VAERS_ID",
        "RECVDATE",
        "AGE_YRS",
        "SEX",
        "DIED",
        "HOSPITAL",
        "VAX_TYPE",
        "VAX_NAME",
        "SYMPTOM_TEXT",
        "ALL_SYMPTOMS",
    ]
    merged = merged[keep_cols]

    if covid_only:
        merged = merged[merged.apply(_is_covid_row, axis=1)]

    output_file = os.path.join(data_path, output_name)
    merged.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Merged dataset saved at {output_file} with {len(merged)} rows")
    print(merged.head())


if __name__ == "__main__":
    load_and_merge_vaers()
