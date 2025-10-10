# backend/data_preparation.py
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_and_merge_vaers(data_path=os.path.join(BASE_DIR, "data")):
    # Read CSVs with correct encoding
    df_data = pd.read_csv(os.path.join(data_path, "2025VAERSDATA.csv"),
                          low_memory=False, encoding="latin-1")
    df_symptoms = pd.read_csv(os.path.join(data_path, "2025VAERSSYMPTOMS.csv"),
                              low_memory=False, encoding="latin-1")
    df_vax = pd.read_csv(os.path.join(data_path, "2025VAERSVAX.csv"),
                         low_memory=False, encoding="latin-1")

    # Merge on VAERS_ID
    merged = df_data.merge(df_symptoms, on="VAERS_ID", how="left")
    merged = merged.merge(df_vax, on="VAERS_ID", how="left")

    # Collect all symptom text into one column
    symptom_cols = [col for col in df_symptoms.columns if col.startswith("SYMPTOM")]
    merged["ALL_SYMPTOMS"] = merged[symptom_cols].astype(str).apply(lambda x: " | ".join(x), axis=1)

    # Keep relevant fields
    keep_cols = [
        "VAERS_ID", "RECVDATE", "AGE_YRS", "SEX", "DIED", "HOSPITAL",
        "VAX_TYPE", "VAX_NAME", "SYMPTOM_TEXT", "ALL_SYMPTOMS"
    ]
    merged = merged[keep_cols]

    # Save merged dataset
    output_file = os.path.join(data_path, "merged_2025.csv")
    merged.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Merged dataset saved at {output_file} with {len(merged)} rows")
    print(merged.head())

if __name__ == "__main__":
    load_and_merge_vaers()
