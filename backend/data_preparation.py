import pandas as pd
import os
import zipfile
from typing import Optional, Tuple

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


def _find_year_file_in_zip(zf: zipfile.ZipFile, year: int, kind: str) -> str:
    kind_upper = kind.upper()
    year_str = str(year)
    candidates = []
    for name in zf.namelist():
        upper = name.upper()
        if year_str in upper and kind_upper in upper and upper.endswith(".CSV"):
            candidates.append(name)
    if not candidates:
        raise FileNotFoundError(f"Missing {year} VAERS {kind} file in zip")
    # Prefer shortest path (top-level)
    return sorted(candidates, key=len)[0]


def _load_year_frames(year: int, data_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zip_name = f"{year}VAERSData.zip"
    zip_path = os.path.join(data_root, zip_name)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing VAERS zip for year {year}: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        data_file = _find_year_file_in_zip(zf, year, "VAERSDATA")
        symp_file = _find_year_file_in_zip(zf, year, "VAERSSYMPTOMS")
        vax_file = _find_year_file_in_zip(zf, year, "VAERSVAX")

        with zf.open(data_file) as f:
            df_data = pd.read_csv(f, low_memory=False, encoding="latin-1")
        with zf.open(symp_file) as f:
            df_symptoms = pd.read_csv(f, low_memory=False, encoding="latin-1")
        with zf.open(vax_file) as f:
            df_vax = pd.read_csv(f, low_memory=False, encoding="latin-1")

    return df_data, df_symptoms, df_vax


def load_and_merge_vaers_year(
    year: int,
    data_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    covid_only: bool = False,
) -> pd.DataFrame:
    data_root = data_root or os.path.join(BASE_DIR, "data")
    df_data, df_symptoms, df_vax = _load_year_frames(year, data_root)

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

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"merged_{year}.csv")
        merged.to_csv(output_file, index=False, encoding="utf-8")
        print(f"✅ Merged dataset saved at {output_file} with {len(merged)} rows")

    return merged
