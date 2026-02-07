#!/usr/bin/env python3
"""Process the CiPA Blinova 2018 dataset into a clean, analysis-ready table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "cipa_blinova_2018" / "Blinova_etal_2018_data.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "cipa_blinova_2018.csv"
REPORTS_DIR = ROOT / "reports"
DATA_DICT_PATH = REPORTS_DIR / "data_dictionary_cipa_blinova_2018.md"


COLUMN_MAP = {
    "Drug_Name": "drug_name",
    "Cell_type": "cell_type",
    "risk": "risk_class",
    "Platform": "platform",
    "Type_of_EADs": "ead_type",
    "conc": "concentration_level",
    "EAD": "ead",
    "ddFPDc": "dd_fpdc",
    "site": "site",
}


def _standardize_strings(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def process() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    df = pd.read_excel(RAW_PATH)
    df = df.rename(columns=COLUMN_MAP)

    # Keep only known columns in a consistent order.
    ordered_cols = [
        "drug_name",
        "cell_type",
        "risk_class",
        "platform",
        "ead_type",
        "concentration_level",
        "ead",
        "dd_fpdc",
        "site",
    ]
    df = df[ordered_cols]

    df["drug_name"] = _standardize_strings(df["drug_name"])
    df["cell_type"] = _standardize_strings(df["cell_type"])
    df["risk_class"] = _standardize_strings(df["risk_class"]).str.upper()
    df["platform"] = _standardize_strings(df["platform"])
    df["ead_type"] = _standardize_strings(df["ead_type"]).str.upper()

    df["concentration_level"] = pd.to_numeric(df["concentration_level"], errors="coerce").astype("Int64")
    df["ead"] = pd.to_numeric(df["ead"], errors="coerce").astype("Int64")
    df["dd_fpdc"] = pd.to_numeric(df["dd_fpdc"], errors="coerce")
    df["site"] = pd.to_numeric(df["site"], errors="coerce").astype("Int64")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DICT_PATH.write_text(
        """
# Data Dictionary: CiPA Blinova 2018 (processed)

Unit of observation: one measurement row per drug × concentration level × platform × cell type × site (as recorded in the source file).

Columns:
- drug_name: Compound name (string).
- cell_type: Cell type label from source (e.g., CDI, AXG).
- risk_class: Risk class label from source (L/M/H). Missing values kept as-is.
- platform: Platform label from source (e.g., ACA, AXN, MCS, AMD, ECR, CLY).
- ead_type: EAD type code from source (e.g., A, B, C, D, Q). Missing values kept as-is.
- concentration_level: Concentration level as provided in the source file (integer 1–4). Units not specified in the file.
- ead: Early afterdepolarization flag from source (0/1).
- dd_fpdc: Numeric metric from source column ddFPDc. Units not specified in the file.
- site: Site identifier from source (integer).
""".lstrip(),
        encoding="utf-8",
    )

    return df


if __name__ == "__main__":
    processed = process()
    print(f"Wrote {len(processed)} rows to {PROCESSED_PATH}.")
