"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "cipa_blinova_2018.csv"


def load_processed_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the processed CiPA Blinova 2018 dataset as a pandas DataFrame."""
    csv_path = path or PROCESSED_PATH
    return pd.read_csv(csv_path)
