"""Crumb et al. 2016 CiPA ion-channel panel supplement parser."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pdfplumber
from scipy.optimize import curve_fit

from arie.data import load_processed_dataset
from arie.names import normalize_compound

DATASET_ID = "mechanistic_multichannel_crumb2016"
SUPPLEMENT_URL = (
    "https://cipaproject.org/wp-content/uploads/2016/12/"
    "Crumb-supplement-30-drugs-on-CiPA-ion-channels-J-Pharm-Tox-Sci-2016.pdf"
)
PROOF_URL = (
    "https://cipaproject.org/wp-content/uploads/2016/12/"
    "Crumb-30-drugs-on-CiPA-ion-channels-J-Pharm-Tox-Sci-2016-proof-ahead-of-print.pdf"
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / DATASET_ID
CACHE_DIR = ROOT / "data" / "cache"
CACHE_META = CACHE_DIR / f"{DATASET_ID}.json"

PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / f"{DATASET_ID}.csv"
DEBUG_PATH = ROOT / "results" / f"{DATASET_ID}_debug.json"
JOIN_SUMMARY_PATH = ROOT / "results" / f"{DATASET_ID}_join_summary.json"
REPORT_PATH = ROOT / "reports" / "mechanistic_coverage_multichannel_crumb2016.md"
DATA_DICT_PATH = ROOT / "reports" / f"data_dictionary_{DATASET_ID}.md"

PARSER_VERSION = "2026-02-08a"

CHANNEL_MAP = {
    "herg": "hERG",
    "nav1.5-peak": "Nav1.5_peak",
    "nav1.5-late": "Nav1.5_late",
    "cav1.2": "Cav1.2",
    "kvlqt1/mink": "IKs",
    "kvlqt1/mink": "IKs",
    "kir2.1": "IK1",
    "kv4.3": "Kv4.3",
}

CHANNEL_ORDER = [
    "hERG",
    "Nav1.5_peak",
    "Nav1.5_late",
    "Cav1.2",
    "IKs",
    "IK1",
    "Kv4.3",
]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_crumb2016(force: bool = False, include_proof: bool = False) -> dict:
    """Download the Crumb 2016 supplement PDF (and optional proof PDF)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    downloads = []

    def _download(url: str, filename: str) -> dict:
        target = RAW_DIR / filename
        if target.exists() and target.stat().st_size > 0 and not force:
            return {
                "url": url,
                "path": str(target),
                "sha256": _sha256_file(target),
                "skipped": True,
            }
        import requests

        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(target)
        return {
            "url": url,
            "path": str(target),
            "sha256": _sha256_file(target),
            "skipped": False,
        }

    downloads.append(_download(SUPPLEMENT_URL, "Crumb2016_supplement.pdf"))
    if include_proof:
        downloads.append(_download(PROOF_URL, "Crumb2016_proof.pdf"))

    meta = {
        "dataset": DATASET_ID,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloads": downloads,
    }
    CACHE_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _parse_concentration_line(line: str) -> List[float]:
    # Accept µM, uM, nM
    line = _normalize_line(line)
    matches = re.findall(r"([0-9]*\.?[0-9]+)\s*(µM|uM|μM|nM)", line)
    concs = []
    for val, unit in matches:
        val_f = float(val)
        unit = unit.replace("μ", "µ")
        if unit == "nM":
            concs.append(val_f / 1000.0)
        else:
            concs.append(val_f)
    return concs


def _normalize_line(line: str) -> str:
    # Normalize private-use glyphs used for digits in some PDF lines.
    # Map \uf030-\uf039 to 0-9, \uf02e to '.', and micro symbols to 'µ'.
    out = []
    for ch in line:
        code = ord(ch)
        if 0xF030 <= code <= 0xF039:
            out.append(chr(ord("0") + (code - 0xF030)))
        elif code == 0xF02E:
            out.append(".")
        elif ch in {"\uf06d", "\uf0b5"}:
            out.append("µ")
        else:
            out.append(ch)
    return "".join(out)


def _extract_channel(line: str) -> str | None:
    lower = _normalize_line(line).lower().strip()
    for key in CHANNEL_MAP:
        if lower.startswith(key):
            return CHANNEL_MAP[key]
    return None


def _extract_means(mean_line: str) -> List[float]:
    if not mean_line:
        return []
    mean_line = _normalize_line(mean_line)
    values = re.findall(r"([0-9]*\.?[0-9]+)\s*±", mean_line)
    return [float(v) for v in values]


def _hill_func(conc: np.ndarray, ic50: float, nh: float) -> np.ndarray:
    return 100.0 / (1.0 + np.power(ic50 / conc, nh))


def _fit_hill(concs: List[float], means: List[float]) -> Tuple[float | None, float | None, str]:
    x = np.array(concs, dtype=float)
    y = np.array(means, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return None, None, "NA"

    if np.nanmax(y) < 50.0:
        return float(np.nanmax(x)), None, ">"

    try:
        popt, _ = curve_fit(
            _hill_func,
            x,
            y,
            p0=[np.median(x), 1.0],
            bounds=([1e-6, 0.1], [1e6, 5.0]),
            maxfev=20000,
        )
        ic50, nh = popt
        if not np.isfinite(ic50) or not np.isfinite(nh):
            return None, None, "NA"
        return float(ic50), float(nh), "="
    except Exception:
        return None, None, "NA"


def parse_supplement(
    pdf_path: Path,
    enable_identity_alias: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    rows: List[dict] = []
    debug_entries: List[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                continue
            for idx, line in enumerate(lines):
                if not line.lower().startswith("current"):
                    continue
                if idx == 0:
                    continue
                drug_raw = lines[idx - 1].strip()
                concs = _parse_concentration_line(line)
                if not concs:
                    continue
                j = idx + 1
                while j < len(lines):
                    channel_line = lines[j]
                    channel = _extract_channel(channel_line)
                    if channel is None:
                        j += 1
                        continue
                    mean_line = lines[j + 1] if j + 1 < len(lines) else ""
                    means = _extract_means(mean_line)
                    if len(means) != len(concs):
                        j += 1
                        continue
                    norm = normalize_compound(drug_raw, enable_identity_alias=enable_identity_alias)
                    ic50_uM, hill_n, ic50_relation = _fit_hill(concs, means)

                    rows.append(
                        {
                            "drug_name_raw": drug_raw,
                            "drug_name_parent": norm["drug_name_parent"],
                            "drug_name_normalized": norm["drug_name_normalized"],
                            "identity_alias_enabled": enable_identity_alias,
                            "identity_alias_applied": norm["identity_alias_applied"],
                            "channel": channel,
                            "ic50_uM": ic50_uM,
                            "ic50_relation": ic50_relation,
                            "hill_n": hill_n,
                            "block_free_cmax_pct": np.nan,
                            "block_3x_free_cmax_pct": np.nan,
                            "source_name": "Crumb2016_CiPA",
                            "source_url": SUPPLEMENT_URL,
                            "retrieved_at_utc": None,
                            "source_sha256": None,
                            "parser_version": PARSER_VERSION,
                        }
                    )
                    debug_entries.append(
                        {
                            "drug_name_raw": drug_raw,
                            "channel": channel,
                            "concentrations_uM": concs,
                            "mean_block_pct": means,
                            "raw_channel_line": channel_line,
                            "mean_line": mean_line,
                        }
                    )
                    j += 2

    debug = {"entries": debug_entries}
    df = pd.DataFrame(rows)
    return df, debug


def summarize_coverage(df: pd.DataFrame) -> dict:
    cipa = load_processed_dataset()
    cipa_parents_off = sorted(
        {
            normalize_compound(name, enable_identity_alias=False)["drug_name_parent"]
            for name in cipa["drug_name"].dropna()
        }
    )
    cipa_parents_on = sorted(
        {
            normalize_compound(name, enable_identity_alias=True)["drug_name_parent"]
            for name in cipa["drug_name"].dropna()
        }
    )

    def _coverage(alias: bool) -> List[dict]:
        coverage = []
        for channel in CHANNEL_ORDER:
            subset = df[df["channel"] == channel]
            if alias:
                parents = sorted(
                    {
                        normalize_compound(name, enable_identity_alias=True)["drug_name_parent"]
                        for name in subset["drug_name_raw"]
                    }
                )
                base = cipa_parents_on
            else:
                parents = sorted(subset["drug_name_parent"].unique().tolist())
                base = cipa_parents_off
            covered = sorted(set(base).intersection(set(parents)))
            missing = sorted(set(base).difference(set(parents)))
            coverage.append(
                {
                    "channel": channel,
                    "covered_parents": len(covered),
                    "missing_parents": missing,
                }
            )
        return coverage

    return {
        "channels": CHANNEL_ORDER,
        "coverage_alias_off": _coverage(alias=False),
        "coverage_alias_on": _coverage(alias=True),
        "cipa_parent_count": len(cipa_parents_off),
    }


def write_reports(
    df: pd.DataFrame,
    metadata: dict,
    debug: dict,
) -> dict:
    summary = summarize_coverage(df)
    summary.update(
        {
            "dataset_id": DATASET_ID,
            "source_url": SUPPLEMENT_URL,
            "source_sha256": metadata.get("sha256"),
            "retrieved_at_utc": metadata.get("retrieved_at_utc"),
            "parser_version": PARSER_VERSION,
        }
    )

    JOIN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOIN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_PATH.write_text(json.dumps(debug, indent=2) + "\n", encoding="utf-8")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mechanistic Coverage: Crumb 2016 CiPA Panel",
        "",
        f"Source URL: {SUPPLEMENT_URL}",
        f"Source SHA256: {metadata.get('sha256')}",
        f"Retrieved at (UTC): {metadata.get('retrieved_at_utc')}",
        f"Parser version: {PARSER_VERSION}",
        "",
        "## Coverage by channel (alias OFF)",
    ]
    for entry in summary["coverage_alias_off"]:
        missing = entry["missing_parents"]
        lines.append(
            f"- {entry['channel']}: {entry['covered_parents']} / {summary['cipa_parent_count']} "
            f"(missing: {', '.join(missing) if missing else 'none'})"
        )
    lines.extend(["", "## Coverage by channel (alias ON)"])
    for entry in summary["coverage_alias_on"]:
        missing = entry["missing_parents"]
        lines.append(
            f"- {entry['channel']}: {entry['covered_parents']} / {summary['cipa_parent_count']} "
            f"(missing: {', '.join(missing) if missing else 'none'})"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    DATA_DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_DICT_PATH.write_text(
        """
# Data Dictionary: Crumb 2016 CiPA Panel (processed)

Source: Crumb et al. 2016 CiPA ion-channel panel supplement PDF.

Unit of observation: one row per drug × channel.

Channels:
- hERG
- Nav1.5_peak
- Nav1.5_late
- Cav1.2
- IKs (KCNQ1/KCNE1; labeled KvLQT1/minK in the supplement)
- IK1 (Kir2.1)
- Kv4.3 (Ito)

Parsing assumptions:
- Concentration-response means are extracted from the “X ± SEM” rows.
- IC50 and Hill n are estimated by fitting a Hill equation to mean % block vs concentration.
- If maximum block < 50% across tested concentrations, IC50 is treated as censored with relation “>”
  and numeric value equal to the max tested concentration.
- If fitting fails, IC50 and Hill n are set to NaN and relation “NA”.

Censored values:
- `ic50_relation` indicates “=”, “>”, “<”, or “NA”.

Provenance:
- `source_url`, `retrieved_at_utc`, `source_sha256`, and `parser_version` are recorded.

Block at Cmax:
- `block_free_cmax_pct` and `block_3x_free_cmax_pct` are not available in the supplement and are NaN.
""".lstrip(),
        encoding="utf-8",
    )
    return summary
