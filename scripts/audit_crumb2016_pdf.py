#!/usr/bin/env python3
"""Forensic audit of Crumb2016 PDFs for CiPA-28 coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import difflib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pdfplumber

from arie.data import load_processed_dataset
from arie.names import normalize_compound

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "mechanistic_multichannel_crumb2016"
REPORT_PATH = ROOT / "reports" / "crumb2016_pdf_audit.md"
JSON_PATH = ROOT / "results" / "crumb2016_pdf_audit.json"
TABLES_DIR = ROOT / "results" / "crumb2016_pdf_tables_raw"
LONG_PATH = ROOT / "results" / "crumb2016_pdf_long_extracted.csv"

CHANNEL_TERMS = [
    "IKs",
    "IK1",
    "Kv4.3",
    "Ito",
    "KCNQ1",
    "Kir2.1",
    "Nav1.5",
    "Cav1.2",
    "hERG",
]

CHANNEL_MAP = {
    "IKs": "IKs",
    "IK1": "IK1",
    "Kv4.3": "Kv4.3",
    "Ito": "Kv4.3",
    "KCNQ1": "IKs",
    "Kir2.1": "IK1",
    "Nav1.5": "Nav1.5",
    "Cav1.2": "Cav1.2",
    "hERG": "hERG",
}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _is_numeric(value: str) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if s == "":
        return False
    return bool(re.search(r"\d", s))


def _extract_numeric(value: str) -> Tuple[float | None, str]:
    if value is None:
        return None, "NA"
    s = str(value).strip()
    if s == "":
        return None, "NA"
    relation = "="
    if ">" in s:
        relation = ">"
    elif "<" in s:
        relation = "<"
    matches = re.findall(r"-?\d+\.\d+|-?\d+", s.replace(",", ""))
    if not matches:
        return None, "NA"
    try:
        return float(matches[0]), relation
    except ValueError:
        return None, "NA"


def _table_signature(table: List[List[str]]) -> dict:
    rows = len(table)
    cols = max((len(row) for row in table), default=0)
    total_cells = rows * cols if rows and cols else 0
    numeric_cells = 0
    for row in table:
        for cell in row:
            if _is_numeric(cell):
                numeric_cells += 1
    header_like = False
    if rows:
        first = table[0]
        non_numeric = sum(1 for c in first if c and not _is_numeric(c))
        header_like = non_numeric >= max(1, len(first) // 2)
    return {
        "rows": rows,
        "cols": cols,
        "numeric_pct": float(numeric_cells / total_cells) if total_cells else 0.0,
        "header_like": header_like,
    }


def _save_table(table: List[List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in table:
            writer.writerow(row)


def _heuristic_tables(text: str) -> List[List[List[str]]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tables = []
    current = []
    for line in lines:
        parts = re.split(r"\s{2,}", line)
        numeric_count = sum(1 for p in parts if _is_numeric(p))
        if len(parts) >= 4 and numeric_count >= 2:
            current.append(parts)
        else:
            if len(current) >= 2:
                tables.append(current)
            current = []
    if len(current) >= 2:
        tables.append(current)
    return tables


def _extract_candidate_drugs(table: List[List[str]]) -> List[str]:
    candidates = []
    if not table:
        return candidates
    header = table[0]
    header_text = " ".join(str(c) for c in header if c)
    for row in table[1:] if "drug" in header_text.casefold() else table:
        if not row:
            continue
        cell = str(row[0]).strip()
        if not cell:
            continue
        if re.search(r"[a-zA-Z]", cell) and len(cell) <= 40:
            lowered = cell.casefold()
            if any(key in lowered for key in ["ic50", "block", "conc", "drug", "mean", "sem"]):
                continue
            candidates.append(cell)
    return candidates


def _guess_channels(page_text: str) -> List[str]:
    matches = []
    for term in CHANNEL_TERMS:
        if term.casefold() in page_text.casefold():
            matches.append(CHANNEL_MAP.get(term, term))
    return sorted(set(matches))


def _value_type_from_header(header: List[str]) -> str:
    header_text = " ".join(str(c) for c in header if c).casefold()
    if "ic50" in header_text:
        return "ic50"
    if "block" in header_text or "%" in header_text:
        return "pct_block"
    if "cmax" in header_text:
        return "block_cmax"
    return "unknown"


def _units_from_header(header: List[str]) -> str | None:
    header_text = " ".join(str(c) for c in header if c)
    if re.search(r"(uM|µM|um)", header_text):
        return "uM"
    if re.search(r"nM", header_text):
        return "nM"
    if re.search(r"mM", header_text):
        return "mM"
    return None


def _long_format_from_table(
    table: List[List[str]],
    pdf_name: str,
    page_number: int,
    table_id: str,
    strategy: str,
    channel_candidates: List[str],
) -> List[dict]:
    if not table:
        return []
    header = table[0]
    header_like = _table_signature(table)["header_like"]
    rows = table[1:] if header_like else table
    value_type = _value_type_from_header(header) if header_like else "unknown"
    units = _units_from_header(header) if header_like else None
    records = []
    for row in rows:
        if not row:
            continue
        drug_raw = str(row[0]).strip()
        if not drug_raw or not re.search(r"[a-zA-Z]", drug_raw):
            continue
        for idx, cell in enumerate(row[1:], start=1):
            if cell is None or str(cell).strip() == "":
                continue
            value, relation = _extract_numeric(str(cell))
            if value is None:
                continue
            header_label = str(header[idx]).strip() if idx < len(header) else ""
            concentration = None
            if header_label and _is_numeric(header_label):
                concentration, _ = _extract_numeric(header_label)
            record = {
                "drug_name_raw": drug_raw,
                "drug_name_parent": normalize_compound(drug_raw, enable_identity_alias=False)[
                    "drug_name_parent"
                ],
                "drug_name_normalized": normalize_compound(drug_raw, enable_identity_alias=False)[
                    "drug_name_normalized"
                ],
                "channel": channel_candidates[0] if len(channel_candidates) == 1 else "multiple",
                "channel_candidates": "|".join(channel_candidates),
                "value_type": value_type,
                "value": value,
                "value_raw": str(cell).strip(),
                "relation": relation,
                "units": units,
                "concentration_uM": concentration,
                "source_pdf": pdf_name,
                "page": page_number,
                "table_id": table_id,
                "strategy": strategy,
                "header_label": header_label,
            }
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Crumb2016 PDFs for coverage.")
    parser.add_argument("--save-raw-tables", action="store_true", help="Dump raw tables to CSV.")
    args = parser.parse_args()

    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found under {RAW_DIR}")

    cipa = load_processed_dataset()
    cipa_parents = sorted(
        {
            normalize_compound(name, enable_identity_alias=False)["drug_name_parent"]
            for name in cipa["drug_name"].dropna()
        }
    )

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdfs": [],
        "cipa_parent_count": len(cipa_parents),
        "cipa_parents": cipa_parents,
        "tables": [],
        "extracted_drugs_raw": [],
        "extracted_parents_alias_off": [],
        "extracted_parents_alias_on": [],
        "text_layer": {},
        "channels_text_layer": {},
        "long_format_path": str(LONG_PATH),
    }

    raw_drug_candidates = set()
    long_records: List[dict] = []

    for pdf_path in pdf_paths:
        pdf_name = pdf_path.name
        pdf_info = {
            "path": str(pdf_path),
            "size_bytes": pdf_path.stat().st_size,
            "sha256": _sha256_file(pdf_path),
        }

        with pdfplumber.open(pdf_path) as pdf:
            pdf_info["page_count"] = len(pdf.pages)
            audit["pdfs"].append(pdf_info)

            # Text-layer search
            found_drugs = {parent: [] for parent in cipa_parents}
            found_channels = {term: [] for term in CHANNEL_TERMS}

            for page_idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text_norm = _normalize_text(text)
                for parent in cipa_parents:
                    if parent and parent in text_norm:
                        found_drugs[parent].append(page_idx)
                for term in CHANNEL_TERMS:
                    if term.casefold() in text.casefold():
                        found_channels[term].append(page_idx)

            audit["text_layer"][pdf_name] = {
                "found_drugs_text_layer": {
                    k: v for k, v in found_drugs.items() if v
                },
                "missing_drugs_text_layer": [
                    k for k, v in found_drugs.items() if not v
                ],
            }
            audit["channels_text_layer"][pdf_name] = {
                "found_channels_text_layer": {
                    k: v for k, v in found_channels.items() if v
                },
                "missing_channels_text_layer": [
                    k for k, v in found_channels.items() if not v
                ],
            }

            # Table extraction strategies
            strategies = {
                "plumber_default": {},
                "plumber_lines": {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                },
                "plumber_text": {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                },
            }

            camelot_available = False
            try:
                import camelot  # type: ignore

                camelot_available = True
            except Exception:
                camelot_available = False

            for page_idx, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                channel_candidates = _guess_channels(page_text)

                for strategy, settings in strategies.items():
                    tables = page.extract_tables(table_settings=settings) if settings else page.extract_tables()
                    for t_idx, table in enumerate(tables):
                        signature = _table_signature(table)
                        table_id = f"{pdf_path.stem}_p{page_idx:03d}_{strategy}_t{t_idx}"
                        audit["tables"].append(
                            {
                                "pdf": pdf_name,
                                "page": page_idx,
                                "strategy": strategy,
                                "table_id": table_id,
                                "signature": signature,
                            }
                        )
                        if args.save_raw_tables:
                            out_path = TABLES_DIR / f"{table_id}.csv"
                            _save_table(table, out_path)
                        raw_drug_candidates.update(_extract_candidate_drugs(table))
                        long_records.extend(
                            _long_format_from_table(
                                table,
                                pdf_name=pdf_name,
                                page_number=page_idx,
                                table_id=table_id,
                                strategy=strategy,
                                channel_candidates=channel_candidates,
                            )
                        )

                # Heuristic strategy
                heur_tables = _heuristic_tables(page_text)
                for t_idx, table in enumerate(heur_tables):
                    signature = _table_signature(table)
                    table_id = f"{pdf_path.stem}_p{page_idx:03d}_heur_t{t_idx}"
                    audit["tables"].append(
                        {
                            "pdf": pdf_name,
                            "page": page_idx,
                            "strategy": "heuristic",
                            "table_id": table_id,
                            "signature": signature,
                        }
                    )
                    if args.save_raw_tables:
                        out_path = TABLES_DIR / f"{table_id}.csv"
                        _save_table(table, out_path)
                    raw_drug_candidates.update(_extract_candidate_drugs(table))
                    long_records.extend(
                        _long_format_from_table(
                            table,
                            pdf_name=pdf_name,
                            page_number=page_idx,
                            table_id=table_id,
                            strategy="heuristic",
                            channel_candidates=channel_candidates,
                        )
                    )

            audit["pdfs"][-1]["camelot_available"] = camelot_available

            if camelot_available:
                # Camelot extraction is optional; record availability only.
                audit["pdfs"][-1]["camelot_note"] = "camelot available but not run (use if needed)"
            else:
                audit["pdfs"][-1]["camelot_note"] = "camelot not available"

    raw_unique = sorted(set(raw_drug_candidates))
    parents_off = sorted(
        {
            normalize_compound(name, enable_identity_alias=False)["drug_name_parent"]
            for name in raw_unique
        }
    )
    parents_on = sorted(
        {
            normalize_compound(name, enable_identity_alias=True)["drug_name_parent"]
            for name in raw_unique
        }
    )
    audit["extracted_drugs_raw"] = raw_unique
    audit["extracted_parents_alias_off"] = parents_off
    audit["extracted_parents_alias_on"] = parents_on
    intersection_off = sorted(set(cipa_parents) & set(parents_off))
    missing_off = sorted(set(cipa_parents) - set(parents_off))
    audit["extracted_parents_intersection_alias_off"] = intersection_off
    audit["extracted_parents_missing_alias_off"] = missing_off
    missing_parent_summary = {}
    for parent in missing_off:
        found_in_text = any(
            parent in info["found_drugs_text_layer"]
            for info in audit["text_layer"].values()
        )
        found_in_tables = parent in parents_off
        missing_parent_summary[parent] = {
            "found_in_text_layer": found_in_text,
            "found_in_tables_raw": found_in_tables,
            "closest_matches": difflib.get_close_matches(parent, parents_off, n=5),
        }
    audit["missing_parent_summary"] = missing_parent_summary

    # Long-format extraction and coverage
    if long_records:
        long_df = pd.DataFrame(long_records)
        LONG_PATH.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(LONG_PATH, index=False)
    else:
        long_df = pd.DataFrame()

    # Coverage vs CiPA parents
    coverage = {}
    missing_info = {}
    if not long_df.empty:
        for channel in sorted(long_df["channel"].unique().tolist()):
            if channel == "multiple":
                continue
            subset = long_df[long_df["channel"] == channel]
            parents = sorted(set(subset["drug_name_parent"]))
            missing = sorted(set(cipa_parents) - set(parents))
            coverage[channel] = {
                "present": len(parents),
                "missing": len(missing),
                "missing_parents": missing,
            }
            missing_info[channel] = {}
            for parent in missing:
                raw_matches = [r for r in raw_unique if normalize_compound(r, False)["drug_name_parent"] == parent]
                missing_info[channel][parent] = {
                    "found_in_text_layer": any(
                        parent in audit["text_layer"][pdf["path"].split("/")[-1]]["found_drugs_text_layer"]
                        for pdf in audit["pdfs"]
                    ),
                    "found_in_tables_raw": len(raw_matches) > 0,
                    "closest_matches": difflib.get_close_matches(parent, parents_off, n=5),
                }

    audit["coverage_by_channel"] = coverage
    audit["missing_diagnostics"] = missing_info

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    # Markdown report
    lines = []
    lines.append("# Crumb2016 PDF Extraction Audit")
    lines.append("")
    lines.append(f"Generated: {audit['generated_at_utc']}")
    lines.append("")
    lines.append("## PDFs Audited")
    for pdf in audit["pdfs"]:
        lines.append(f"- {pdf['path']}")
        lines.append(f"  size_bytes: {pdf['size_bytes']}")
        lines.append(f"  sha256: {pdf['sha256']}")
        lines.append(f"  pages: {pdf['page_count']}")
        lines.append(f"  camelot: {pdf.get('camelot_note')}")
    lines.append("")
    lines.append("## Text-Layer Search Summary")
    for pdf_name, info in audit["text_layer"].items():
        lines.append(f"- {pdf_name}: found {len(info['found_drugs_text_layer'])} / {len(cipa_parents)} drugs")
        lines.append(f"  missing_drugs_text_layer: {len(info['missing_drugs_text_layer'])}")
        if info["found_drugs_text_layer"]:
            lines.append(
                "  found_drugs_text_layer: "
                + ", ".join(sorted(info["found_drugs_text_layer"].keys()))
            )
        if info["missing_drugs_text_layer"]:
            lines.append(
                "  missing_drugs_text_layer_list: "
                + ", ".join(info["missing_drugs_text_layer"])
            )
        channel_info = audit["channels_text_layer"].get(pdf_name, {})
        found_channels = channel_info.get("found_channels_text_layer", {})
        if found_channels:
            lines.append(
                "  channels_found_text_layer: "
                + ", ".join(sorted(found_channels.keys()))
            )
        missing_channels = channel_info.get("missing_channels_text_layer", [])
        if missing_channels:
            lines.append(
                "  channels_missing_text_layer: "
                + ", ".join(missing_channels)
            )
    lines.append("")
    lines.append("## Extracted Drug Names")
    lines.append(f"- raw unique: {len(raw_unique)}")
    lines.append(f"- parents (alias off): {len(parents_off)}")
    lines.append(f"- parents (alias on): {len(parents_on)}")
    lines.append(f"- parents intersect CiPA (alias off): {len(intersection_off)}")
    lines.append(f"- parents missing from CiPA (alias off): {len(missing_off)}")
    lines.append("")
    lines.append("Missing CiPA parents (alias off):")
    lines.append(", ".join(missing_off) if missing_off else "None")
    lines.append("")
    lines.append("## Coverage by Channel (from extracted tables)")
    if coverage:
        for channel, info in coverage.items():
            lines.append(
                f"- {channel}: present {info['present']} / {len(cipa_parents)}"
            )
            lines.append(f"  missing: {len(info['missing_parents'])}")
    else:
        lines.append("- No channel coverage detected (no long-format rows extracted).")
    lines.append("")
    found_text = set()
    for info in audit["text_layer"].values():
        found_text.update(info["found_drugs_text_layer"].keys())
    if intersection_off and len(intersection_off) < len(cipa_parents) and len(found_text) == len(intersection_off):
        outcome = "Outcome B (PDF likely lacks missing CiPA parents; text layer and tables show same subset)."
    elif not raw_unique:
        outcome = "Outcome C (ambiguous; no tables extracted)."
    else:
        outcome = "Outcome A/B (see JSON diagnostics for missing parents and table evidence)."

    lines.append("## Outcome")
    lines.append(outcome)
    lines.append(
        "See JSON for per-drug diagnostics, closest-match suggestions, and table signatures."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Console summary (short)
    print("PDFs audited:")
    for pdf in audit["pdfs"]:
        print(f"- {pdf['path']} | sha256 {pdf['sha256']}")
    print(f"CiPA parent count: {len(cipa_parents)}")
    print(f"Extracted parents (alias off): {len(parents_off)}")
    print(f"Extracted parents ∩ CiPA (alias off): {len(intersection_off)}")
    print(f"Found in text layer (unique): {len(found_text)}")
    outcome = "B (likely absence)" if len(found_text) == len(intersection_off) and len(intersection_off) < len(cipa_parents) else "A/B (see report)"
    print(f"Outcome: {outcome}")
    print(f"Report: {REPORT_PATH}")
    print(f"JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
