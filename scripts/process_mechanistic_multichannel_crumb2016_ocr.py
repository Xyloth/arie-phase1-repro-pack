#!/usr/bin/env python3
"""OCR-based extraction of Crumb2016 multichannel tables."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pdfplumber

from arie.names import normalize_compound

try:
    import easyocr
except Exception as exc:  # pragma: no cover - import guard
    easyocr = None

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "mechanistic_multichannel_crumb2016"
OUT_PATH = ROOT / "data" / "processed" / "mechanistic_multichannel_crumb2016_ocr.csv"
DEBUG_PATH = ROOT / "results" / "crumb2016_ocr_extraction_debug.json"
REPORT_PATH = ROOT / "reports" / "mechanistic_coverage_multichannel_crumb2016_ocr.md"

CHANNEL_PATTERNS = [
    (re.compile(r"kvlqt1|kv1qt1|min[km]", re.I), "IKs"),
    (re.compile(r"kir2\.?1", re.I), "IK1"),
    (re.compile(r"kv4\.?3|ito", re.I), "Kv4.3"),
    (re.compile(r"nav1\.?5.*late", re.I), "Nav1.5_late"),
    (re.compile(r"nav1\.?5.*peak", re.I), "Nav1.5_peak"),
    (re.compile(r"nav1\.?5", re.I), "Nav1.5_peak"),
    (re.compile(r"cav1\.?2", re.I), "Cav1.2"),
    (re.compile(r"herg", re.I), "hERG"),
]

UNIT_MAP = {
    "nm": 1e-3,
    "um": 1.0,
    "µm": 1.0,
    "mm": 1e3,
}


def _normalize_token(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def _extract_numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", text)]


def _parse_concentration(token: str) -> Tuple[float | None, str | None]:
    token = token.replace("µ", "u")
    match = re.search(r"(-?\d+\.\d+|-?\d+)\s*(nM|uM|mM)", token, re.I)
    if not match:
        return None, None
    value = float(match.group(1))
    unit = match.group(2).lower()
    return value, unit


def _map_channel(text: str) -> str | None:
    for pattern, channel in CHANNEL_PATTERNS:
        if pattern.search(text):
            return channel
    return None


def _group_lines(words: List[dict], y_tol: float) -> List[List[dict]]:
    lines: List[List[dict]] = []
    for word in sorted(words, key=lambda w: w["y_center"]):
        placed = False
        for line in lines:
            if abs(word["y_center"] - line[0]["y_center"]) <= y_tol:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])
    for line in lines:
        line.sort(key=lambda w: w["x_center"])
    return lines


def _line_text(line: List[dict]) -> str:
    return " ".join(word["text"] for word in line if word["text"])


def _extract_header_concentrations(line: List[dict]) -> List[dict]:
    concentrations = []
    for word in line:
        token = word["text"]
        if not token:
            continue
        value, unit = _parse_concentration(token)
        if value is None:
            continue
        concentrations.append(
            {
                "value": value,
                "unit": unit,
                "x_center": word["x_center"],
            }
        )
    concentrations = sorted(concentrations, key=lambda c: c["x_center"])
    return concentrations


def _assign_tokens_to_columns(tokens: List[dict], columns: List[dict]) -> List[List[dict]]:
    if not columns:
        return [[]]
    centers = [c["x_center"] for c in columns]
    boundaries = []
    for i in range(len(centers) - 1):
        boundaries.append((centers[i] + centers[i + 1]) / 2)
    buckets = [[] for _ in columns]
    for token in tokens:
        x = token["x_center"]
        idx = 0
        while idx < len(boundaries) and x > boundaries[idx]:
            idx += 1
        if idx < len(buckets):
            buckets[idx].append(token)
    return buckets


def _ocr_page(reader, page, dpi: int = 120) -> Tuple[List[dict], Tuple[int, int]]:
    image = page.to_image(resolution=dpi).original.convert("RGB")
    width, height = image.size
    # Crop to table region to reduce OCR cost.
    table_crop = image.crop((0, int(height * 0.18), width, int(height * 0.95)))
    max_width = 1000
    if table_crop.size[0] > max_width:
        ratio = max_width / table_crop.size[0]
        new_size = (max_width, int(table_crop.size[1] * ratio))
        table_crop = table_crop.resize(new_size)
    results = reader.readtext(np.array(table_crop), detail=1, paragraph=False)
    words = []
    for bbox, text, conf in results:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        ys = [y + height * 0.18 for y in ys]
        words.append(
            {
                "text": text,
                "conf": float(conf),
                "x_center": float(sum(xs) / len(xs)),
                "y_center": float(sum(ys) / len(ys)),
            }
        )
    return words, (width, height)


def _detect_drug_name(reader, page, dpi: int = 120) -> str | None:
    image = page.to_image(resolution=dpi).original.convert("RGB")
    width, height = image.size
    header_crop = image.crop((0, 0, width, int(height * 0.15)))
    results = reader.readtext(np.array(header_crop), detail=1, paragraph=False)
    words = []
    for bbox, text, conf in results:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        words.append(
            {
                "text": text,
                "conf": float(conf),
                "x_center": float(sum(xs) / len(xs)),
                "y_center": float(sum(ys) / len(ys)),
            }
        )
    if not words:
        return None
    lines = _group_lines(words, y_tol=8)
    for line in lines:
        text = _line_text(line)
        if re.search(r"[a-zA-Z]", text) and not re.search(r"\d", text):
            return text.strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--max-pages", type=int, default=None, help="Optional cap on pages processed.")
    args = parser.parse_args()

    if easyocr is None:
        raise RuntimeError("easyocr is not available in this environment.")

    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found under {RAW_DIR}")

    reader = easyocr.Reader(["en"], gpu=False)

    records = []
    debug = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdfs": [],
        "pages": [],
        "failures": [],
    }

    for pdf_path in pdf_paths:
        debug["pdfs"].append(str(pdf_path))
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                if args.max_pages and page_idx > args.max_pages:
                    break
                if page_idx % 5 == 0:
                    print(f"OCR progress: page {page_idx} / {len(pdf.pages)}")
                if not page.images:
                    debug["failures"].append(
                        {"page": page_idx, "reason": "no_images", "pdf": str(pdf_path)}
                    )
                    continue
                words, (width, height) = _ocr_page(reader, page, dpi=args.dpi)
                if not words:
                    debug["failures"].append(
                        {"page": page_idx, "reason": "no_words", "pdf": str(pdf_path)}
                    )
                    continue

                lines = _group_lines(words, y_tol=10)
                header_line = None
                for line in lines:
                    if "current" in _line_text(line).casefold():
                        header_line = line
                        break
                if header_line is None:
                    debug["failures"].append(
                        {"page": page_idx, "reason": "no_header", "pdf": str(pdf_path)}
                    )
                    continue

                concentrations = _extract_header_concentrations(header_line)
                if not concentrations:
                    debug["failures"].append(
                        {"page": page_idx, "reason": "no_concentrations", "pdf": str(pdf_path)}
                    )
                    continue

                drug_name = _detect_drug_name(reader, page, dpi=args.dpi)
                if not drug_name:
                    debug["failures"].append(
                        {"page": page_idx, "reason": "no_drug_name", "pdf": str(pdf_path)}
                    )
                    continue

                debug["pages"].append(
                    {
                        "pdf": str(pdf_path),
                        "page": page_idx,
                        "drug_name_raw": drug_name,
                        "ocr_conf_mean": float(np.mean([w["conf"] for w in words])) if words else None,
                        "concentration_headers": [f"{c['value']}{c['unit']}" for c in concentrations],
                        "channels_detected": [],
                    }
                )

                for line in lines:
                    line_text = _line_text(line)
                    channel = _map_channel(line_text)
                    if channel is None:
                        continue
                    channel_tokens = [w for w in line if w["x_center"] > header_line[0]["x_center"]]
                    columns = _assign_tokens_to_columns(channel_tokens, concentrations)
                    debug["pages"][-1]["channels_detected"].append(channel)

                    for col_idx, col_tokens in enumerate(columns):
                        if col_idx >= len(concentrations):
                            continue
                        raw = " ".join(t["text"] for t in col_tokens if t["text"]).strip()
                        nums = _extract_numbers(raw)
                        if not nums:
                            continue
                        mean = float(np.mean(nums))
                        sem = float(np.std(nums, ddof=1) / math.sqrt(len(nums))) if len(nums) > 1 else None
                        conc = concentrations[col_idx]
                        unit = conc["unit"]
                        factor = UNIT_MAP.get(unit, None)
                        conc_um = conc["value"] * factor if factor else None

                        records.append(
                            {
                                "drug_name_raw": drug_name,
                                "drug_name_parent": normalize_compound(drug_name, enable_identity_alias=False)[
                                    "drug_name_parent"
                                ],
                                "drug_name_normalized": normalize_compound(drug_name, enable_identity_alias=False)[
                                    "drug_name_normalized"
                                ],
                                "channel": channel,
                                "concentration": conc["value"],
                                "concentration_units": unit,
                                "concentration_uM": conc_um,
                                "block_values_raw": raw,
                                "block_mean": mean,
                                "block_sem": sem,
                                "source_pdf": pdf_path.name,
                                "page": page_idx,
                            }
                        )

    out_df = pd.DataFrame(records)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    # Coverage report
    cipa = pd.read_csv(ROOT / "data" / "processed" / "cipa_blinova_2018.csv")
    cipa_parents = sorted(
        {
            normalize_compound(name, enable_identity_alias=False)["drug_name_parent"]
            for name in cipa["drug_name"].dropna()
        }
    )
    ocr_parents = sorted(out_df["drug_name_parent"].dropna().unique().tolist())
    overlap = sorted(set(cipa_parents) & set(ocr_parents))
    ocr_parents_alias = sorted(
        {
            normalize_compound(name, enable_identity_alias=True)["drug_name_parent"]
            for name in out_df["drug_name_raw"].dropna().unique().tolist()
        }
    )
    cipa_parents_alias = sorted(
        {
            normalize_compound(name, enable_identity_alias=True)["drug_name_parent"]
            for name in cipa["drug_name"].dropna()
        }
    )
    overlap_alias = sorted(set(cipa_parents_alias) & set(ocr_parents_alias))

    coverage_all = (
        out_df.groupby("channel")["drug_name_parent"]
        .nunique()
        .reindex(sorted(out_df["channel"].unique()))
        .to_dict()
    )
    overlap_df = out_df[out_df["drug_name_parent"].isin(overlap)]
    coverage_overlap = (
        overlap_df.groupby("channel")["drug_name_parent"]
        .nunique()
        .reindex(sorted(out_df["channel"].unique()))
        .fillna(0)
        .astype(int)
        .to_dict()
    )

    report_lines = []
    report_lines.append("# Crumb2016 OCR Coverage Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    report_lines.append("")
    report_lines.append(f"Extracted drugs (OCR): {len(ocr_parents)}")
    report_lines.append(f"Overlap with CiPA-28 (alias off): {len(overlap)} / {len(cipa_parents)}")
    report_lines.append(f"Overlap with CiPA-28 (alias on): {len(overlap_alias)} / {len(cipa_parents_alias)}")
    report_lines.append("")
    report_lines.append("## Extracted drugs (OCR)")
    report_lines.append(", ".join(ocr_parents))
    report_lines.append("")
    report_lines.append("## Overlap with CiPA-28")
    report_lines.append(", ".join(overlap))
    report_lines.append("")
    report_lines.append("## Channel coverage (all OCR drugs)")
    for channel, count in coverage_all.items():
        report_lines.append(f"- {channel}: {count}")
    report_lines.append("")
    report_lines.append("## Channel coverage (CiPA overlap)")
    for channel, count in coverage_overlap.items():
        report_lines.append(f"- {channel}: {count}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_PATH.write_text(json.dumps(debug, indent=2) + "\n", encoding="utf-8")

    print(f"Pages processed: {len(debug['pages'])}")
    print(f"Drugs extracted: {len(ocr_parents)}")
    print(f"Channels extracted: {sorted(out_df['channel'].unique()) if not out_df.empty else []}")
    print(f"Overlap vs CiPA-28 (alias off): {len(overlap)} / {len(cipa_parents)}")
    print(f"Overlap vs CiPA-28 (alias on): {len(overlap_alias)} / {len(cipa_parents_alias)}")
    print(f"Wrote: {OUT_PATH}")
    print(f"Wrote: {DEBUG_PATH}")
    print(f"Wrote: {REPORT_PATH}")


if __name__ == "__main__":
    main()
