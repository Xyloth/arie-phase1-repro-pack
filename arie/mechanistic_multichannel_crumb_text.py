"""Parse manually extracted Crumb 2016 text tables into long-format data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from arie.data import load_processed_dataset
from arie.names import normalize_compound

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "mechanistic_multichannel_crumb_text"

DEFAULT_INPUT = ROOT / "raw" / "crumb_extraction.txt"
PROCESSED_PATH = ROOT / "data" / "processed" / f"{DATASET_ID}.csv"
DEBUG_PATH = ROOT / "results" / f"{DATASET_ID}_debug.json"
JOIN_SUMMARY_PATH = ROOT / "results" / f"{DATASET_ID}_join_summary.json"
REPORT_PATH = ROOT / "reports" / "mechanistic_coverage_multichannel_crumb_text.md"

CHANNEL_MAP = {
    "herg": "hERG",
    "nav1.5-peak": "Nav1.5_peak",
    "nav1.5-late": "Nav1.5_late",
    "cav1.2": "Cav1.2",
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


@dataclass
class ParseWarning:
    drug: str
    channel: str | None
    issue: str
    detail: str


def _normalize_line(line: str) -> str:
    out = []
    for ch in line:
        code = ord(ch)
        if 0xF030 <= code <= 0xF039:
            out.append(chr(ord("0") + (code - 0xF030)))
        elif code == 0xF02E:
            out.append(".")
        elif ch in {"\uf06d", "\uf0b5"}:
            out.append("u")
        else:
            out.append(ch)
    return "".join(out)


def _resolve_input_path(path: Path) -> Path:
    if path.exists():
        return path
    candidates = []
    for base in [ROOT, ROOT / "raw"]:
        if not base.exists():
            continue
        for item in base.iterdir():
            if item.is_file() and "crumb_extraction" in item.name.casefold():
                candidates.append(item)
    if candidates:
        return sorted(candidates)[0]
    raise FileNotFoundError(
        f"Input file not found at {path}. Provide --input pointing to crumb_extraction.txt."
    )


def _extract_channel(line: str) -> Tuple[str | None, str]:
    lower = line.casefold().strip()
    for key in sorted(CHANNEL_MAP.keys(), key=len, reverse=True):
        if lower.startswith(key.casefold()):
            return CHANNEL_MAP[key], line[len(key) :].strip()
    return None, line


def _parse_concentrations(line: str) -> Tuple[List[float], List[str], List[str]]:
    line = _normalize_line(line)
    matches = re.findall(r"(-?\d+(?:\.\d+)?)(\s*(?:nM|uM|mM))", line, flags=re.IGNORECASE)
    conc_uM: List[float] = []
    conc_raw: List[str] = []
    conc_unit: List[str] = []
    for val, unit in matches:
        unit_clean = unit.strip()
        conc_raw.append(f"{val}{unit_clean}")
        conc_unit.append(unit_clean)
        val_f = float(val)
        unit_lower = unit_clean.casefold()
        if unit_lower == "nm":
            conc_uM.append(val_f / 1000.0)
        elif unit_lower == "mm":
            conc_uM.append(val_f * 1000.0)
        else:
            conc_uM.append(val_f)
    return conc_uM, conc_raw, conc_unit


def _parse_mean_sem(line: str) -> Tuple[List[float], List[float]]:
    if not line:
        return [], []
    line = _normalize_line(line)
    lower = line.casefold()
    if "sem" in lower:
        idx = lower.find("sem")
        line = line[idx + 3 :]
    pairs = re.findall(r"(-?\d+(?:\.\d+)?)(?:\s*±\s*(-?\d+(?:\.\d+)?))?", line)
    means: List[float] = []
    sems: List[float] = []
    for mean, sem in pairs:
        if not mean:
            continue
        means.append(float(mean))
        sems.append(float(sem) if sem not in {None, ""} else float("nan"))
    return means, sems


def _parse_numbers(text: str) -> List[float]:
    if not text:
        return []
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _partition_by_means(values: List[float], means: List[float]) -> List[List[float]] | None:
    n = len(values)
    k = len(means)
    if n < k or k == 0:
        return None

    dp_cost = [[float("inf")] * (k + 1) for _ in range(n + 1)]
    dp_prev = [[None] * (k + 1) for _ in range(n + 1)]
    dp_cost[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, k + 1):
            for prev in range(j - 1, i):
                if dp_cost[prev][j - 1] == float("inf"):
                    continue
                group = values[prev:i]
                if not group:
                    continue
                mean = sum(group) / len(group)
                cost = dp_cost[prev][j - 1] + (mean - means[j - 1]) ** 2
                if cost < dp_cost[i][j]:
                    dp_cost[i][j] = cost
                    dp_prev[i][j] = prev

    if dp_cost[n][k] == float("inf"):
        return None

    groups: List[List[float]] = []
    i = n
    j = k
    while j > 0:
        prev = dp_prev[i][j]
        if prev is None:
            return None
        groups.append(values[prev:i])
        i = prev
        j -= 1
    return list(reversed(groups))


def _group_replicates(rep_text: str, n_conc: int, mean_vals: List[float]) -> Tuple[List[List[float]], str]:
    rep_text = _normalize_line(rep_text)
    rep_text = re.sub(r"\s+", " ", rep_text).strip()

    if mean_vals and len(mean_vals) == n_conc:
        values = _parse_numbers(rep_text)
        groups = _partition_by_means(values, mean_vals)
        if groups and len(groups) == n_conc:
            return groups, "dp_means"

    groups_raw = re.findall(r"-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+", rep_text)
    if len(groups_raw) == n_conc:
        return [_parse_numbers(g) for g in groups_raw], "regex"

    values = _parse_numbers(rep_text)
    if n_conc and values:
        base = max(1, len(values) // n_conc)
        groups: List[List[float]] = []
        idx = 0
        for _ in range(n_conc):
            size = base
            if idx + size > len(values):
                size = max(1, len(values) - idx)
            groups.append(values[idx : idx + size])
            idx += size
        return groups, "fallback_equal"

    return [[] for _ in range(n_conc)], "empty"


def parse_text(input_path: Path, enable_identity_alias: bool = False) -> Tuple[pd.DataFrame, dict]:
    input_path = _resolve_input_path(input_path)
    raw_text = input_path.read_text(encoding="utf-8", errors="ignore")
    raw_lines = [line.rstrip() for line in raw_text.splitlines()]
    norm_lines = [_normalize_line(line).strip() for line in raw_lines]

    current_indices = [
        idx for idx, line in enumerate(norm_lines) if line.casefold().startswith("current")
    ]
    if not current_indices:
        raise ValueError("No 'current' lines found in crumb extraction text.")

    retrieved_at_utc = datetime.now(timezone.utc).isoformat()
    rows: List[dict] = []
    warnings: List[ParseWarning] = []
    drug_channel_counts: Dict[str, Dict[str, int]] = {}
    concentration_counts_by_drug: Dict[str, int] = {}

    for idx, current_idx in enumerate(current_indices):
        end_idx = current_indices[idx + 1] if idx + 1 < len(current_indices) else len(norm_lines)

        j = current_idx - 1
        drug_raw = None
        while j >= 0:
            if raw_lines[j].strip():
                drug_raw = raw_lines[j].strip()
                break
            j -= 1
        if not drug_raw:
            warnings.append(ParseWarning("<unknown>", None, "missing_drug_name", f"at line {current_idx}"))
            continue

        conc_uM, conc_raw, conc_unit = _parse_concentrations(norm_lines[current_idx])
        if not conc_uM:
            warnings.append(ParseWarning(drug_raw, None, "missing_concentration", norm_lines[current_idx]))
            continue

        concentration_counts_by_drug[drug_raw] = len(conc_uM)
        i = current_idx + 1
        drug_channel_counts.setdefault(drug_raw, {})

        while i < end_idx:
            line = norm_lines[i]
            if not line:
                i += 1
                continue

            channel, remainder = _extract_channel(line)
            if channel is None:
                i += 1
                continue

            rep_parts: List[str] = []
            if remainder:
                rep_parts.append(remainder)

            i += 1
            mean_line = ""
            while i < end_idx:
                next_line = norm_lines[i]
                if not next_line:
                    i += 1
                    continue
                next_channel, _ = _extract_channel(next_line)
                if next_channel is not None:
                    break

                if "sem" in next_line.casefold():
                    if "x" in next_line.casefold():
                        split = re.split(r"x\s*±\s*sem", next_line, flags=re.IGNORECASE)
                        rep_head = split[0].strip() if split else ""
                        mean_tail = split[1].strip() if len(split) > 1 else ""
                        if rep_head:
                            rep_parts.append(rep_head)
                        mean_line = f"X ± SEM {mean_tail}".strip()
                    else:
                        mean_line = next_line
                    i += 1
                    break

                rep_parts.append(next_line)
                i += 1

            rep_text = " ".join(rep_parts).strip()
            mean_vals, sem_vals = _parse_mean_sem(mean_line)
            groups, method = _group_replicates(rep_text, len(conc_uM), mean_vals)

            if method not in {"dp_means", "regex"}:
                warnings.append(ParseWarning(drug_raw, channel, "replicate_group_fallback", f"method={method}"))

            if len(groups) != len(conc_uM):
                warnings.append(
                    ParseWarning(
                        drug_raw,
                        channel,
                        "replicate_group_mismatch",
                        f"groups={len(groups)} conc={len(conc_uM)} method={method}",
                    )
                )

            if len(groups) < len(conc_uM):
                warnings.append(
                    ParseWarning(
                        drug_raw,
                        channel,
                        "channel_cells_padded",
                        f"groups={len(groups)} conc={len(conc_uM)}",
                    )
                )
                groups.extend([[] for _ in range(len(conc_uM) - len(groups))])

            norm = normalize_compound(drug_raw, enable_identity_alias=enable_identity_alias)
            for idx_conc, conc in enumerate(conc_uM):
                reps = groups[idx_conc] if idx_conc < len(groups) else []
                reps_arr = np.array(reps, dtype=float) if reps else np.array([])

                mean = float(np.mean(reps_arr)) if reps else float("nan")
                std = float(np.std(reps_arr, ddof=1)) if len(reps_arr) > 1 else float("nan")
                mean_reported = mean_vals[idx_conc] if idx_conc < len(mean_vals) else float("nan")
                sem_reported = sem_vals[idx_conc] if idx_conc < len(sem_vals) else float("nan")

                if np.isnan(mean) and np.isfinite(mean_reported):
                    mean = mean_reported

                rows.append(
                    {
                        "drug_name_raw": drug_raw,
                        "drug_name_parent": norm["drug_name_parent"],
                        "drug_name_normalized": norm["drug_name_normalized"],
                        "identity_alias_enabled": enable_identity_alias,
                        "identity_alias_applied": norm["identity_alias_applied"],
                        "channel": channel,
                        "concentration_raw": conc_raw[idx_conc] if idx_conc < len(conc_raw) else None,
                        "concentration_unit": conc_unit[idx_conc] if idx_conc < len(conc_unit) else None,
                        "concentration_uM": conc,
                        "block_pct_replicates": json.dumps(reps) if reps else None,
                        "block_pct_n": len(reps),
                        "block_pct_mean": mean,
                        "block_pct_std": std,
                        "block_pct_mean_reported": mean_reported,
                        "block_pct_sem_reported": sem_reported,
                        "source_name": "crumb2016_text",
                        "source_path": str(input_path),
                        "retrieved_at_utc": retrieved_at_utc,
                    }
                )

            drug_channel_counts[drug_raw][channel] = drug_channel_counts[drug_raw].get(channel, 0) + 1

    df = pd.DataFrame(rows)

    warn_counts: Dict[str, int] = {}
    for warning in warnings:
        warn_counts[warning.issue] = warn_counts.get(warning.issue, 0) + 1

    debug = {
        "dataset_id": DATASET_ID,
        "parsed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "drugs_parsed": sorted({row["drug_name_raw"] for row in rows}),
        "channels_parsed": sorted({row["channel"] for row in rows}),
        "per_drug_channel_counts": drug_channel_counts,
        "concentration_counts_by_drug": concentration_counts_by_drug,
        "warning_counts": warn_counts,
        "warnings": [warning.__dict__ for warning in warnings],
    }
    return df, debug


def _cipa_parent_set(enable_identity_alias: bool) -> List[str]:
    cipa = load_processed_dataset()
    return sorted(
        {
            normalize_compound(name, enable_identity_alias=enable_identity_alias)["drug_name_parent"]
            for name in cipa["drug_name"].dropna()
        }
    )


def _parsed_parent_set(df: pd.DataFrame, enable_identity_alias: bool) -> List[str]:
    if df.empty:
        return []
    parents = sorted(set(df["drug_name_parent"].dropna().tolist()))
    if not enable_identity_alias:
        return parents
    return sorted(
        {
            normalize_compound(name, enable_identity_alias=True)["drug_name_parent"]
            for name in parents
        }
    )


def _valid_channel_mask(df: pd.DataFrame) -> pd.Series:
    mask = df["block_pct_mean"].notna()
    if "block_pct_mean_reported" in df.columns:
        mask = mask | df["block_pct_mean_reported"].notna()
    return mask


def _coverage_block(df: pd.DataFrame, enable_identity_alias: bool) -> dict:
    cipa_parents = _cipa_parent_set(enable_identity_alias=enable_identity_alias)
    cipa_set = set(cipa_parents)

    work = df.copy()
    if enable_identity_alias and not work.empty:
        work["drug_name_parent"] = work["drug_name_parent"].apply(
            lambda x: normalize_compound(x, enable_identity_alias=True)["drug_name_parent"]
        )

    parsed_parents = _parsed_parent_set(work, enable_identity_alias=False)
    parsed_set = set(parsed_parents)

    intersection = sorted(cipa_set & parsed_set)
    missing_cipa = sorted(cipa_set - parsed_set)
    extra_parsed = sorted(parsed_set - cipa_set)

    valid_mask = _valid_channel_mask(work) if not work.empty else pd.Series([], dtype=bool)

    per_channel = []
    for channel in CHANNEL_ORDER:
        if work.empty:
            channel_parents_present: List[str] = []
        else:
            chan = work[(work["channel"] == channel) & valid_mask]
            channel_parents_present = sorted(set(chan["drug_name_parent"].dropna().tolist()))

        channel_set = set(channel_parents_present)
        coverage_count = len(channel_set & cipa_set)
        missing_parents = sorted(cipa_set - channel_set)

        per_channel.append(
            {
                "channel": channel,
                "channel_parents_present": channel_parents_present,
                "channel_coverage_count": coverage_count,
                "channel_missing_count": len(cipa_set) - coverage_count,
                "missing_cipa_parents": missing_parents,
            }
        )

    return {
        "identity_alias_enabled": enable_identity_alias,
        "cipa_parent_count": len(cipa_parents),
        "parsed_parent_count": len(parsed_parents),
        "intersection_count": len(intersection),
        "missing_cipa_count": len(missing_cipa),
        "extra_parsed_count": len(extra_parsed),
        "cipa_parents": cipa_parents,
        "parsed_parents": parsed_parents,
        "intersection_parents": intersection,
        "missing_cipa_parents": missing_cipa,
        "extra_parents": extra_parsed,
        "per_channel": per_channel,
    }


def _closest_parent_matches(
    missing_parents: List[str],
    parsed_parents: List[str],
    top_k: int = 5,
) -> Dict[str, List[dict]]:
    suggestions: Dict[str, List[dict]] = {}
    for missing in missing_parents:
        scored = []
        for candidate in parsed_parents:
            score = SequenceMatcher(None, missing, candidate).ratio()
            scored.append((score, candidate))
        scored.sort(reverse=True)
        suggestions[missing] = [
            {"candidate": candidate, "similarity": round(score, 4)}
            for score, candidate in scored[:top_k]
        ]
    return suggestions


def write_reports(df: pd.DataFrame, debug: dict, enable_identity_alias: bool) -> dict:
    coverage_alias_off = _coverage_block(df, enable_identity_alias=False)
    coverage_alias_on = _coverage_block(df, enable_identity_alias=True)

    rows_by_channel = {channel: int((df["channel"] == channel).sum()) for channel in CHANNEL_ORDER}
    missing_suggestions = _closest_parent_matches(
        coverage_alias_off["missing_cipa_parents"],
        coverage_alias_off["parsed_parents"],
        top_k=5,
    )

    summary = {
        "dataset_id": DATASET_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity_alias_enabled": enable_identity_alias,
        "channels": CHANNEL_ORDER,
        "rows_by_channel": rows_by_channel,
        "parse_warning_counts": debug.get("warning_counts", {}),
        "cipa_parent_count": coverage_alias_off["cipa_parent_count"],
        "cipa_parents": coverage_alias_off["cipa_parents"],
        "parsed_parents": coverage_alias_off["parsed_parents"],
        "intersection_parents": coverage_alias_off["intersection_parents"],
        "missing_cipa_parents": coverage_alias_off["missing_cipa_parents"],
        "extra_parents": coverage_alias_off["extra_parents"],
        "missing_cipa_parent_suggestions": missing_suggestions,
        "coverage_alias_off": coverage_alias_off["per_channel"],
        "coverage_alias_on": coverage_alias_on["per_channel"],
        "alias_off": coverage_alias_off,
        "alias_on": coverage_alias_on,
    }

    JOIN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    DEBUG_PATH.write_text(json.dumps(debug, indent=2) + "\n", encoding="utf-8")

    lines = []
    lines.append("# Crumb 2016 Text Extraction Coverage")
    lines.append("")
    lines.append(f"Dataset: {DATASET_ID}")
    lines.append("")
    lines.append("## Set comparison (alias OFF)")
    lines.append(f"- CiPA parents: {coverage_alias_off['cipa_parent_count']}")
    lines.append(f"- Parsed parents: {coverage_alias_off['parsed_parent_count']}")
    lines.append(f"- Intersection: {coverage_alias_off['intersection_count']}")
    lines.append(f"- Missing CiPA parents: {coverage_alias_off['missing_cipa_count']}")
    lines.append(f"- Extra parsed parents: {coverage_alias_off['extra_parsed_count']}")
    lines.append("")
    lines.append("## Coverage by channel (alias OFF)")
    for entry in coverage_alias_off["per_channel"]:
        lines.append(
            f"- {entry['channel']}: {entry['channel_coverage_count']} / {coverage_alias_off['cipa_parent_count']}"
        )
    lines.append("")
    lines.append("## Coverage by channel (alias ON)")
    for entry in coverage_alias_on["per_channel"]:
        lines.append(
            f"- {entry['channel']}: {entry['channel_coverage_count']} / {coverage_alias_on['cipa_parent_count']}"
        )
    lines.append("")
    lines.append("## Parse warning counts")
    if summary["parse_warning_counts"]:
        for key, val in summary["parse_warning_counts"].items():
            lines.append(f"- {key}: {val}")
    else:
        lines.append("- None")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "parse_text",
    "write_reports",
    "PROCESSED_PATH",
    "JOIN_SUMMARY_PATH",
    "DEBUG_PATH",
    "REPORT_PATH",
    "DEFAULT_INPUT",
]
