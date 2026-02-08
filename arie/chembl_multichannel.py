"""ChEMBL multi-channel (CiPA panel) ingestion helpers."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests

from arie.chembl import get_release_info, resolve_compound
from arie.data import load_processed_dataset
from arie.names import normalize_compound

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "cache" / "chembl_multichannel"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

RESOLUTION_CACHE_PATH = CACHE_ROOT / "molecule_resolutions.json"
RUN_META_PATH = CACHE_ROOT / "run_metadata.json"

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

CHANNEL_TARGETS = {
    "hERG": [
        {"chembl_target_id": "CHEMBL240", "target_name": "KCNH2"},
    ],
    "Cav1.2": [
        {"chembl_target_id": "CHEMBL1940", "target_name": "CACNA1C"},
    ],
    "Nav1.5_peak": [
        {"chembl_target_id": "CHEMBL1980", "target_name": "SCN5A"},
    ],
    "Nav1.5_late": [
        {
            "chembl_target_id": "CHEMBL1980",
            "target_name": "SCN5A",
            "note": "ChEMBL does not distinguish late vs peak; SCN5A used for both.",
        },
    ],
    "IKs": [
        {"chembl_target_id": "CHEMBL1866", "target_name": "KCNQ1"},
        {"chembl_target_id": "CHEMBL4872", "target_name": "KCNE1"},
    ],
    "IK1": [
        {"chembl_target_id": "CHEMBL1914276", "target_name": "KCNJ2"},
    ],
    "Kv4.3": [
        {"chembl_target_id": "CHEMBL1964", "target_name": "KCND3"},
    ],
}

CHANNEL_ORDER = [
    "hERG",
    "Cav1.2",
    "Nav1.5_peak",
    "Nav1.5_late",
    "IKs",
    "IK1",
    "Kv4.3",
]

RELAXED_TYPES = {"IC50", "Ki"}
RELAXED_RELATIONS = {"=", "<", "<=", ">", ">=", "~"}


def _get_json(url: str, params: dict | None = None) -> dict:
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _to_um(value: float, units: str) -> float | None:
    units = (units or "").strip().lower()
    if units in {"nm", "nanomolar"}:
        return value / 1000.0
    if units in {"um", "µm", "micromolar"}:
        return value
    if units in {"mm", "millimolar"}:
        return value * 1000.0
    if units in {"pm", "picomolar"}:
        return value / 1_000_000.0
    return None


def _cache_path_for(molecule_id: str, target_id: str, suffix: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{molecule_id}_{target_id}_{suffix}")
    return CACHE_ROOT / f"{safe}.json"


def _load_resolution_cache() -> dict:
    if RESOLUTION_CACHE_PATH.exists():
        return json.loads(RESOLUTION_CACHE_PATH.read_text())
    return {"alias_off": {}, "alias_on": {}}


def _save_resolution_cache(cache: dict) -> None:
    RESOLUTION_CACHE_PATH.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")


def _resolve_molecules(
    parents: Iterable[str],
    enable_identity_alias: bool,
    force: bool = False,
) -> Dict[str, dict]:
    cache = _load_resolution_cache()
    key = "alias_on" if enable_identity_alias else "alias_off"
    resolved = cache.get(key, {})

    updated = False
    for parent in parents:
        if parent in resolved and not force:
            continue
        result = resolve_compound(parent, enable_identity_alias=enable_identity_alias)
        if result is None:
            resolved[parent] = {
                "resolved": False,
                "query_name": parent,
                "query_parent": parent,
                "molecule_chembl_id": None,
                "resolved_name": None,
                "resolution_method": None,
            }
        else:
            resolved[parent] = {
                "resolved": True,
                "query_name": parent,
                "query_parent": parent,
                "molecule_chembl_id": result.get("molecule_chembl_id"),
                "resolved_name": result.get("resolved_name"),
                "resolution_method": result.get("resolution_method"),
                "score": result.get("score"),
            }
        updated = True

    if updated:
        cache[key] = resolved
        _save_resolution_cache(cache)
    return resolved


def _fetch_activities(
    molecule_id: str,
    target_id: str,
    release: dict,
    force: bool = False,
) -> dict:
    cache_path = _cache_path_for(molecule_id, target_id, "IC50_eq")
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    activities: List[dict] = []
    limit = 200
    offset = 0
    while True:
        params = {
            "target_chembl_id": target_id,
            "molecule_chembl_id": molecule_id,
            "standard_type": "IC50",
            "standard_relation": "=",
            "format": "json",
            "limit": limit,
            "offset": offset,
        }
        payload = _get_json(f"{CHEMBL_BASE}/activity", params=params)
        batch = payload.get("activities", [])
        activities.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    blob = {
        "query": {
            "target_chembl_id": target_id,
            "molecule_chembl_id": molecule_id,
            "standard_type": "IC50",
            "standard_relation": "=",
        },
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "chembl_release": release.get("chembl_db_version"),
        "activities": activities,
    }
    cache_path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    return blob


def _fetch_activities_all(
    molecule_id: str,
    target_id: str,
    release: dict,
    force: bool = False,
) -> dict:
    cache_path = _cache_path_for(molecule_id, target_id, "all")
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    activities: List[dict] = []
    limit = 200
    offset = 0
    while True:
        params = {
            "target_chembl_id": target_id,
            "molecule_chembl_id": molecule_id,
            "format": "json",
            "limit": limit,
            "offset": offset,
        }
        payload = _get_json(f"{CHEMBL_BASE}/activity", params=params)
        batch = payload.get("activities", [])
        activities.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    blob = {
        "query": {
            "target_chembl_id": target_id,
            "molecule_chembl_id": molecule_id,
        },
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "chembl_release": release.get("chembl_db_version"),
        "activities": activities,
    }
    cache_path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    return blob


def build_target_map() -> dict:
    mapping = {}
    for channel in CHANNEL_ORDER:
        targets = CHANNEL_TARGETS.get(channel, [])
        mapping[channel] = {
            "gene_symbols": sorted({t.get("target_name") for t in targets if t.get("target_name")}),
            "search_method": "target_chembl_id (direct)",
            "chembl_target_ids": [t.get("chembl_target_id") for t in targets],
            "notes": [t.get("note") for t in targets if t.get("note")],
        }
    return mapping


def fetch_multichannel_chembl(
    enable_identity_alias: bool = False,
    force: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    cipa = load_processed_dataset()
    cipa_parents = sorted(
        {
            normalize_compound(name, enable_identity_alias=False)["drug_name_parent"]
            for name in cipa["drug_name"].dropna()
        }
    )

    release = get_release_info()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    resolutions = _resolve_molecules(
        cipa_parents,
        enable_identity_alias=enable_identity_alias,
        force=force,
    )

    rows: List[dict] = []
    for parent in cipa_parents:
        resolved = resolutions.get(parent, {})
        if not resolved or not resolved.get("resolved"):
            continue
        molecule_id = resolved.get("molecule_chembl_id")
        if not molecule_id:
            continue
        for channel in CHANNEL_ORDER:
            targets = CHANNEL_TARGETS.get(channel, [])
            for target in targets:
                target_id = target["chembl_target_id"]
                cache_blob = _fetch_activities(
                    molecule_id=molecule_id,
                    target_id=target_id,
                    release=release,
                    force=force,
                )
                for act in cache_blob.get("activities", []):
                    if act.get("standard_type") != "IC50":
                        continue
                    if act.get("standard_relation") != "=":
                        continue
                    try:
                        value = float(act.get("standard_value"))
                    except (TypeError, ValueError):
                        continue
                    units = act.get("standard_units")
                    value_um = _to_um(value, units)
                    if value_um is None or math.isnan(value_um):
                        continue
                    rows.append(
                        {
                            "drug_name_parent": parent,
                            "target_channel": channel,
                            "chembl_target_id": target_id,
                            "standard_type": act.get("standard_type"),
                            "standard_relation": act.get("standard_relation"),
                            "standard_value": value,
                            "standard_units": units,
                            "value_uM": value_um,
                            "assay_chembl_id": act.get("assay_chembl_id"),
                            "activity_chembl_id": act.get("activity_id") or act.get("activity_chembl_id"),
                            "source": "ChEMBL",
                            "retrieved_at_utc": cache_blob.get("retrieved_at_utc", retrieved_at),
                            "chembl_release": cache_blob.get("chembl_release", release.get("chembl_db_version")),
                            "identity_alias_enabled": enable_identity_alias,
                            "provenance_note": target.get("note") or "",
                        }
                    )

    df = pd.DataFrame(rows)

    run_meta = {
        "retrieved_at_utc": retrieved_at,
        "chembl_release": release.get("chembl_db_version"),
        "chembl_release_date": release.get("chembl_release_date"),
        "targets": CHANNEL_TARGETS,
        "identity_alias_enabled": enable_identity_alias,
        "rows": int(len(df)),
    }
    RUN_META_PATH.write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    return df, run_meta


def build_relaxed_dataset(
    enable_identity_alias: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    cipa = load_processed_dataset()
    cipa_parents = sorted(
        {
            normalize_compound(name, enable_identity_alias=False)["drug_name_parent"]
            for name in cipa["drug_name"].dropna()
        }
    )

    release = get_release_info()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    resolutions = _resolve_molecules(
        cipa_parents,
        enable_identity_alias=enable_identity_alias,
        force=force,
    )

    rows: List[dict] = []
    for parent in cipa_parents:
        resolved = resolutions.get(parent, {})
        if not resolved or not resolved.get("resolved"):
            continue
        molecule_id = resolved.get("molecule_chembl_id")
        if not molecule_id:
            continue
        for channel in CHANNEL_ORDER:
            targets = CHANNEL_TARGETS.get(channel, [])
            for target in targets:
                target_id = target["chembl_target_id"]
                cache_blob = _fetch_activities_all(
                    molecule_id=molecule_id,
                    target_id=target_id,
                    release=release,
                    force=force,
                )
                for act in cache_blob.get("activities", []):
                    standard_type = act.get("standard_type")
                    standard_relation = act.get("standard_relation")
                    if standard_type not in RELAXED_TYPES:
                        continue
                    if standard_relation not in RELAXED_RELATIONS:
                        continue
                    try:
                        value = float(act.get("standard_value"))
                    except (TypeError, ValueError):
                        continue
                    units = act.get("standard_units")
                    value_um = _to_um(value, units)
                    if value_um is None or math.isnan(value_um):
                        continue
                    is_strict = standard_type == "IC50" and standard_relation == "="
                    rows.append(
                        {
                            "drug_name_parent": parent,
                            "target_channel": channel,
                            "chembl_target_id": target_id,
                            "standard_type": standard_type,
                            "standard_relation": standard_relation,
                            "standard_value": value,
                            "standard_units": units,
                            "value_uM": value_um,
                            "assay_chembl_id": act.get("assay_chembl_id"),
                            "activity_chembl_id": act.get("activity_id") or act.get("activity_chembl_id"),
                            "source": "ChEMBL",
                            "retrieved_at_utc": cache_blob.get("retrieved_at_utc", retrieved_at),
                            "chembl_release": cache_blob.get("chembl_release", release.get("chembl_db_version")),
                            "identity_alias_enabled": enable_identity_alias,
                            "is_strict_row": is_strict,
                            "provenance_note": target.get("note") or "",
                        }
                    )
    return pd.DataFrame(rows)


def prefilter_diagnostics(
    enable_identity_alias: bool = False,
    force: bool = False,
) -> dict:
    cipa = load_processed_dataset()
    cipa_parents = sorted(
        {
            normalize_compound(name, enable_identity_alias=False)["drug_name_parent"]
            for name in cipa["drug_name"].dropna()
        }
    )
    release = get_release_info()
    resolutions = _resolve_molecules(
        cipa_parents,
        enable_identity_alias=enable_identity_alias,
        force=force,
    )

    diagnostics = {}
    for channel in CHANNEL_ORDER:
        counts_type: Dict[str, int] = {}
        counts_rel: Dict[str, int] = {}
        counts_units: Dict[str, int] = {}
        strict_count = 0
        total = 0

        targets = CHANNEL_TARGETS.get(channel, [])
        for parent in cipa_parents:
            resolved = resolutions.get(parent, {})
            if not resolved or not resolved.get("resolved"):
                continue
            molecule_id = resolved.get("molecule_chembl_id")
            if not molecule_id:
                continue
            for target in targets:
                target_id = target["chembl_target_id"]
                blob = _fetch_activities_all(
                    molecule_id=molecule_id,
                    target_id=target_id,
                    release=release,
                    force=force,
                )
                for act in blob.get("activities", []):
                    total += 1
                    stype = act.get("standard_type") or "UNKNOWN"
                    srel = act.get("standard_relation") or "UNKNOWN"
                    sunits = act.get("standard_units") or "UNKNOWN"
                    counts_type[stype] = counts_type.get(stype, 0) + 1
                    counts_rel[srel] = counts_rel.get(srel, 0) + 1
                    counts_units[sunits] = counts_units.get(sunits, 0) + 1

                    if stype == "IC50" and srel == "=":
                        try:
                            val = float(act.get("standard_value"))
                        except (TypeError, ValueError):
                            continue
                        if _to_um(val, sunits) is None:
                            continue
                        strict_count += 1

        diagnostics[channel] = {
            "total_activities": total,
            "by_standard_type": dict(sorted(counts_type.items(), key=lambda x: x[0])),
            "by_standard_relation": dict(sorted(counts_rel.items(), key=lambda x: x[0])),
            "by_standard_units": dict(sorted(counts_units.items(), key=lambda x: x[0])),
            "strict_count": strict_count,
        }

    return {
        "identity_alias_enabled": bool(enable_identity_alias),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "chembl_release": release.get("chembl_db_version"),
        "channels": diagnostics,
    }


def summarize_coverage(
    df: pd.DataFrame,
    enable_identity_alias: bool = False,
) -> dict:
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

    def _parent_set(series: pd.Series, alias: bool) -> List[str]:
        return sorted(
            {
                normalize_compound(name, enable_identity_alias=alias)["drug_name_parent"]
                for name in series.dropna().astype("string")
            }
        )

    coverage = []
    coverage_alias_on = []
    for channel in CHANNEL_ORDER:
        subset = df[df["target_channel"] == channel]
        parents_off = _parent_set(subset["drug_name_parent"], alias=False)
        parents_on = _parent_set(subset["drug_name_parent"], alias=True)

        covered_off = sorted(set(cipa_parents_off).intersection(set(parents_off)))
        covered_on = sorted(set(cipa_parents_on).intersection(set(parents_on)))

        coverage.append(
            {
                "target_channel": channel,
                "covered_parents": len(covered_off),
                "missing_parents": sorted(set(cipa_parents_off).difference(set(parents_off))),
                "activity_count": int(len(subset)),
            }
        )
        coverage_alias_on.append(
            {
                "target_channel": channel,
                "covered_parents": len(covered_on),
                "missing_parents": sorted(set(cipa_parents_on).difference(set(parents_on))),
                "activity_count": int(len(subset)),
            }
        )

    def _coverage_any(parent_set: List[str], alias: bool) -> int:
        present = set()
        for channel in CHANNEL_ORDER:
            subset = df[df["target_channel"] == channel]
            parents = _parent_set(subset["drug_name_parent"], alias=alias)
            present.update(parents)
        return len(set(parent_set).intersection(present))

    def _coverage_all(parent_set: List[str], alias: bool) -> int:
        covered = set(parent_set)
        for channel in CHANNEL_ORDER:
            subset = df[df["target_channel"] == channel]
            parents = set(_parent_set(subset["drug_name_parent"], alias=alias))
            covered = covered.intersection(parents)
        return len(covered)

    summary = {
        "identity_alias_enabled": bool(enable_identity_alias),
        "channels": CHANNEL_ORDER,
        "coverage_by_channel": coverage,
        "coverage_by_channel_identity": coverage_alias_on,
        "cipa_parent_count": int(len(cipa_parents_off)),
        "coverage_any_channel": _coverage_any(cipa_parents_off, alias=False),
        "coverage_all_channels": _coverage_all(cipa_parents_off, alias=False),
        "coverage_any_channel_identity": _coverage_any(cipa_parents_on, alias=True),
        "coverage_all_channels_identity": _coverage_all(cipa_parents_on, alias=True),
    }
    return summary
