"""ChEMBL REST API helpers for targeted hERG (KCNH2) gap-fill."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import requests

from arie.names import normalize_compound

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
TARGET_KCNH2 = "CHEMBL240"


def _get_json(url: str, params: dict | None = None) -> dict:
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def get_release_info() -> dict:
    return _get_json(f"{CHEMBL_BASE}/status", params={"format": "json"})


def _match_name(candidate_name: str, query_parent: str, enable_identity_alias: bool = False) -> bool:
    parent = normalize_compound(candidate_name, enable_identity_alias=enable_identity_alias).get(
        "drug_name_parent"
    )
    return parent == query_parent


def resolve_compound(query_name: str, enable_identity_alias: bool = False) -> Optional[dict]:
    query_norm = normalize_compound(query_name, enable_identity_alias=enable_identity_alias)
    query_parent = query_norm["drug_name_parent"]

    payload = _get_json(
        f"{CHEMBL_BASE}/molecule/search",
        params={"q": query_name, "format": "json"},
    )
    candidates = payload.get("molecules", [])

    # Prefer exact parent match on pref_name.
    for cand in candidates:
        pref = cand.get("pref_name") or ""
        if pref and _match_name(pref, query_parent, enable_identity_alias=enable_identity_alias):
            return {
                "query_name": query_name,
                "query_parent": query_parent,
                "molecule_chembl_id": cand.get("molecule_chembl_id"),
                "resolved_name": pref,
                "resolution_method": "pref_name",
                "score": cand.get("score"),
            }

    # Fall back to synonym matches.
    for cand in candidates:
        for syn in cand.get("molecule_synonyms", []) or []:
            syn_name = syn.get("molecule_synonym") or syn.get("synonyms") or ""
            if syn_name and _match_name(syn_name, query_parent, enable_identity_alias=enable_identity_alias):
                return {
                    "query_name": query_name,
                    "query_parent": query_parent,
                    "molecule_chembl_id": cand.get("molecule_chembl_id"),
                    "resolved_name": syn_name,
                    "resolution_method": "synonym",
                    "score": cand.get("score"),
                }

    return None


def _to_um(value: float, units: str) -> Optional[float]:
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


def fetch_kcnh2_activities(chembl_id: str) -> List[float]:
    values_um: List[float] = []
    limit = 200
    offset = 0
    while True:
        params = {
            "target_chembl_id": TARGET_KCNH2,
            "molecule_chembl_id": chembl_id,
            "standard_type": "IC50",
            "standard_relation": "=",
            "format": "json",
            "limit": limit,
            "offset": offset,
        }
        payload = _get_json(f"{CHEMBL_BASE}/activity", params=params)
        activities = payload.get("activities", [])
        for act in activities:
            try:
                val = float(act.get("standard_value"))
            except (TypeError, ValueError):
                continue
            units = act.get("standard_units")
            um = _to_um(val, units)
            if um is None or math.isnan(um):
                continue
            values_um.append(um)
        if len(activities) < limit:
            break
        offset += limit
    return values_um


def summarize_values(values_um: List[float]) -> dict:
    if not values_um:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "iqr": None,
            "min": None,
            "max": None,
        }
    arr = np.array(values_um, dtype=float)
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
