"""Compound name normalization utilities shared across datasets."""

from __future__ import annotations

import re
from typing import Dict, List

# Salt/counter-ion tokens removed when forming parent compound names.
SALT_TOKENS = {
    "hydrochloride",
    "hydrobromide",
    "hydroiodide",
    "hemifumarate",
    "fumarate",
    "maleate",
    "tartrate",
    "succinate",
    "mesylate",
    "mesilate",
    "besylate",
    "besilate",
    "tosylate",
    "phosphate",
    "sulfate",
    "sulphate",
    "acetate",
    "citrate",
    "nitrate",
    "oxalate",
    "carbonate",
    "bromide",
    "chloride",
    "iodide",
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "hydrate",
    "monohydrate",
    "dihydrate",
    "trihydrate",
    "hemihydrate",
    "sesquihydrate",
    "anhydrous",
    "hcl",
    "hbr",
}

# Simple alias rules for obvious typos (applied after normalization).
TYPO_ALIAS_MAP = {
    "diltizem": "diltiazem",
}

# Identity-changing aliases (disabled by default; must be explicitly enabled).
IDENTITY_ALIAS_MAP = {
    "dlsotalol": "sotalol",
}

IDENTITY_ALIAS_REASON = {
    "dlsotalol": 'ChEMBL synonym "DL-SOTALOL"; stereoisomer mixture alias',
}
# Non-name tokens removed when forming parent compound names.
DROP_TOKENS = {
    "blinded",
}


def _tokenize(name: str) -> list[str]:
    tokens = re.split(r"[^a-z0-9]+", name.casefold())
    return [t for t in tokens if t]


def normalize_compound(name: str, enable_identity_alias: bool = False) -> Dict[str, object]:
    """Normalize compound names consistently.

    Returns a dict with:
    - drug_name_raw: original string, stripped.
    - drug_name_normalized: casefolded, punctuation removed.
    - drug_name_parent: normalized name with salt/counter-ion tokens removed.
    - identity_alias_applied: bool flag for identity-changing alias usage.
    - identity_alias_reason: list of applied identity alias reasons.
    """
    raw = str(name).strip()
    normalized = re.sub(r"[^a-z0-9]+", "", raw.casefold())

    tokens = _tokenize(raw)
    parent_tokens = [t for t in tokens if t not in SALT_TOKENS and t not in DROP_TOKENS]
    if not parent_tokens:
        parent_tokens = tokens
    parent = "".join(parent_tokens)

    # Apply alias mapping for obvious typos (always on).
    parent = TYPO_ALIAS_MAP.get(parent, parent)
    normalized = TYPO_ALIAS_MAP.get(normalized, normalized)

    identity_alias_applied = False
    identity_alias_reasons: List[str] = []
    if enable_identity_alias:
        if parent in IDENTITY_ALIAS_MAP:
            new_parent = IDENTITY_ALIAS_MAP[parent]
            if new_parent != parent:
                identity_alias_applied = True
                reason = IDENTITY_ALIAS_REASON.get(parent)
                msg = f"{parent} -> {new_parent}"
                if reason:
                    msg = f"{msg} ({reason})"
                identity_alias_reasons.append(msg)
                parent = new_parent
        if normalized in IDENTITY_ALIAS_MAP:
            new_norm = IDENTITY_ALIAS_MAP[normalized]
            if new_norm != normalized:
                identity_alias_applied = True
                reason = IDENTITY_ALIAS_REASON.get(normalized)
                msg = f"{normalized} -> {new_norm}"
                if reason:
                    msg = f"{msg} ({reason})"
                if msg not in identity_alias_reasons:
                    identity_alias_reasons.append(msg)
                normalized = new_norm

    return {
        "drug_name_raw": raw,
        "drug_name_normalized": normalized,
        "drug_name_parent": parent,
        "identity_alias_applied": identity_alias_applied,
        "identity_alias_reason": identity_alias_reasons,
    }
