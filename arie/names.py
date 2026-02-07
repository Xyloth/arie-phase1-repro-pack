"""Compound name normalization utilities shared across datasets."""

from __future__ import annotations

import re
from typing import Dict

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
ALIAS_MAP = {
    "diltizem": "diltiazem",
    "dlsotalol": "sotalol",
}
# Non-name tokens removed when forming parent compound names.
DROP_TOKENS = {
    "blinded",
}


def _tokenize(name: str) -> list[str]:
    tokens = re.split(r"[^a-z0-9]+", name.casefold())
    return [t for t in tokens if t]


def normalize_compound(name: str) -> Dict[str, str]:
    """Normalize compound names consistently.

    Returns a dict with:
    - drug_name_raw: original string, stripped.
    - drug_name_normalized: casefolded, punctuation removed.
    - drug_name_parent: normalized name with salt/counter-ion tokens removed.
    """
    raw = str(name).strip()
    normalized = re.sub(r"[^a-z0-9]+", "", raw.casefold())

    tokens = _tokenize(raw)
    parent_tokens = [t for t in tokens if t not in SALT_TOKENS and t not in DROP_TOKENS]
    if not parent_tokens:
        parent_tokens = tokens
    parent = "".join(parent_tokens)

    # Apply alias mapping for obvious typos.
    parent = ALIAS_MAP.get(parent, parent)
    normalized = ALIAS_MAP.get(normalized, normalized)

    return {
        "drug_name_raw": raw,
        "drug_name_normalized": normalized,
        "drug_name_parent": parent,
    }
