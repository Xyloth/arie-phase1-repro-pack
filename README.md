# ARIE

## Data
We start with the CiPA Myocyte Validation Study dataset (Blinova et al., 2018) from the CiPA project website.

Run this from repo root (inside the venv) to download and cache the raw file:

```
./.venv/bin/python scripts/download_cipa_data.py
```

The raw file is stored under `data/raw/cipa_blinova_2018/` and cache metadata under `data/cache/`.

Source: https://cipaproject.org/data-resources/

## Mechanistic Data
We use a CiPA-relevant mechanistic dataset: multi-laboratory manual patch clamp hERG
IC50 and Hill coefficient measurements from the OSF repository referenced by the
open-access Scientific Reports study (Alvarez-Baron et al., 2025). This provides
mechanistic potency features grounded in ion-channel pharmacology.

Download + process from repo root (inside the venv):

```
./.venv/bin/python scripts/download_mechanistic_data.py
./.venv/bin/python scripts/process_mechanistic_data.py
```

Raw files are stored under `data/raw/herg_multilab_2025/`, processed features under
`data/processed/mechanistic_herg_multilab_2025.csv`, and cache metadata under `data/cache/`.

Source article:
https://www.nature.com/articles/s41598-025-15761-8

OSF dataset:
https://osf.io/a6k5t/

ChEMBL gap-fill (target KCNH2 / CHEMBL240) for missing compounds:
License: CC BY-SA 3.0. Release: ChEMBL_36 (2025-07-28).
