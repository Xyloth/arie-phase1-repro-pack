# Data Dictionary: Mechanistic Features hERG Multi-lab 2025

Source: Derived from `data/processed/mechanistic_herg_multilab_2025.csv` (OSF multi-lab hERG dataset with optional ChEMBL gap-fill).

Unit of observation: one row per parent compound (drug_name_parent).

Aggregation rules:
- For each parent compound, mechanistic sources present are recorded as a "+"-joined set (e.g., "OSF", "ChEMBL", "OSF+ChEMBL").
- Numeric features are taken from the preferred source (OSF if present; otherwise ChEMBL), preserving the aggregated summaries computed in the mechanistic table.

Columns:
- drug_name_parent: Normalized parent compound name (casefold + punctuation removal + salt stripping + alias rules).
- mechanistic_sources_present: Source set for the compound ("OSF", "ChEMBL", or "OSF+ChEMBL").
- herg_ic50_uM_mean: Mean IC50 in µM.
- herg_ic50_uM_median: Median IC50 in µM.
- herg_ic50_uM_std: Standard deviation of IC50 in µM.
- herg_ic50_uM_min: Minimum IC50 in µM.
- herg_ic50_uM_max: Maximum IC50 in µM.
- herg_ic50_uM_count: Count of IC50 values used for aggregation.
- herg_nh_mean: Mean Hill coefficient.
- herg_nh_median: Median Hill coefficient.
- herg_nh_std: Standard deviation of Hill coefficient.
- herg_nh_min: Minimum Hill coefficient.
- herg_nh_max: Maximum Hill coefficient.
- herg_nh_count: Count of Hill coefficient values used for aggregation.
- ic50_count_osf: Count of IC50 measurements from OSF (0 if OSF not present).
- ic50_count_chembl: Count of IC50 activities from ChEMBL (0 if ChEMBL not present).
- nh_count_osf: Count of Hill coefficient measurements from OSF (0 if OSF not present).
- nh_count_chembl: Count of Hill coefficient measurements from ChEMBL (0 if ChEMBL not present).
