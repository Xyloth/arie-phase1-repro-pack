# Data Dictionary: Mechanistic hERG Multi-lab 2025 (OSF, processed)

Source: OSF project a6k5t ("Alvarez Baron et al. HESI BAA manual patch clamp hERG data from five laboratories").
We aggregate per-drug IC50 and Hill coefficient values derived from the `ic50nh` sheet
within each `concentration-inhibition.xlsx` file (per lab, per drug).

Unit of observation: one row per drug (aggregated across labs).

Normalization:
- Parent compound names are produced by shared normalization rules (casefold + punctuation removal + salt stripping).
- Alias rules (applied after normalization) handle obvious typos and stereochemistry variants:
  - diltizem -> diltiazem
  - dlsotalol -> sotalol (supported by ChEMBL synonym "DL-SOTALOL")

Columns:
- drug_name_raw: Drug name as reported in the OSF tables (mode across labs).
- drug_name_normalized: Normalized drug name (casefolded, alphanumeric only) used for joins.
- drug_name_parent: Normalized name with salt/counter-ion tokens removed (plus alias rules).
- herg_ic50_uM_mean: Mean IC50 in µM across labs.
- herg_ic50_uM_std: Standard deviation of IC50 in µM across labs.
- herg_ic50_uM_median: Median IC50 in µM across labs.
- herg_ic50_uM_min: Minimum IC50 in µM across labs.
- herg_ic50_uM_max: Maximum IC50 in µM across labs.
- herg_ic50_uM_count: Count of non-null IC50 values used for aggregation.
- herg_nh_mean: Mean Hill coefficient across labs.
- herg_nh_std: Standard deviation of Hill coefficient across labs.
- herg_nh_median: Median Hill coefficient across labs.
- herg_nh_min: Minimum Hill coefficient across labs.
- herg_nh_max: Maximum Hill coefficient across labs.
- herg_nh_count: Count of non-null Hill coefficient values used for aggregation.
- labs_n: Number of labs contributing data for the drug.
- mechanistic_source: Source label for the row ("OSF" or "ChEMBL" when gap-fill is enabled).
