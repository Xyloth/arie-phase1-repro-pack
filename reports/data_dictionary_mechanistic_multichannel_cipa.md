# Data Dictionary: Mechanistic Multi-channel CiPA (processed)

Source: FDA CiPA GitHub repository (Hill_fitting/data/mergedpatchclampdata-20160514.csv).
Primary reference: Crumb et al. 2016 (CiPA ion channel panel).

Normalization:
- Compound names are normalized with shared rules (casefold + punctuation removal + salt stripping).
- Identity-changing aliases are disabled by default (see `normalize_compound(..., enable_identity_alias=False)`).

Channel mapping (`channel_key`):
- Calcium -> Cav1.2
- Peak sodium -> Nav1.5_peak
- Late sodium -> Nav1.5_late
- hERG -> hERG
- IKs -> IKs
- IK1 -> IK1
- Kv4.3 -> Kv4.3

Measurements:
- Each row represents a single patch-clamp measurement.
- `metric` is `pct_inhibition` with `value` from the `block` column.
- `units` is recorded as `percent` (the source does not explicitly state units for `block`).

Concentration:
- `concentration_raw` and `concentration_unit` are taken from `Conc` and `Units`.
- `concentration_uM` is derived from `concentration_raw` using `Units` (nM -> µM, µM -> µM, mM -> µM).

Columns:
- drug_name_raw, drug_name_normalized, drug_name_parent
- channel_key
- metric, value, units
- concentration_raw, concentration_unit, concentration_uM
- mechanistic_source, lab_or_provider, n_measurements, provenance_note
