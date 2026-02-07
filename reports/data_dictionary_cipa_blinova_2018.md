# Data Dictionary: CiPA Blinova 2018 (processed)

Unit of observation: one measurement row per drug × concentration level × platform × cell type × site (as recorded in the source file).

Columns:
- drug_name: Compound name (string).
- cell_type: Cell type label from source (e.g., CDI, AXG).
- risk_class: Risk class label from source (L/M/H). Missing values kept as-is.
- platform: Platform label from source (e.g., ACA, AXN, MCS, AMD, ECR, CLY).
- ead_type: EAD type code from source (e.g., A, B, C, D, Q). Missing values kept as-is.
- concentration_level: Concentration level as provided in the source file (integer 1–4). Units not specified in the file.
- ead: Early afterdepolarization flag from source (0/1).
- dd_fpdc: Numeric metric from source column ddFPDc. Units not specified in the file.
- site: Site identifier from source (integer).
