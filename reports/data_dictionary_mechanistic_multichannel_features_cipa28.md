# Data Dictionary: mechanistic_multichannel_features_cipa28

One row per CiPA parent drug; channels are stored as IC50 in µM.

Source priority:
1. ChEMBL strict IC50 '=' (µM-convertible).
2. ChEMBL relaxed IC50 (non '=' relations) if strict absent.
3. CiPA GitHub IC50 (only if any IC50 rows are present).
4. Crumb2016 IC50 with relation '=' (fitted from block curves).
5. Missing if no comparable IC50.

CiPA GitHub panel provides percent inhibition only in the current file;
no IC50 rows were observed, so it is not used for IC50 selection.

Columns:
- drug_name_parent: normalized parent compound name.
- Column naming policy: channel names are used verbatim in column suffixes (e.g., ic50_uM_Nav1.5_peak).
- ic50_uM_hERG: selected IC50 in µM for hERG.
- source_hERG: data source used (chembl_strict, chembl_relaxed, crumb2016, missing).
- n_records_hERG: number of activity rows used in the aggregate.
- is_strict_hERG: True only if strict ChEMBL row used.
- ic50_uM_Nav1.5_peak: selected IC50 in µM for Nav1.5_peak.
- source_Nav1.5_peak: data source used (chembl_strict, chembl_relaxed, crumb2016, missing).
- n_records_Nav1.5_peak: number of activity rows used in the aggregate.
- is_strict_Nav1.5_peak: True only if strict ChEMBL row used.
- ic50_uM_Nav1.5_late: selected IC50 in µM for Nav1.5_late.
- source_Nav1.5_late: data source used (chembl_strict, chembl_relaxed, crumb2016, missing).
- n_records_Nav1.5_late: number of activity rows used in the aggregate.
- is_strict_Nav1.5_late: True only if strict ChEMBL row used.
- ic50_uM_Cav1.2: selected IC50 in µM for Cav1.2.
- source_Cav1.2: data source used (chembl_strict, chembl_relaxed, crumb2016, missing).
- n_records_Cav1.2: number of activity rows used in the aggregate.
- is_strict_Cav1.2: True only if strict ChEMBL row used.
- ic50_uM_IKs: selected IC50 in µM for IKs.
- source_IKs: data source used (chembl_strict, chembl_relaxed, crumb2016, missing).
- n_records_IKs: number of activity rows used in the aggregate.
- is_strict_IKs: True only if strict ChEMBL row used.
- ic50_uM_IK1: selected IC50 in µM for IK1.
- source_IK1: data source used (chembl_strict, chembl_relaxed, crumb2016, missing).
- n_records_IK1: number of activity rows used in the aggregate.
- is_strict_IK1: True only if strict ChEMBL row used.
- ic50_uM_Kv4.3: selected IC50 in µM for Kv4.3.
- source_Kv4.3: data source used (chembl_strict, chembl_relaxed, crumb2016, missing).
- n_records_Kv4.3: number of activity rows used in the aggregate.
- is_strict_Kv4.3: True only if strict ChEMBL row used.
- n_channels_present: number of channels with IC50 values present.
- n_channels_missing: number of channels missing IC50 values.
- missing_channels_list: pipe-delimited list of missing channels.

Identity aliasing is disabled for feature construction. Coverage with aliasing
enabled is reported in the join summary as a secondary diagnostic.
