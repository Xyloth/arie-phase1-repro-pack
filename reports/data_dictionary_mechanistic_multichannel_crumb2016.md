# Data Dictionary: Crumb 2016 CiPA Panel (processed)

Source: Crumb et al. 2016 CiPA ion-channel panel supplement PDF.

Unit of observation: one row per drug × channel.

Channels:
- hERG
- Nav1.5_peak
- Nav1.5_late
- Cav1.2
- IKs (KCNQ1/KCNE1; labeled KvLQT1/minK in the supplement)
- IK1 (Kir2.1)
- Kv4.3 (Ito)

Parsing assumptions:
- Concentration-response means are extracted from the “X ± SEM” rows.
- IC50 and Hill n are estimated by fitting a Hill equation to mean % block vs concentration.
- If maximum block < 50% across tested concentrations, IC50 is treated as censored with relation “>”
  and numeric value equal to the max tested concentration.
- If fitting fails, IC50 and Hill n are set to NaN and relation “NA”.

Censored values:
- `ic50_relation` indicates “=”, “>”, “<”, or “NA”.

Provenance:
- `source_url`, `retrieved_at_utc`, `source_sha256`, and `parser_version` are recorded.

Block at Cmax:
- `block_free_cmax_pct` and `block_3x_free_cmax_pct` are not available in the supplement and are NaN.
