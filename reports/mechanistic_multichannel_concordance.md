# Multichannel IC50 Concordance

This report compares IC50_uM values across sources where both provide data.

## chembl_strict_vs_crumb_text

Pairs compared: 23
Per-channel overlap counts: {'hERG': 9, 'Nav1.5_peak': 5, 'Nav1.5_late': 5, 'Cav1.2': 4, 'IKs': 0, 'IK1': 0, 'Kv4.3': 0}
Median abs log10(IC50 ratio): 0.42252804702818664
Mean abs log10(IC50 ratio): 0.99197101093374
Spearman corr log10(IC50): 0.24404788926093965

Top mismatches (abs log10 ratio):

drug_name_parent | channel | abs_log10_ratio
---|---|---
dofetilide | Cav1.2 | 4.9932
dofetilide | Nav1.5_late | 4.8973
dofetilide | Nav1.5_peak | 4.8973
ranolazine | Nav1.5_peak | 0.9938
dofetilide | hERG | 0.9544
terfenadine | hERG | 0.8696
ranolazine | Nav1.5_late | 0.7585
ranolazine | Cav1.2 | 0.6539
quinidine | Cav1.2 | 0.5647
cisapride | hERG | 0.5296


## chembl_vs_cipa_repo

Pairs compared: 0
Median fold-diff: None
Mean fold-diff: None
Max fold-diff: None

No overlapping IC50 pairs for this comparison.

## chembl_vs_crumb2016

Pairs compared: 0
Median fold-diff: None
Mean fold-diff: None
Max fold-diff: None

No overlapping IC50 pairs for this comparison.

## cipa_repo_vs_crumb2016

Pairs compared: 0
Median fold-diff: None
Mean fold-diff: None
Max fold-diff: None

No overlapping IC50 pairs for this comparison.

