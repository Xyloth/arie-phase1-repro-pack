# Reproducibility Appendix (Phase 1)

## Environment
- Python: 3.10
- Install:

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## One-command reproduction

```bash
./.venv/bin/python scripts/repro_pack.py
```

## Equivalent step-by-step commands

```bash
./.venv/bin/python scripts/process_mechanistic_multichannel_crumb_text.py --force
./.venv/bin/python scripts/fetch_mechanistic_multichannel_chembl.py --diagnose
./.venv/bin/python scripts/build_mechanistic_multichannel_features.py --force --print-summary
./.venv/bin/python scripts/score_mech_plausibility.py
./.venv/bin/python scripts/run_trust_policy.py
./.venv/bin/python scripts/run_trust_policy_mech.py
```

## Expected outputs
- `data/processed/mechanistic_multichannel_features_cipa28.csv`
- `results/mechanistic_multichannel_feature_join_summary.json`
- `results/mechanistic_multichannel_concordance.json`
- `reports/mechanistic_multichannel_concordance.md`
- `results/mechanistic_multichannel_crumb_text_join_summary.json`
- `results/mech_plausibility_scores.csv`
- `results/mech_plausibility_summary.json`
- `results/abstention_trust_policy_curve.csv`
- `results/abstention_trust_policy_summary.json`
- `results/abstention_trust_policy_mech_curve.csv`
- `results/abstention_trust_policy_mech_summary.json`
- `results/trust_policy_scores.csv`
- `results/trust_policy_scores_with_mech.csv`
- `results/repro_manifest.json`

## Public repo notes
- Raw files and caches are not committed (`data/raw/`, `data/cache/`).
- Generated artifacts are not committed (`results/`, `data/processed/`, `figures/`).
- The reproducibility manifest stores hashes and row counts for required outputs.
- During `scripts/run_trust_policy_mech.py`, scikit-learn may emit
  `y_pred contains classes not in y_true` on some fold/coverage subsets where a
  class is absent after abstention; this is an expected metric warning, not a
  pipeline failure.

## Data provenance
- Crumb multichannel panel: parsed from local text extraction input file (`raw/crumb_extraction.txt`) via `scripts/process_mechanistic_multichannel_crumb_text.py`.
- ChEMBL multichannel records: pulled via `scripts/fetch_mechanistic_multichannel_chembl.py` and cached under `data/cache/`.
- Canonical feature table is rebuilt deterministically by `scripts/build_mechanistic_multichannel_features.py` using source-priority rules documented in `reports/data_dictionary_mechanistic_multichannel_features_cipa28.md`.
