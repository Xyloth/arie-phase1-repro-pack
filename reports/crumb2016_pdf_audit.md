# Crumb2016 PDF Extraction Audit

Generated: 2026-02-14T13:19:09.229901+00:00

## PDFs Audited
- /mnt/c/ARIE/data/raw/mechanistic_multichannel_crumb2016/Crumb2016_supplement.pdf
  size_bytes: 996183
  sha256: 6a10abf62775f2dd6e0ad5b4897010c813496d58da4da97042f8e53dcfd65683
  pages: 65
  camelot: camelot not available

## Text-Layer Search Summary
- Crumb2016_supplement.pdf: found 10 / 28 drugs
  missing_drugs_text_layer: 18
  found_drugs_text_layer: bepridil, chlorpromazine, cisapride, diltiazem, dofetilide, mexiletine, quinidine, ranolazine, terfenadine, verapamil
  missing_drugs_text_layer_list: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, ondansetron, pimozide, risperidone, tamoxifen, vandetanib
  channels_found_text_layer: Cav1.2, Ito, Kir2.1, Kv4.3, Nav1.5, hERG
  channels_missing_text_layer: IKs, IK1, KCNQ1

## Extracted Drug Names
- raw unique: 48
- parents (alias off): 48
- parents (alias on): 48
- parents intersect CiPA (alias off): 10
- parents missing from CiPA (alias off): 18

Missing CiPA parents (alias off):
astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, ondansetron, pimozide, risperidone, tamoxifen, vandetanib

## Coverage by Channel (from extracted tables)
- Nav1.5: present 1 / 28
  missing: 28

## Outcome
Outcome B (PDF likely lacks missing CiPA parents; text layer and tables show same subset).
See JSON for per-drug diagnostics, closest-match suggestions, and table signatures.
