# Source Hunt: Multichannel Candidates (CiPA-28)

Coverage counts are computed against the CiPA-28 parent set (alias OFF by default).
No public dataset found that provides IKs/IK1/Kv4.3 potency for all 28 drugs; candidates below are the best available.

## fda_cipa_manual_training_2018

Title: Manual patch-clamp training panel (CiPA Model-Validation-2018)
Citation: Crumb WJ Jr, Vicente J, Johannesen L, Strauss DG (2016) J Pharmacol Toxicol Methods 81:251-262. DOI:10.1016/j.vascn.2016.03.009.
License/terms: FDA/CiPA GitHub repository GPL-3.0 (LICENSE in repo)
URLs: https://github.com/FDA/CiPA/tree/Model-Validation-2018/Hill_Fitting/data, https://raw.githubusercontent.com/FDA/CiPA/Model-Validation-2018/Hill_Fitting/data/manual_trainingdrug_block.csv
Data acquisition: Direct raw CSV download from GitHub (reproducible).
Verdict: Good secondary gap-fill (limited overlap).
Recommended role: Secondary gap-fill for IKs/IK1/Kv4.3 (limited drug overlap).
Notes: Percent block vs concentration across 7 channels; 31 drugs.

Coverage by channel (alias OFF):
- hERG: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Cav1.2: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Nav1.5_peak: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Nav1.5_late: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- IKs: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- IK1: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Kv4.3: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)

Coverage by channel (alias ON):
- hERG: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Cav1.2: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Nav1.5_peak: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Nav1.5_late: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- IKs: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- IK1: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Kv4.3: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)

## fda_cipa_manual_validation_2018

Title: Manual patch-clamp validation panel (CiPA Model-Validation-2018)
Citation: FDA/CiPA Model-Validation-2018 data (see repo). Underlying patch-clamp context cites Crumb et al. 2016.
License/terms: FDA/CiPA GitHub repository GPL-3.0 (LICENSE in repo)
URLs: https://github.com/FDA/CiPA/tree/Model-Validation-2018/Hill_Fitting/data, https://raw.githubusercontent.com/FDA/CiPA/Model-Validation-2018/Hill_Fitting/data/manual_validationdrug_block.csv
Data acquisition: Direct raw CSV download from GitHub (reproducible).
Verdict: Reject for IKs/IK1/Kv4.3 coverage.
Recommended role: Not suitable as primary; only hERG/Na/Ca.
Notes: Percent block vs concentration for 4 channels only (no IKs/IK1/Kv4.3).

Coverage by channel (alias OFF):
- hERG: 16 / 28 (missing: bepridil, chlorpromazine, cisapride, diltiazem, dlsotalol, dofetilide, mexiletine, ondansetron, quinidine, ranolazine, terfenadine, verapamil)
- Cav1.2: 16 / 28 (missing: bepridil, chlorpromazine, cisapride, diltiazem, dlsotalol, dofetilide, mexiletine, ondansetron, quinidine, ranolazine, terfenadine, verapamil)
- Nav1.5_peak: 16 / 28 (missing: bepridil, chlorpromazine, cisapride, diltiazem, dlsotalol, dofetilide, mexiletine, ondansetron, quinidine, ranolazine, terfenadine, verapamil)
- Nav1.5_late: 16 / 28 (missing: bepridil, chlorpromazine, cisapride, diltiazem, dlsotalol, dofetilide, mexiletine, ondansetron, quinidine, ranolazine, terfenadine, verapamil)
- IKs: 0 / 28 (missing: astemizole, azimilide, bepridil, chlorpromazine, cisapride, clarithromycin, clozapine, diltiazem, disopyramide, dlsotalol, dofetilide, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, ondansetron, pimozide, quinidine, ranolazine, risperidone, tamoxifen, terfenadine, vandetanib, verapamil)
- IK1: 0 / 28 (missing: astemizole, azimilide, bepridil, chlorpromazine, cisapride, clarithromycin, clozapine, diltiazem, disopyramide, dlsotalol, dofetilide, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, ondansetron, pimozide, quinidine, ranolazine, risperidone, tamoxifen, terfenadine, vandetanib, verapamil)
- Kv4.3: 0 / 28 (missing: astemizole, azimilide, bepridil, chlorpromazine, cisapride, clarithromycin, clozapine, diltiazem, disopyramide, dlsotalol, dofetilide, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, ondansetron, pimozide, quinidine, ranolazine, risperidone, tamoxifen, terfenadine, vandetanib, verapamil)

Coverage by channel (alias ON):
- hERG: 16 / 28 (missing: bepridil, chlorpromazine, cisapride, diltiazem, dofetilide, mexiletine, ondansetron, quinidine, ranolazine, sotalol, terfenadine, verapamil)
- Cav1.2: 16 / 28 (missing: bepridil, chlorpromazine, cisapride, diltiazem, dofetilide, mexiletine, ondansetron, quinidine, ranolazine, sotalol, terfenadine, verapamil)
- Nav1.5_peak: 16 / 28 (missing: bepridil, chlorpromazine, cisapride, diltiazem, dofetilide, mexiletine, ondansetron, quinidine, ranolazine, sotalol, terfenadine, verapamil)
- Nav1.5_late: 16 / 28 (missing: bepridil, chlorpromazine, cisapride, diltiazem, dofetilide, mexiletine, ondansetron, quinidine, ranolazine, sotalol, terfenadine, verapamil)
- IKs: 0 / 28 (missing: astemizole, azimilide, bepridil, chlorpromazine, cisapride, clarithromycin, clozapine, diltiazem, disopyramide, dofetilide, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, ondansetron, pimozide, quinidine, ranolazine, risperidone, sotalol, tamoxifen, terfenadine, vandetanib, verapamil)
- IK1: 0 / 28 (missing: astemizole, azimilide, bepridil, chlorpromazine, cisapride, clarithromycin, clozapine, diltiazem, disopyramide, dofetilide, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, ondansetron, pimozide, quinidine, ranolazine, risperidone, sotalol, tamoxifen, terfenadine, vandetanib, verapamil)
- Kv4.3: 0 / 28 (missing: astemizole, azimilide, bepridil, chlorpromazine, cisapride, clarithromycin, clozapine, diltiazem, disopyramide, dofetilide, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, ondansetron, pimozide, quinidine, ranolazine, risperidone, sotalol, tamoxifen, terfenadine, vandetanib, verapamil)

## fda_cipa_hts_training_2018

Title: HTS patch-clamp training panel (CiPA Model-Validation-2018)
Citation: FDA/CiPA Model-Validation-2018 data (see repo). Underlying context cites Crumb et al. 2016.
License/terms: FDA/CiPA GitHub repository GPL-3.0 (LICENSE in repo)
URLs: https://github.com/FDA/CiPA/tree/Model-Validation-2018/Hill_Fitting/data, https://raw.githubusercontent.com/FDA/CiPA/Model-Validation-2018/Hill_Fitting/data/HTS_trainingdrug_block.csv
Data acquisition: Direct raw CSV download from GitHub (reproducible).
Verdict: Reject as primary; possible secondary with HTS caveats.
Recommended role: Secondary gap-fill; limited overlap and HTS assay heterogeneity.
Notes: High-throughput screening version of training panel; 7 channels, 12 drugs.

Coverage by channel (alias OFF):
- hERG: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Cav1.2: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Nav1.5_peak: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Nav1.5_late: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- IKs: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- IK1: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Kv4.3: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)

Coverage by channel (alias ON):
- hERG: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Cav1.2: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Nav1.5_peak: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Nav1.5_late: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- IKs: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- IK1: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Kv4.3: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)

## fda_cipa_li2017_ic50

Title: Li2017 IC50 summary (CiPA Model-Validation-2018)
Citation: FDA/CiPA Model-Validation-2018 data (Li2017_IC50.csv). Paper citation not explicit in repo; use repo link as stable archive.
License/terms: FDA/CiPA GitHub repository GPL-3.0 (LICENSE in repo)
URLs: https://github.com/FDA/CiPA/tree/Model-Validation-2018/Hill_Fitting/data, https://raw.githubusercontent.com/FDA/CiPA/Model-Validation-2018/Hill_Fitting/data/Li2017_IC50.csv
Data acquisition: Direct raw CSV download from GitHub (reproducible).
Verdict: Reject as primary (12 drugs only).
Recommended role: Possible reference for 7-channel IC50s.
Notes: Aggregated IC50/Hill values for 7 channels; units not specified in file.

Coverage by channel (alias OFF):
- hERG: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Cav1.2: 10 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, ranolazine, risperidone, tamoxifen, vandetanib)
- Nav1.5_peak: 8 / 28 (missing: astemizole, azimilide, cisapride, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib, verapamil)
- Nav1.5_late: 10 / 28 (missing: astemizole, azimilide, cisapride, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- IKs: 6 / 28 (missing: astemizole, azimilide, chlorpromazine, clarithromycin, clozapine, diltiazem, disopyramide, dlsotalol, dofetilide, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib, verapamil)
- IK1: 5 / 28 (missing: astemizole, azimilide, bepridil, clarithromycin, clozapine, diltiazem, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, ondansetron, pimozide, ranolazine, risperidone, tamoxifen, terfenadine, vandetanib)
- Kv4.3: 9 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, dlsotalol, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, pimozide, ranolazine, risperidone, tamoxifen, vandetanib)

Coverage by channel (alias ON):
- hERG: 12 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib)
- Cav1.2: 11 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, ranolazine, risperidone, tamoxifen, vandetanib)
- Nav1.5_peak: 9 / 28 (missing: astemizole, azimilide, cisapride, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib, verapamil)
- Nav1.5_late: 10 / 28 (missing: astemizole, azimilide, cisapride, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, nifedipine, nitrendipine, pimozide, risperidone, sotalol, tamoxifen, vandetanib)
- IKs: 7 / 28 (missing: astemizole, azimilide, chlorpromazine, clarithromycin, clozapine, diltiazem, disopyramide, dofetilide, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, pimozide, risperidone, tamoxifen, vandetanib, verapamil)
- IK1: 6 / 28 (missing: astemizole, azimilide, bepridil, clarithromycin, clozapine, diltiazem, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, ondansetron, pimozide, ranolazine, risperidone, tamoxifen, terfenadine, vandetanib)
- Kv4.3: 10 / 28 (missing: astemizole, azimilide, clarithromycin, clozapine, disopyramide, domperidone, droperidol, ibutilide, loratadine, metoprolol, mexiletine, nifedipine, nitrendipine, pimozide, ranolazine, risperidone, tamoxifen, vandetanib)
