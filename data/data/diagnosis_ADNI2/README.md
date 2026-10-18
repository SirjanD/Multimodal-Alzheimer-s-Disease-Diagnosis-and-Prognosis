# ADNI2 Preprocessed Dataset Used for the Thesis

This folder contains the processed, participant-level ADNI2 MRI and AV45 amyloid-PET regional features prepared for the thesis diagnostic analysis.

## Cohort and scope

- 159 participants: 22 with an existing Alzheimer’s disease (AD) diagnosis and 137 cognitively normal (CN) participants.
- MRI and AV45 amyloid-PET scans are paired within 90 acquisition-date days.
- 66 anatomical regions are represented, with matched MRI and PET measurements per region (132 regional imaging features in total).
- MRI regional volumes are normalized by intracranial volume. PET regional uptake is expressed relative to the cerebellar reference.
- The GNN selected regions file records the ranked 15-region panel from the expanded candidate analysis.

This package is for the AD-versus-CN diagnostic analysis. The source project reports that this ADNI2 archive does not contain dated visit-level conversion and censoring records sufficient to form a valid MCI-to-AD survival dataset. No prognosis data are included here.

## Files

- `adni2_diagnostic_features_wide.csv`: one row per participant; diagnosis and participant metadata followed by the regional MRI and PET features. This is the primary model-ready table. It also retains one summary PET-panel mean column from feature extraction; that summary is not a 67th brain region or graph node.
- `adni2_diagnostic_features_long.csv`: one row per participant, modality, and region; includes raw and normalized feature values and QC coverage fields.
- `feature_dictionary.csv`: feature names, modality, source, normalization, and whether the feature belongs to the primary model input.
- `gnn_selected_regions.csv`: GNN region-importance rank, relative importance, and selection frequency. Importance is model-relative and is not a biological effect size.

## Notes for use

The wide and long tables contain ADNI participant codes and study labels. ADNI data are access-controlled; retain the original ADNI terms of use and do not redistribute these files publicly unless the applicable agreement explicitly permits it. This curated copy omits imaging files and source archive paths.

For reproducibility, use `target_ad` as the binary outcome (AD = 1, CN = 0) and avoid using diagnosis-derived fields as predictors. Fit transformations, feature selection, graph edges, and model tuning within each training fold only. The table itself contains all 66 candidate regions; use the GNN-selected panel only as defined by the training-fold analysis when reproducing validation results.
