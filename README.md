# ADNI MRI/PET Graph Analysis for Alzheimer's Disease

Reproducible research code for MRI and AV45 amyloid-PET regional feature extraction, graph neural network (GNN) guided region selection, AD-versus-cognitively-normal (CN) diagnosis, and MRI-based MCI-to-AD prognosis.

The repository separates two analyses that use different participant cohorts:

- **Diagnosis:** ADNI2, paired MRI and AV45 PET, 159 participants (22 AD and 137 CN), with 66 matched anatomical regions.
- **Prognosis:** a separate longitudinal ADNI MCI cohort, 159 participants (84 conversions by 24 months and 75 non-converters with adequate follow-up). Its provenance does not establish that it is specifically an ADNI2 subset.

The prognosis model uses 13 eTIV-normalized MRI regions selected through a separate GNN analysis, plus age and MMSE at the index MRI. It is a penalised Cox model. The diagnostic graph uses regions as nodes, MRI and PET regional values as node features, and training-fold structural covariance for edges. The final diagnosis classifier is logistic regression. GNN embeddings and direct GNN probabilities are exploratory comparisons, not the primary clinical prediction scores.

## Repository layout

- `preprocessing/`: ADNI2 ZIP inventory, MRI/PET regional feature extraction, and longitudinal MCI cohort preparation.
- `diagnostic/`: graph construction, GCN/GAT models, region ranking, calibrated logistic models, and diagnostic validation.
- `prognostic/`: 24-month MCI conversion analysis, six-month checkpoint survival evaluation, and application of the fitted Cox model.
- `data/`: local ADNI2 diagnostic and longitudinal MCI preprocessed feature tables. They are excluded from Git.
- `results/`: selected aggregate metrics, thesis figures, and reports. The directory is excluded from Git by default.

## Data access and release

ADNI data are access-controlled and are not included. Obtain the relevant imaging, clinical, and diagnosis files through ADNI and follow its current data-use agreement. Do not commit ADNI participant tables, identifiers, scans, clinical records, predictions, or model artifacts to a public repository.

The MUSE mapping files and AAL atlas files are also not bundled. Obtain them from their authorized source and place them in `data/private/`. This local working copy includes preprocessed participant-level ADNI-derived tables under `data/processed/`; these are supplied for local reproducibility only and must not be uploaded publicly unless ADNI's terms explicitly permit redistribution. The repository's `.gitignore` excludes private data, processed data, results, NIfTI images, ZIP archives, and fitted model files.

The `results/` folder contains aggregate diagnostic and prognostic summaries, GNN region rankings, computational-efficiency summaries, reports, and figures. Participant-level prediction files are intentionally not included. Results are research outputs from the recorded internal validation workflow, not evidence of prospective clinical performance.

Before making this repository public, confirm that the code and any third-party atlas resources meet your university's and data providers' distribution terms. Keep `data/processed/` and `results/` out of any public commit unless data rights and disclosure review explicitly allow them. The accompanying GitHub upload ZIP is source-only and excludes those local folders. No license is included yet; add one only after confirming the rights holder and intended reuse terms.

## Setup

Use Python 3.10 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch Geometric installation can depend on the installed PyTorch and platform. If the requirements installation cannot resolve its wheel, follow the official PyTorch Geometric installation instructions for the matching environment.

## Prepare inputs

Place authorized inputs under `data/private/`. Do not add these files to Git.

For ADNI2 diagnostic preprocessing, the code expects the ADNI2 NIfTI ZIP, a participant catalogue, and the source acquisition-date table used to match PET tracer and Acq Date. It also needs the MUSE mapping tables and AAL-SPM12 atlas/label files. Use command-line options to point to the exact files available to you.

For prognosis preprocessing, the two-MRI cohort builder expects the authorized longitudinal MRI archive and visit-level diagnosis data. The clinical preparation script uses the cohort and diagnosis TSV files. The Cox script then expects the eligible MCI cohort, matched clinical feature table, and ADNI2 feature table used to audit participant overlap. These are distinct inputs from the ADNI2 diagnostic archive.

## Run the analyses

### 1. Audit the ADNI2 imaging archive

```powershell
python preprocessing/audit_adni2_dataset.py `
  --archive data/private/ADNI2_dataset.zip `
  --catalog data/private/ADNI2_catalog.csv `
  --source-acquisitions data/private/date_level_acquisitions.csv `
  --output-dir data/processed/adni2_inventory
```

### 2. Extract paired MRI/PET regional features

```powershell
python preprocessing/extract_adni2_features.py `
  --archive data/private/ADNI2_dataset.zip `
  --pairs data/processed/adni2_inventory/pet_mri_acq_date_pairs.csv `
  --inventory data/processed/adni2_inventory/nifti_inventory.csv `
  --muse-mapping data/private/MUSE_mapping_consecutive_indices.csv `
  --muse-complete data/private/MUSE_ROI_complete_list.csv `
  --aal data/private/ROI_MNI_V4.nii `
  --aal-labels data/private/aal_labels.csv `
  --output-dir data/processed/adni2_features
```

### 3. Run diagnostic GNN feature selection and classification

Copy `diagnostic/config.example.json` to `diagnostic/config.json`, adjust paths if needed, then run:

```powershell
python diagnostic/run_pipeline.py --config diagnostic/config.json --stage all
```

The configured cross-validation keeps graph construction, region selection, scaling, tuning, and calibration within training folds. The archive's static group labels do not supply dated conversion outcomes; the diagnostic pipeline records prognosis as unavailable when valid visit-level outcomes are absent.

### 4. Prepare the longitudinal MCI cohort and clinical table

Place the authorized longitudinal MRI archive and diagnosis files under `data/private/`.

```powershell
python preprocessing/build_mci_mri_cohort.py `
  --source-zip data/private/longitudinal_mri_source.zip `
  --diagnosis-dir data/private/diagnosis_tables `
  --output-dir data/processed/longitudinal_mri_change

python preprocessing/prepare_clinical_features.py `
  --cohort-csv data/processed/longitudinal_mri_change/eligible_two_mri_cohort.csv `
  --diagnosis-files data/private/Train_diagnosis_ADNI.tsv data/private/Val_diagnosis_ADNI.tsv data/private/Test_diagnosis_ADNI.tsv `
  --output-dir data/processed/mci_clinical
```

### 5. Run MCI-to-AD survival evaluation

```powershell
python prognostic/evaluate_six_month_ad_progression.py `
  --cohort data/processed/longitudinal_mri_change/eligible_two_mri_cohort.csv `
  --clinical data/processed/mci_clinical/matched_clinical_feature_table.csv `
  --adni2-features data/processed/adni2_features/multimodal_features_wide.csv `
  --output-dir results/prognosis
```

This evaluates the separate longitudinal MCI cohort with participant-level repeated cross-validation and saves out-of-fold predictions and six-, twelve-, eighteen-, and twenty-four-month metrics. It requires genuine dated event and censoring information; do not replace missing dates with labels inferred from archive folder names.

## Interpretation limits

These scripts support retrospective research analysis. Diagnostic discrimination is not prospective diagnosis. The prognosis model estimates risk among participants already diagnosed with MCI; it does not predict conversion from CN. Longer-horizon projections beyond 24 months are extrapolations and are not validated by the reported 24-month evaluation. Region importance is model-relative and does not show that a region causes disease.

## Local output handling

Keep outputs under `results/` or `data/processed/`; both locations are excluded by `.gitignore`. Review `git status` and inspect the final staged files before publishing. Never publish participant-level predictions or identifiers.
