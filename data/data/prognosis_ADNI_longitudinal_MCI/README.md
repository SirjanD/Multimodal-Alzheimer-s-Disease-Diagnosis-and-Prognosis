# Preprocessed MCI-to-AD Prognosis Dataset

This folder contains the participant-level table used for the thesis's internally validated MCI-to-AD survival analysis.

## Cohort and source distinction

- 159 participants have MCI at the index MRI: 84 have a recorded AD conversion within 24 months and 75 remain non-converters with at least 24 months of follow-up.
- The index MRI is the first eligible follow-up MRI at least 120 days after the baseline MCI MRI, with no AD diagnosis by the index date.
- Time to event begins at the index MRI. Converters use time to the first subsequent AD diagnosis; non-converters are censored at their last available follow-up.
- This is a separate ADNI longitudinal MCI cohort from the ADNI2 MRI-plus-PET diagnostic cohort. The project audit records zero participant overlap between these cohorts. Do not describe this prognosis subset as ADNI2 unless the source data provenance independently confirms that designation.
- The prognostic model uses MRI-only FreeSurfer regional volumes, normalized by eTIV, from 13 GNN-transferred regions, plus age and MMSE at the index scan. PET is not part of this prognosis model.

## Files

- `adni_mci_prognosis_preprocessed.csv`: one row per participant; index MRI predictors, age and MMSE, event indicator, time-to-event or censoring, and the 24-month conversion label.
- `model_feature_list.csv`: exact 15 model input feature names (13 MRI regional features, age, and MMSE).
- `cohort_audit.json`: cohort counts and the recorded participant-overlap audit.

## Column interpretation

- `event_observed`: 1 means an AD diagnosis is recorded after the index MRI and within 24 months; 0 means no conversion is recorded and the participant has at least 24 months of follow-up.
- `time_to_event_or_censor_days`: days from the index MRI to the first AD diagnosis for an event, or to last observed follow-up for a censored participant.
- `conversion_by_24_months`: binary 24-month endpoint used in the survival cohort; it matches `event_observed` because of the cohort eligibility rules above.
- `index_*_icv_fraction`: index-MRI FreeSurfer region volume divided by follow-up eTIV. `index_age` and `index_mmse` are measured at the index visit.

The Cox model handles the event and censoring fields as survival outcomes. It does not use the outcome columns as predictors. Feature imputation, scaling, and penalty selection are fitted within training folds.

## Data handling

The table contains ADNI participant codes and clinical outcomes. ADNI data are access-controlled; retain the applicable terms of use and do not redistribute the table publicly unless the governing agreement permits it. The source images are not included. Longer-horizon projections beyond 24 months are extrapolations and are not validated by this cohort's reported evaluation.
