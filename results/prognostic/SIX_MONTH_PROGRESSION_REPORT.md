# Six-Month MCI-to-AD Progression Analysis

## Research question

Among participants with MCI at the index MRI, what proportion had a recorded AD diagnosis by 6, 12, 18, and 24 months, and how well did the prespecified MRI-based model rank and estimate this risk?

## Data and endpoint

- Participants: 159 ADNI MCI participants.
- Recorded conversions within 24 months: 84; stable MCI with at least 24 months of follow-up: 75.
- Time origin: the first eligible follow-up MCI MRI acquired at least 120 days after the participant's baseline MCI MRI; this is the index (second) MRI.
- Index selection used scan timing and the requirement that no AD diagnosis had occurred through that date, not the future conversion date.
- Event time: days from the index MRI to the first subsequent recorded AD diagnosis.
- Stable participants were censored at their last available follow-up, all after the 24-month horizon.
- Therefore conversion status is observed, rather than imputed, at every reported checkpoint.

## Model inputs

The model used eTIV-normalised index MRI volumes from 13 independently transferred regions, plus age and index MMSE. The regions were bilateral precuneus, middle temporal, parahippocampal, inferior temporal, hippocampus and amygdala, plus left fusiform. Region selection came from a separate ADNI2 diagnostic GNN cohort with zero participant overlap; the progression model itself is a penalised Cox model, not a GNN.

## Validation

Preprocessing and Cox regularisation selection were repeated within 5-fold cross-validation for 5 repeats. Each participant therefore received 5 out-of-fold predictions, which were averaged. Confidence intervals used 5,000 participant bootstrap samples. No synthetic augmentation was used.

## Observed and predicted progression

| Horizon | New conversions in interval | Cumulative conversions | Observed cumulative (95% CI) | Mean OOF predicted | Time-dependent AUC (95% CI) | Brier score |
|---:|---:|---:|---:|---:|---:|---:|
| 6 months | 21 | 21 | 13.21% (8.18-18.87%) | 13.59% | 61.49% (48.27-74.08%) | 0.112 |
| 12 months | 27 | 48 | 30.19% (23.27-37.11%) | 31.02% | 71.38% (62.61-79.45%) | 0.187 |
| 18 months | 25 | 73 | 45.91% (37.74-53.46%) | 47.29% | 71.54% (63.27-79.60%) | 0.217 |
| 24 months | 11 | 84 | 52.83% (45.28-60.38%) | 54.24% | 75.21% (67.18-82.75%) | 0.211 |

Overall Harrell C-index was 67.83% (95% CI 61.83-73.73%). This is the probability that, for a comparable pair, the participant ranked at higher predicted risk converted earlier.

## Risk-group illustration

The table below is a descriptive split at the median repeated out-of-fold 24-month probability. It illustrates risk stratification and is not a clinical cut-off.

| Group | Participants | 6 months | 12 months | 18 months | 24 months |
|---|---:|---:|---:|---:|---:|
| Lower predicted risk | 80 | 10.00% | 17.50% | 31.25% | 35.00% |
| Higher predicted risk | 79 | 16.46% | 43.04% | 60.76% | 70.89% |

## Interpretation

The observed cumulative percentage describes how many participants had received an AD diagnosis by each checkpoint. The model probability estimates the same cumulative endpoint for an individual from their index MRI, age and MMSE. AUC measures ranking, while the Brier score measures probability error; neither proves causation or clinical utility.

This analysis supports a thesis-safe claim of internally validated, MRI-based short-term risk stratification among already-MCI participants. It does not predict disease in cognitively normal people, establish biological mechanisms, or constitute prospective/external clinical validation. PET was not included because the available PET cohort did not have compatible dated prognostic outcomes.

## Reproducibility

- Random seed: 20260815.
- Selected full-cohort parameters: {"penalizer": 1.0, "l1_ratio": 0.0}.
- Full-cohort inner-CV C-index used only for final tuning: 68.39%.
- Outer-fold parameter selections: {'penalizer=1, l1_ratio=0.5': 7, 'penalizer=0.1, l1_ratio=0.5': 10, 'penalizer=1, l1_ratio=0': 8}.
- All participant-level probabilities are saved in `participant_six_month_probabilities.csv`.
