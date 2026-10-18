# GNN-Enhanced Multimodal ADNI2 Pipeline Report

## Verified Analysis Set

- Diagnostic participants: 159 (22 AD and 137 CN).
- Candidate graph nodes: 66 anatomically matched bilateral MRI/PET regions.
- Node measurements: MRI regional volume divided by MUSE ICV and PET AAL cerebellar-reference ratio.
- Edges: mean absolute MRI/PET Pearson structural covariance, sparsified to the six strongest neighbours per node and recomputed inside every outer training fold.
- Validation: five-fold participant-level outer validation repeated five times; logistic tuning and probability calibration use training data only.

## Diagnostic Results

Values are estimates with 95% participant-bootstrap intervals. Lower Brier score is better.

| Model | ROC-AUC | Balanced accuracy | Sensitivity | Specificity | Precision | F1 | Brier score |
|---|---:|---:|---:|---:|---:|---:|---:|
| MRI_only | 0.966 (0.929-0.993) | 0.811 (0.713-0.905) | 0.636 (0.455-0.818) | 0.985 (0.964-1.000) | 0.875 (0.722-1.000) | 0.737 (0.562-0.872) | 0.050 (0.035-0.066) |
| PET_only | 0.937 (0.889-0.977) | 0.701 (0.606-0.804) | 0.409 (0.227-0.636) | 0.993 (0.978-1.000) | 0.900 (0.700-1.000) | 0.562 (0.345-0.743) | 0.066 (0.049-0.083) |
| MRI_PET_raw | 0.978 (0.951-0.996) | 0.815 (0.720-0.909) | 0.636 (0.455-0.818) | 0.993 (0.978-1.000) | 0.933 (0.800-1.000) | 0.757 (0.581-0.900) | 0.046 (0.032-0.062) |
| MRI_PET_GNN_selected | 0.980 (0.958-0.995) | 0.853 (0.758-0.940) | 0.727 (0.545-0.909) | 0.978 (0.949-1.000) | 0.842 (0.684-1.000) | 0.780 (0.632-0.900) | 0.044 (0.031-0.060) |

The direct GNN probabilities are saved only as exploratory uncalibrated outputs. The evaluated GNN contribution is training-fold region ranking followed by calibrated logistic regression using the selected raw MRI and PET measurements.

Raw MRI+PET minus MRI-only AUC was +0.012 (95% CI -0.010 to +0.045; p=0.421), so an incremental PET benefit was not demonstrated.

The GNN-selected panel minus raw-fusion AUC difference was +0.003 (95% CI -0.009 to +0.015; paired bootstrap p=0.643). An improvement is not claimed when this interval includes zero.

## Calibration Diagnostics

Ideal calibration corresponds to intercept 0 and slope 1. These are internally estimated point values and should not be interpreted as external clinical calibration.

- MRI_only: intercept 1.101; slope 2.036.
- PET_only: intercept 0.907; slope 1.747.
- MRI_PET_raw: intercept 1.221; slope 2.155.
- MRI_PET_GNN_selected: intercept 1.085; slope 2.065.

## GNN Region Ranking

- Left hippocampus: importance 0.06245; selected in 96% of outer-training folds.
- Left middle temporal gyrus: importance 0.06214; selected in 100% of outer-training folds.
- Right hippocampus: importance 0.06043; selected in 80% of outer-training folds.
- Left precuneus: importance 0.05906; selected in 88% of outer-training folds.
- Right middle temporal gyrus: importance 0.05891; selected in 100% of outer-training folds.
- Right superior frontal gyrus: importance 0.05868; selected in 80% of outer-training folds.
- Right cuneus: importance 0.05763; selected in 76% of outer-training folds.
- Right parahippocampal gyrus: importance 0.05665; selected in 88% of outer-training folds.
- Left cuneus: importance 0.05644; selected in 68% of outer-training folds.
- Left amygdala: importance 0.05635; selected in 52% of outer-training folds.

The Spearman correlation between modality-specific logistic and GNN importance ranks was 0.346. Importance values are model-dependent associations, not causal biological effects.

## Prognostic Status

Status: **blocked**.

The available diagnoses are static archive folder labels rather than dated visit-level assessments. Conversion and censoring times cannot be derived validly.

No MCI-to-AD C-index, time-dependent AUC, Brier score, or calibration value is reported when genuine event and censoring dates are unavailable.

## Thesis-Safe Scope

This is an internally validated, retrospective diagnostic feature-selection experiment. It does not establish prospective clinical diagnosis, external generalisability, or future MCI-to-AD prediction. The GNN is used to test whether graph-informed region selection improves the transparent calibrated logistic baseline; superiority is claimed only if the saved confidence intervals support it.
