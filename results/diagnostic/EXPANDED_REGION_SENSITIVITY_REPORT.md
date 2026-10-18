# Expanded Candidate-Region Sensitivity Analysis

The same 159 participants and repeated participant-level outer-validation design were used in both analyses. The original analysis offered 20 AD-focused candidate nodes to the GNN; the expanded analysis offered 66 anatomically matched MUSE-AAL nodes. Both selected 15 regions independently inside each outer training fold.

## Selected-panel estimates

| Analysis | Metric | Estimate | 95% CI |
|---|---|---:|---:|
| Original 20-candidate GNN panel | roc_auc | 98.44% | 96.42% to 99.77% |
| Original 20-candidate GNN panel | balanced_accuracy | 88.64% | 79.55% to 95.45% |
| Original 20-candidate GNN panel | f1 | 87.18% | 74.29% to 95.24% |
| Original 20-candidate GNN panel | brier_score | 3.94% | 2.65% to 5.51% |
| Expanded 66-candidate GNN panel | roc_auc | 98.04% | 95.79% to 99.54% |
| Expanded 66-candidate GNN panel | balanced_accuracy | 85.27% | 75.80% to 93.99% |
| Expanded 66-candidate GNN panel | f1 | 78.05% | 63.16% to 90.00% |
| Expanded 66-candidate GNN panel | brier_score | 4.40% | 3.07% to 6.00% |

## Paired expanded-minus-targeted differences

| Metric | Difference | 95% CI | Paired bootstrap p |
|---|---:|---:|---:|
| roc_auc | -0.40% | -1.59% to +0.46% | 0.399 |
| balanced_accuracy | -3.37% | -8.29% to -0.36% | 0.033 |
| f1 | -9.13% | -17.92% to -2.13% | 0.033 |
| brier_score | +0.46% | +0.02% to +0.98% | 0.036 |

## Interpretation

The expanded candidate search is a sensitivity analysis, not a replacement primary model. A difference is not described as an improvement when its paired confidence interval includes zero. The comparison tests predictive behaviour only; GNN importance remains model-dependent and does not establish that a selected region causes Alzheimer disease.
