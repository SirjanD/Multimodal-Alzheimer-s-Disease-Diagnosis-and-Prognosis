"""Run the reproducible ADNI2 multimodal GNN-enhanced pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adni2_pipeline.config import PipelineConfig, load_config
from adni2_pipeline.data_loading import (
    OutcomeUnavailableError,
    build_participants_master,
    save_survival_feasibility,
)
from adni2_pipeline.diagnostic_models import run_diagnostic_models
from adni2_pipeline.feature_extraction import prepare_feature_tables
from adni2_pipeline.gnn_model import run_gnn_cross_validation
from adni2_pipeline.interpretability import compare_interpretability
from adni2_pipeline.plots import generate_all_plots
from adni2_pipeline.prognostic_models import run_prognostic_models, save_blocked_prognostic_outputs
from adni2_pipeline.regions import REGION_NODES


def prepare(config: PipelineConfig) -> pd.DataFrame:
    """Build master, feasibility, and regional feature tables."""

    master = build_participants_master(config)
    save_survival_feasibility(config, master)
    return prepare_feature_tables(config)


def write_report(config: PipelineConfig) -> Path:
    """Write a compact, automatically populated thesis-safe execution report."""

    output = config.output_dir
    features = pd.read_csv(output / "multimodal_features_long.csv")
    metrics = pd.read_csv(output / "diagnostic_results_summary.csv")
    differences = pd.read_csv(output / "diagnostic_paired_model_differences.csv")
    importance = pd.read_csv(output / "gnn_region_importance.csv")
    interpretability = json.loads((output / "interpretability_summary.json").read_text(encoding="utf-8"))
    prognosis = json.loads((output / "prognostic_model_status.json").read_text(encoding="utf-8"))
    metric_order = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier_score"]
    rows = []
    for model in metrics["model"].drop_duplicates():
        subset = metrics[metrics["model"] == model].set_index("metric")
        cells = [
            f"{subset.loc[metric, 'mean']:.3f} ({subset.loc[metric, 'ci_lower']:.3f}-{subset.loc[metric, 'ci_upper']:.3f})"
            for metric in metric_order
        ]
        rows.append("| " + model + " | " + " | ".join(cells) + " |")
    top_regions = "\n".join(
        f"- {row.region}: importance {row.importance_score:.5f}; selected in {row.selection_frequency:.0%} of outer-training folds."
        for row in importance.head(10).itertuples(index=False)
    )
    gnn_auc = differences[
        (differences["comparator"] == "MRI_PET_GNN_selected")
        & (differences["reference"] == "MRI_PET_raw")
        & (differences["metric"] == "roc_auc")
    ].iloc[0]
    fusion_auc = differences[
        (differences["comparator"] == "MRI_PET_raw")
        & (differences["reference"] == "MRI_only")
        & (differences["metric"] == "roc_auc")
    ].iloc[0]
    calibration_lines = "\n".join(
        f"- {model}: intercept {subset.loc['calibration_intercept', 'mean']:.3f}; slope {subset.loc['calibration_slope', 'mean']:.3f}."
        for model in metrics["model"].drop_duplicates()
        for subset in [metrics[metrics["model"] == model].set_index("metric")]
    )
    report = f"""# GNN-Enhanced Multimodal ADNI2 Pipeline Report

## Verified Analysis Set

- Diagnostic participants: {len(features)} ({int(features['target_ad'].sum())} AD and {int((features['target_ad'] == 0).sum())} CN).
- Candidate graph nodes: {len(REGION_NODES)} anatomically matched bilateral MRI/PET regions.
- Node measurements: MRI regional volume divided by MUSE ICV and PET AAL cerebellar-reference ratio.
- Edges: mean absolute MRI/PET Pearson structural covariance, sparsified to the six strongest neighbours per node and recomputed inside every outer training fold.
- Validation: five-fold participant-level outer validation repeated five times; logistic tuning and probability calibration use training data only.

## Diagnostic Results

Values are estimates with 95% participant-bootstrap intervals. Lower Brier score is better.

| Model | ROC-AUC | Balanced accuracy | Sensitivity | Specificity | Precision | F1 | Brier score |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The direct GNN probabilities are saved only as exploratory uncalibrated outputs. The evaluated GNN contribution is training-fold region ranking followed by calibrated logistic regression using the selected raw MRI and PET measurements.

Raw MRI+PET minus MRI-only AUC was {fusion_auc.difference:+.3f} (95% CI {fusion_auc.ci_lower:+.3f} to {fusion_auc.ci_upper:+.3f}; p={fusion_auc.paired_bootstrap_p:.3f}), so an incremental PET benefit was not demonstrated.

The GNN-selected panel minus raw-fusion AUC difference was {gnn_auc.difference:+.3f} (95% CI {gnn_auc.ci_lower:+.3f} to {gnn_auc.ci_upper:+.3f}; paired bootstrap p={gnn_auc.paired_bootstrap_p:.3f}). An improvement is not claimed when this interval includes zero.

## Calibration Diagnostics

Ideal calibration corresponds to intercept 0 and slope 1. These are internally estimated point values and should not be interpreted as external clinical calibration.

{calibration_lines}

## GNN Region Ranking

{top_regions}

The Spearman correlation between modality-specific logistic and GNN importance ranks was {interpretability['spearman_rank_correlation']:.3f}. Importance values are model-dependent associations, not causal biological effects.

## Prognostic Status

Status: **{prognosis['status']}**.

{prognosis.get('reason', 'Penalised Cox models completed using dated visit-level outcomes.')}

No MCI-to-AD C-index, time-dependent AUC, Brier score, or calibration value is reported when genuine event and censoring dates are unavailable.

## Thesis-Safe Scope

This is an internally validated, retrospective diagnostic feature-selection experiment. It does not establish prospective clinical diagnosis, external generalisability, or future MCI-to-AD prediction. The GNN is used to test whether graph-informed region selection improves the transparent calibrated logistic baseline; superiority is claimed only if the saved confidence intervals support it.
"""
    path = output / "PIPELINE_EXECUTION_REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    """Dispatch one stage or the complete pipeline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--stage",
        choices=["prepare", "gnn", "diagnostic", "prognostic", "plots", "all"],
        default="all",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.stage in {"prepare", "all"}:
        features = prepare(config)
    else:
        features = pd.read_csv(config.output_dir / "multimodal_features_long.csv")
    if arguments.stage in {"gnn", "all"}:
        run_gnn_cross_validation(config, features)
    if arguments.stage in {"diagnostic", "all"}:
        run_diagnostic_models(config, features)
        compare_interpretability(config)
    if arguments.stage in {"prognostic", "all"}:
        try:
            run_prognostic_models(config)
        except OutcomeUnavailableError as error:
            save_blocked_prognostic_outputs(config, str(error))
    if arguments.stage in {"plots", "all"}:
        generate_all_plots(config)
    if arguments.stage == "all":
        report = write_report(config)
        print(f"Pipeline complete. Report: {report}")


if __name__ == "__main__":
    main()
