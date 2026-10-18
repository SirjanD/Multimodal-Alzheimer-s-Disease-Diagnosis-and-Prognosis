"""Compare transparent logistic coefficients with training-derived GNN saliency."""

from __future__ import annotations

import json

import pandas as pd
from scipy.stats import spearmanr

from .config import PipelineConfig
from .regions import REGION_NODES


def compare_interpretability(config: PipelineConfig) -> pd.DataFrame:
    """Save modality-aware logistic/GNN importance ranks and their correlation."""

    output = config.output_dir
    coefficients = pd.read_csv(output / "diagnostic_fold_coefficients.csv")
    coefficients = coefficients[coefficients["model"] == "MRI_PET_raw"].copy()
    coefficient_summary = (
        coefficients.assign(logistic_coef_abs=coefficients["standardized_coefficient"].abs())
        .groupby("feature", as_index=False)["logistic_coef_abs"]
        .mean()
        .set_index("feature")["logistic_coef_abs"]
    )
    gnn = pd.read_csv(output / "gnn_region_importance.csv").set_index("region")
    rows = []
    for node in REGION_NODES:
        for modality, feature, importance_column in (
            ("MRI", node.mri_feature, "mri_importance"),
            ("PET", node.pet_feature, "pet_importance"),
        ):
            rows.append(
                {
                    "region": node.name,
                    "modality": modality,
                    "feature": feature,
                    "logistic_coef_abs": float(coefficient_summary[feature]),
                    "gnn_importance": float(gnn.loc[node.name, importance_column]),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison["rank_logistic"] = comparison["logistic_coef_abs"].rank(method="min", ascending=False).astype(int)
    comparison["rank_gnn"] = comparison["gnn_importance"].rank(method="min", ascending=False).astype(int)
    comparison = comparison.sort_values("rank_logistic").reset_index(drop=True)
    correlation, p_value = spearmanr(comparison["rank_logistic"], comparison["rank_gnn"])
    logistic_top = set(comparison.nsmallest(10, "rank_logistic")["region"])
    gnn_top = set(comparison.nsmallest(10, "rank_gnn")["region"])
    comparison.to_csv(output / "interpretability_comparison.csv", index=False)
    (output / "interpretability_summary.json").write_text(
        json.dumps(
            {
                "spearman_rank_correlation": float(correlation),
                "spearman_p_value": float(p_value),
                "top_10_region_overlap": sorted(logistic_top.intersection(gnn_top)),
                "overlap_count": len(logistic_top.intersection(gnn_top)),
                "caution": "Both rankings are model-dependent associations, not causal regional effects.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return comparison


if __name__ == "__main__":
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    arguments = parser.parse_args()
    result = compare_interpretability(load_config(arguments.config))
    print(f"Compared {len(result)} modality-specific regional features")
