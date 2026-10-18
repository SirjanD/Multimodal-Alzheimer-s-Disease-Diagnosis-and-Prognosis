"""Publication-friendly figures for diagnosis, GNN interpretation, and prognosis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import PipelineConfig


COLORS = {
    "MRI_only": "#1f5673",
    "PET_only": "#d27c2c",
    "MRI_PET_raw": "#2f7d5b",
    "MRI_PET_GNN_selected": "#a23e48",
}


def _save(fig: plt.Figure, path: Path) -> None:
    """Save a consistently rendered publication PNG."""

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_diagnostic_roc(predictions: pd.DataFrame, path: Path) -> None:
    """Plot participant-averaged outer-test ROC curves."""

    fig, axis = plt.subplots(figsize=(8, 7))
    for model, subset in predictions.groupby("model", sort=False):
        false_positive, true_positive, _ = roc_curve(subset["target_ad"], subset["probability_ad"])
        axis.plot(false_positive, true_positive, linewidth=2.2, label=model.replace("_", " "), color=COLORS[model])
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Chance")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Repeated Outer-Test AD-versus-CN ROC Curves")
    axis.legend(frameon=False, loc="lower right")
    axis.grid(alpha=0.2)
    _save(fig, path)


def plot_diagnostic_calibration(data: pd.DataFrame, path: Path) -> None:
    """Plot observed prevalence against calibrated predicted risk bins."""

    fig, axis = plt.subplots(figsize=(8, 7))
    for model, subset in data.groupby("model", sort=False):
        axis.plot(
            subset["mean_predicted_risk"],
            subset["observed_prevalence"],
            marker="o",
            linewidth=2,
            label=model.replace("_", " "),
            color=COLORS[model],
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Ideal")
    axis.set(xlabel="Mean predicted AD probability", ylabel="Observed AD proportion", title="Diagnostic Calibration")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(alpha=0.2)
    _save(fig, path)


def plot_decision_curves(data: pd.DataFrame, path: Path) -> None:
    """Plot model net benefit across threshold probabilities."""

    fig, axis = plt.subplots(figsize=(9, 7))
    for model, subset in data.groupby("model", sort=False):
        axis.plot(subset["threshold"], subset["net_benefit"], linewidth=2, label=model.replace("_", " "), color=COLORS[model])
    reference = data[data["model"] == data["model"].iloc[0]]
    axis.plot(reference["threshold"], reference["treat_all_net_benefit"], linestyle="--", color="#555555", label="Treat all")
    axis.axhline(0, color="#999999", linestyle=":", label="Treat none")
    axis.set(xlabel="Threshold probability", ylabel="Net benefit", title="Exploratory Diagnostic Decision Curves")
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=0.2)
    _save(fig, path)


def plot_diagnostic_metrics(summary: pd.DataFrame, path: Path) -> None:
    """Plot key diagnostic metrics with bootstrap confidence intervals."""

    metrics = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    subset = summary[summary["metric"].isin(metrics)].copy()
    fig, axis = plt.subplots(figsize=(12, 7))
    width = 0.18
    x = np.arange(len(metrics))
    for model_index, (model, values) in enumerate(subset.groupby("model", sort=False)):
        values = values.set_index("metric").loc[metrics]
        position = x + (model_index - 1.5) * width
        axis.bar(position, values["mean"], width=width, color=COLORS[model], label=model.replace("_", " "))
        axis.errorbar(
            position,
            values["mean"],
            yerr=[values["mean"] - values["ci_lower"], values["ci_upper"] - values["mean"]],
            fmt="none",
            ecolor="#333333",
            capsize=2,
            linewidth=0.8,
        )
    axis.set_xticks(x, [metric.replace("_", " ").title() for metric in metrics])
    axis.set_ylim(0, 1.05)
    axis.set(ylabel="Estimate", title="Diagnostic Performance with 95% Bootstrap Intervals")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    _save(fig, path)


def plot_gnn_importance(importance: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    """Plot the highest consensus training-fold regional saliency scores."""

    top = importance.nsmallest(top_n, "importance_rank").sort_values("importance_score")
    fig, axis = plt.subplots(figsize=(10, 8))
    axis.barh(top["region"], top["importance_score"], color="#2f7d5b")
    axis.set(xlabel="Mean absolute gradient-times-input importance", ylabel="Matched MRI/PET region", title="GNN Region Importance from Outer-Training Folds")
    axis.grid(axis="x", alpha=0.2)
    _save(fig, path)


def plot_interpretability_comparison(comparison: pd.DataFrame, path: Path) -> None:
    """Compare coefficient magnitude and GNN importance by modality."""

    fig, axis = plt.subplots(figsize=(8, 7))
    for modality, subset in comparison.groupby("modality"):
        color = "#1f5673" if modality == "MRI" else "#d27c2c"
        axis.scatter(subset["logistic_coef_abs"], subset["gnn_importance"], s=45, alpha=0.8, label=modality, color=color)
    axis.set(xlabel="Mean absolute standardized logistic coefficient", ylabel="Mean GNN importance", title="Logistic and GNN Regional Importance Comparison")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    _save(fig, path)


def plot_prognostic_summary(summary: pd.DataFrame, path: Path) -> None:
    """Plot C-index and integrated Brier results when prognosis is feasible."""

    subset = summary[summary["metric"].isin(["c_index", "integrated_brier_score"])]
    if subset.empty:
        return
    pivot = subset.pivot(index="model", columns="metric", values="mean")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, metric, title in zip(axes, ["c_index", "integrated_brier_score"], ["C-index (higher is better)", "Integrated Brier score (lower is better)"]):
        values = pivot.get(metric, pd.Series(dtype=float)).dropna()
        axis.barh(values.index.str.replace("_", " "), values, color="#2f7d5b")
        axis.set(xlabel=title)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("MCI-to-AD Prognostic Performance")
    _save(fig, path)


def generate_all_plots(config: PipelineConfig) -> None:
    """Generate every figure supported by available output tables."""

    output = config.output_dir
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(output / "diagnostic_participant_predictions.csv")
    calibration = pd.read_csv(output / "diagnostic_calibration_plots_data.csv")
    decision = pd.read_csv(output / "diagnostic_decision_curve_data.csv")
    summary = pd.read_csv(output / "diagnostic_results_summary.csv")
    importance = pd.read_csv(output / "gnn_region_importance.csv")
    comparison = pd.read_csv(output / "interpretability_comparison.csv")
    plot_diagnostic_roc(predictions, figures / "diagnostic_roc_curves.png")
    plot_diagnostic_calibration(calibration, figures / "diagnostic_calibration_curves.png")
    plot_decision_curves(decision, figures / "diagnostic_decision_curves.png")
    plot_diagnostic_metrics(summary, figures / "diagnostic_metric_comparison.png")
    plot_gnn_importance(importance, figures / "gnn_region_importance.png")
    plot_interpretability_comparison(comparison, figures / "interpretability_comparison.png")
    prognosis = output / "prognostic_results_summary.csv"
    if prognosis.exists():
        prognostic_summary = pd.read_csv(prognosis)
        plot_prognostic_summary(prognostic_summary, figures / "prognostic_metric_comparison.png")


if __name__ == "__main__":
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    arguments = parser.parse_args()
    cfg = load_config(arguments.config)
    generate_all_plots(cfg)
    print(f"Saved figures to {cfg.output_dir / 'figures'}")
