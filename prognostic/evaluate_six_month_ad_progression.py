"""Estimate MCI-to-AD progression at six-month intervals.

This script uses dated post-index diagnoses and adequate follow-up to estimate
cumulative conversion by 6, 12, 18, and 24 months. A penalised Cox model is
evaluated with repeated participant-level cross-validation. The resulting
probabilities are research estimates, not a clinical diagnostic tool.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import joblib
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler

from evaluate_early_ad_all_available import (
    DEFAULT_ADNI2_FEATURES,
    DEFAULT_CLINICAL,
    DEFAULT_COHORT,
    build_feature_matrix,
    load_analysis_data,
    required_prediction_columns,
)


DEFAULT_OUTPUT = ROOT / "six_month_ad_progression_results"
FEATURE_SET = "index_GNN13_MRI_age_MMSE"
HORIZON_MONTHS = np.array([6, 12, 18, 24], dtype=int)
HORIZON_DAYS = np.array([183, 365, 548, 730], dtype=float)
OUTER_FOLDS = 5
OUTER_REPEATS = 5
INNER_FOLDS = 3
BOOTSTRAP_REPETITIONS = 5000
SEED = 20260815
COX_GRID = [
    {"penalizer": 0.001, "l1_ratio": 0.0},
    {"penalizer": 0.01, "l1_ratio": 0.0},
    {"penalizer": 0.1, "l1_ratio": 0.0},
    {"penalizer": 1.0, "l1_ratio": 0.0},
    {"penalizer": 0.01, "l1_ratio": 0.5},
    {"penalizer": 0.1, "l1_ratio": 0.5},
    {"penalizer": 1.0, "l1_ratio": 0.5},
]


def parse_args() -> argparse.Namespace:
    """Parse reproducible command-line settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--clinical", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--adni2-features", type=Path, default=DEFAULT_ADNI2_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-repeats", type=int, default=OUTER_REPEATS)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPETITIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    """Create console and file logging."""

    logger = logging.getLogger("six_month_ad_progression")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def prepare_survival_outcomes(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return observed duration and event arrays after validating ascertainment."""

    event = table["label"].to_numpy(dtype=int)
    event_days = table["days_to_ad_after_second_mri"].to_numpy(dtype=float)
    followup_days = table["available_followup_days"].to_numpy(dtype=float)
    duration = np.where(event == 1, event_days, followup_days)
    if not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError("Every participant must have a positive event or censoring time.")
    if (event_days[event == 1] > HORIZON_DAYS[-1]).any():
        raise ValueError("A converter event lies beyond the prespecified 24-month horizon.")
    if (followup_days[event == 0] < HORIZON_DAYS[-1]).any():
        raise ValueError("A stable participant lacks complete 24-month follow-up.")
    return duration, event


def transform_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, RobustScaler]:
    """Fit median imputation and robust scaling on training data only."""

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(5, 95))
    train_imputed = imputer.fit_transform(x_train)
    return (
        scaler.fit_transform(train_imputed),
        scaler.transform(imputer.transform(x_test)),
        imputer,
        scaler,
    )


def cox_frame(x: np.ndarray, duration: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Create a lifelines-compatible table with formula-safe feature names."""

    frame = pd.DataFrame(x, columns=[f"x{index:03d}" for index in range(x.shape[1])])
    frame["duration"] = duration
    frame["event"] = event
    return frame


def fit_cox(
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    params: dict[str, float],
) -> CoxPHFitter:
    """Fit one penalised Cox proportional-hazards model."""

    model = CoxPHFitter(penalizer=params["penalizer"], l1_ratio=params["l1_ratio"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(cox_frame(x, duration, event), duration_col="duration", event_col="event")
    return model


def select_cox_parameters(
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], float]:
    """Tune Cox regularisation using only an outer training partition."""

    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    candidates: list[tuple[float, int, dict[str, float]]] = []
    for order, params in enumerate(COX_GRID):
        scores: list[float] = []
        for train_index, validation_index in inner.split(x, event):
            x_train, x_validation, _, _ = transform_train_test(x[train_index], x[validation_index])
            try:
                model = fit_cox(x_train, duration[train_index], event[train_index], params)
                risk = model.predict_partial_hazard(
                    cox_frame(
                        x_validation,
                        duration[validation_index],
                        event[validation_index],
                    ).drop(columns=["duration", "event"])
                ).to_numpy()
                score = concordance_index(
                    duration[validation_index], -risk, event[validation_index]
                )
                if np.isfinite(score):
                    scores.append(float(score))
            except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                scores = []
                break
        mean_score = float(np.mean(scores)) if len(scores) == INNER_FOLDS else -np.inf
        candidates.append((mean_score, -order, params))
    best_score, _, best_params = max(candidates, key=lambda item: (item[0], item[1]))
    if not np.isfinite(best_score):
        raise RuntimeError("All inner Cox fits failed.")
    return best_params, best_score


def predict_horizon_risks(model: CoxPHFitter, x: np.ndarray) -> np.ndarray:
    """Predict monotonic cumulative event probabilities at all horizons."""

    frame = pd.DataFrame(x, columns=[f"x{index:03d}" for index in range(x.shape[1])])
    survival = model.predict_survival_function(frame, times=HORIZON_DAYS).T.to_numpy()
    risk = np.clip(1.0 - survival, 0.0, 1.0)
    return np.maximum.accumulate(risk, axis=1)


def repeated_cross_validated_predictions(
    table: pd.DataFrame,
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    repeats: int,
    seed: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one leakage-controlled out-of-fold prediction per repeat."""

    splitter = RepeatedStratifiedKFold(
        n_splits=OUTER_FOLDS,
        n_repeats=repeats,
        random_state=seed,
    )
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for split_number, (train_index, test_index) in enumerate(splitter.split(x, event)):
        repeat = split_number // OUTER_FOLDS + 1
        fold = split_number % OUTER_FOLDS + 1
        fold_seed = seed + split_number + 1
        params, inner_cindex = select_cox_parameters(
            x[train_index], duration[train_index], event[train_index], fold_seed
        )
        x_train, x_test, _, _ = transform_train_test(x[train_index], x[test_index])
        model = fit_cox(x_train, duration[train_index], event[train_index], params)
        horizon_risk = predict_horizon_risks(model, x_test)
        frame = pd.DataFrame(x_test, columns=[f"x{i:03d}" for i in range(x.shape[1])])
        partial_hazard = model.predict_partial_hazard(frame).to_numpy()
        for local_index, participant_index in enumerate(test_index):
            row: dict[str, object] = {
                "participant_id": table.iloc[participant_index]["participant_id"],
                "repeat": repeat,
                "fold": fold,
                "duration_days": duration[participant_index],
                "event": event[participant_index],
                "partial_hazard": partial_hazard[local_index],
            }
            for horizon_index, month in enumerate(HORIZON_MONTHS):
                row[f"probability_by_{month}_months"] = horizon_risk[local_index, horizon_index]
            prediction_rows.append(row)
        selection_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "penalizer": params["penalizer"],
                "l1_ratio": params["l1_ratio"],
                "inner_c_index": inner_cindex,
            }
        )
        logger.info(
            "Completed repeat %d/%d fold %d/%d (penalizer=%g, l1_ratio=%g)",
            repeat,
            repeats,
            fold,
            OUTER_FOLDS,
            params["penalizer"],
            params["l1_ratio"],
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(selection_rows)


def aggregate_participant_predictions(oof: pd.DataFrame) -> pd.DataFrame:
    """Average repeated out-of-fold probabilities for each participant."""

    probability_columns = [f"probability_by_{month}_months" for month in HORIZON_MONTHS]
    aggregated = (
        oof.groupby("participant_id", as_index=False)
        .agg(
            duration_days=("duration_days", "first"),
            event=("event", "first"),
            partial_hazard=("partial_hazard", "mean"),
            prediction_repetitions=("repeat", "nunique"),
            **{column: (column, "mean") for column in probability_columns},
        )
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    return aggregated


def percentile_interval(values: list[float]) -> tuple[float, float]:
    """Return a two-sided 95% percentile interval."""

    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def bootstrap_performance(
    predictions: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate horizon metrics and participant-bootstrap confidence intervals."""

    rng = np.random.default_rng(seed)
    n = len(predictions)
    duration = predictions["duration_days"].to_numpy(float)
    event = predictions["event"].to_numpy(int)
    hazard = predictions["partial_hazard"].to_numpy(float)
    bootstrap_indices = [rng.integers(0, n, n) for _ in range(repetitions)]

    horizon_rows: list[dict[str, float | int]] = []
    previous_day = 0.0
    for month, day in zip(HORIZON_MONTHS, HORIZON_DAYS):
        probability = predictions[f"probability_by_{month}_months"].to_numpy(float)
        observed = ((event == 1) & (duration <= day)).astype(int)
        boot_observed: list[float] = []
        boot_predicted: list[float] = []
        boot_auc: list[float] = []
        boot_brier: list[float] = []
        for index in bootstrap_indices:
            y_sample = observed[index]
            p_sample = probability[index]
            boot_observed.append(float(np.mean(y_sample)))
            boot_predicted.append(float(np.mean(p_sample)))
            boot_brier.append(float(brier_score_loss(y_sample, p_sample)))
            if np.unique(y_sample).size == 2:
                boot_auc.append(float(roc_auc_score(y_sample, p_sample)))
        observed_low, observed_high = percentile_interval(boot_observed)
        predicted_low, predicted_high = percentile_interval(boot_predicted)
        auc_low, auc_high = percentile_interval(boot_auc)
        brier_low, brier_high = percentile_interval(boot_brier)
        horizon_rows.append(
            {
                "horizon_months": int(month),
                "horizon_days": int(day),
                "observed_conversions": int(observed.sum()),
                "at_risk_at_start": int(n - (((event == 1) & (duration <= previous_day)).sum())),
                "observed_cumulative_percent": 100.0 * float(np.mean(observed)),
                "observed_95ci_low_percent": 100.0 * observed_low,
                "observed_95ci_high_percent": 100.0 * observed_high,
                "mean_oof_predicted_percent": 100.0 * float(np.mean(probability)),
                "predicted_95ci_low_percent": 100.0 * predicted_low,
                "predicted_95ci_high_percent": 100.0 * predicted_high,
                "time_dependent_auc_percent": 100.0 * float(roc_auc_score(observed, probability)),
                "auc_95ci_low_percent": 100.0 * auc_low,
                "auc_95ci_high_percent": 100.0 * auc_high,
                "brier_score": float(brier_score_loss(observed, probability)),
                "brier_95ci_low": brier_low,
                "brier_95ci_high": brier_high,
            }
        )
        previous_day = day

    base_cindex = float(concordance_index(duration, -hazard, event))
    boot_cindex: list[float] = []
    for index in bootstrap_indices:
        try:
            score = concordance_index(duration[index], -hazard[index], event[index])
            if np.isfinite(score):
                boot_cindex.append(float(score))
        except ZeroDivisionError:
            continue
    cindex_low, cindex_high = percentile_interval(boot_cindex)
    overall = pd.DataFrame(
        [
            {
                "metric": "Harrell C-index",
                "value_percent": 100.0 * base_cindex,
                "ci_95_low_percent": 100.0 * cindex_low,
                "ci_95_high_percent": 100.0 * cindex_high,
                "bootstrap_repetitions": repetitions,
            }
        ]
    )
    return pd.DataFrame(horizon_rows), overall


def add_interval_statistics(horizon_table: pd.DataFrame) -> pd.DataFrame:
    """Add newly observed conversions in each non-overlapping six-month interval."""

    result = horizon_table.copy()
    cumulative_count = result["observed_conversions"].to_numpy(int)
    interval_count = np.diff(np.r_[0, cumulative_count])
    interval_start_population = np.r_[159, 159 - cumulative_count[:-1]]
    result["new_conversions_in_interval"] = interval_count
    result["interval_percent_of_full_cohort"] = 100.0 * interval_count / 159.0
    result["conditional_interval_conversion_percent"] = 100.0 * interval_count / interval_start_population
    return result


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare observed and predicted progression within OOF risk quartiles."""

    table = predictions.copy()
    table["risk_quartile"] = pd.qcut(
        table["probability_by_24_months"],
        4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
    )
    rows: list[dict[str, object]] = []
    for quartile, group in table.groupby("risk_quartile", observed=True):
        for month, day in zip(HORIZON_MONTHS, HORIZON_DAYS):
            observed = ((group["event"] == 1) & (group["duration_days"] <= day)).mean()
            rows.append(
                {
                    "risk_quartile": str(quartile),
                    "participants": len(group),
                    "horizon_months": int(month),
                    "observed_cumulative_percent": 100.0 * float(observed),
                    "mean_predicted_percent": 100.0 * float(group[f"probability_by_{month}_months"].mean()),
                }
            )
    return pd.DataFrame(rows)


def risk_group_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise observed cumulative conversion for low/high OOF risk groups."""

    table = predictions.copy()
    table["risk_group"] = pd.qcut(
        table["probability_by_24_months"], 2, labels=["Lower predicted risk", "Higher predicted risk"]
    )
    rows: list[dict[str, object]] = []
    for group_name, group in table.groupby("risk_group", observed=True):
        for month, day in zip(HORIZON_MONTHS, HORIZON_DAYS):
            observed = ((group["event"] == 1) & (group["duration_days"] <= day)).mean()
            rows.append(
                {
                    "risk_group": str(group_name),
                    "participants": len(group),
                    "horizon_months": int(month),
                    "observed_cumulative_percent": 100.0 * float(observed),
                    "mean_predicted_percent": 100.0 * float(group[f"probability_by_{month}_months"].mean()),
                }
            )
    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    """Apply a restrained publication-friendly visual style."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "#f7f4ed",
            "axes.facecolor": "#fffdf8",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_progression(horizon: pd.DataFrame, output_path: Path) -> None:
    """Plot observed and cross-validated cumulative progression."""

    configure_plot_style()
    months = np.r_[0, horizon["horizon_months"].to_numpy()]
    observed = np.r_[0.0, horizon["observed_cumulative_percent"].to_numpy()]
    predicted = np.r_[0.0, horizon["mean_oof_predicted_percent"].to_numpy()]
    observed_low = np.r_[0.0, horizon["observed_95ci_low_percent"].to_numpy()]
    observed_high = np.r_[0.0, horizon["observed_95ci_high_percent"].to_numpy()]
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    axis.fill_between(months, observed_low, observed_high, color="#147d79", alpha=0.16, label="Observed 95% CI")
    axis.plot(months, observed, marker="o", linewidth=2.5, color="#147d79", label="Observed conversion")
    axis.plot(months, predicted, marker="s", linewidth=2.2, color="#c6573c", label="Mean repeated OOF prediction")
    axis.set(title="Cumulative MCI-to-AD progression at six-month intervals", xlabel="Months after index MCI MRI", ylabel="Cumulative conversion (%)")
    axis.set_xticks(months)
    axis.set_ylim(0, max(observed_high.max(), predicted.max()) + 8)
    axis.grid(axis="y", color="#d8d2c4", linewidth=0.7)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_risk_groups(risk_groups: pd.DataFrame, output_path: Path) -> None:
    """Plot observed progression separately for lower and higher OOF risk groups."""

    configure_plot_style()
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    colors = {"Lower predicted risk": "#2978a0", "Higher predicted risk": "#b33f2e"}
    for group_name, group in risk_groups.groupby("risk_group", sort=False):
        months = np.r_[0, group["horizon_months"].to_numpy()]
        values = np.r_[0.0, group["observed_cumulative_percent"].to_numpy()]
        axis.step(months, values, where="post", linewidth=2.5, color=colors[group_name], label=group_name)
        axis.scatter(months[1:], values[1:], s=28, color=colors[group_name])
    axis.set(title="Observed progression by cross-validated MRI risk group", xlabel="Months after index MCI MRI", ylabel="Cumulative conversion (%)")
    axis.set_xticks([0, 6, 12, 18, 24])
    axis.set_ylim(0, 100)
    axis.grid(axis="y", color="#d8d2c4", linewidth=0.7)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_model_performance(horizon: pd.DataFrame, output_path: Path) -> None:
    """Plot time-dependent discrimination and prediction error."""

    configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    months = horizon["horizon_months"].to_numpy()
    auc = horizon["time_dependent_auc_percent"].to_numpy()
    axes[0].plot(months, auc, marker="o", linewidth=2.4, color="#147d79")
    axes[0].fill_between(months, horizon["auc_95ci_low_percent"], horizon["auc_95ci_high_percent"], color="#147d79", alpha=0.16)
    axes[0].axhline(50, color="#8c8578", linestyle="--", linewidth=1)
    axes[0].set(title="Time-dependent discrimination", xlabel="Prediction horizon (months)", ylabel="ROC-AUC (%)", xticks=months, ylim=(40, 100))
    axes[1].plot(months, horizon["brier_score"], marker="s", linewidth=2.4, color="#c6573c")
    axes[1].fill_between(months, horizon["brier_95ci_low"], horizon["brier_95ci_high"], color="#c6573c", alpha=0.16)
    axes[1].set(title="Probability prediction error", xlabel="Prediction horizon (months)", ylabel="Brier score (lower is better)", xticks=months, ylim=(0, max(0.3, horizon["brier_95ci_high"].max() + 0.02)))
    for axis in axes:
        axis.grid(axis="y", color="#d8d2c4", linewidth=0.7)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def fit_final_model(
    x: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    feature_names: list[str],
    raw_template: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> dict[str, object]:
    """Fit and save a full-cohort research model after internal evaluation."""

    params, inner_cindex = select_cox_parameters(x, duration, event, seed + 9000)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(5, 95))
    transformed = scaler.fit_transform(imputer.fit_transform(x))
    model = fit_cox(transformed, duration, event, params)
    artifact = {
        "model_type": "penalised Cox proportional hazards",
        "feature_set": FEATURE_SET,
        "feature_names": feature_names,
        "feature_columns": [f"x{index:03d}" for index in range(x.shape[1])],
        "required_raw_columns": raw_template.columns.tolist(),
        "imputer": imputer,
        "scaler": scaler,
        "cox_model": model,
        "horizon_months": HORIZON_MONTHS.tolist(),
        "horizon_days": HORIZON_DAYS.tolist(),
        "selected_parameters": params,
        "full_cohort_inner_c_index": inner_cindex,
        "training_participants": len(x),
        "training_events": int(event.sum()),
        "intended_use": "Research-only MCI-to-recorded-AD progression estimation through 24 months",
    }
    joblib.dump(artifact, output_dir / "final_six_month_progression_model.joblib")
    raw_template.head(1).assign(participant_id="example_participant").to_csv(
        output_dir / "six_month_prediction_input_template.csv", index=False
    )
    return artifact


def write_report(
    output_dir: Path,
    horizon: pd.DataFrame,
    overall: pd.DataFrame,
    risk_groups: pd.DataFrame,
    selections: pd.DataFrame,
    audit: dict[str, object],
    artifact: dict[str, object],
    repeats: int,
    bootstrap: int,
) -> None:
    """Write a thesis-safe plain-language methods and results report."""

    lines = [
        "# Six-Month MCI-to-AD Progression Analysis",
        "",
        "## Research question",
        "",
        "Among participants with MCI at the index MRI, what proportion had a recorded AD diagnosis by 6, 12, 18, and 24 months, and how well did the prespecified MRI-based model rank and estimate this risk?",
        "",
        "## Data and endpoint",
        "",
        f"- Participants: {audit['progression_participants']} ADNI MCI participants.",
        f"- Recorded conversions within 24 months: {audit['future_converters']}; stable MCI with at least 24 months of follow-up: {audit['stable_mci']}.",
        "- Time origin: the first eligible follow-up MCI MRI acquired at least 120 days after the participant's baseline MCI MRI; this is the index (second) MRI.",
        "- Index selection used scan timing and the requirement that no AD diagnosis had occurred through that date, not the future conversion date.",
        "- Event time: days from the index MRI to the first subsequent recorded AD diagnosis.",
        "- Stable participants were censored at their last available follow-up, all after the 24-month horizon.",
        "- Therefore conversion status is observed, rather than imputed, at every reported checkpoint.",
        "",
        "## Model inputs",
        "",
        "The model used eTIV-normalised index MRI volumes from 13 independently transferred regions, plus age and index MMSE. The regions were bilateral precuneus, middle temporal, parahippocampal, inferior temporal, hippocampus and amygdala, plus left fusiform. Region selection came from a separate ADNI2 diagnostic GNN cohort with zero participant overlap; the progression model itself is a penalised Cox model, not a GNN.",
        "",
        "## Validation",
        "",
        f"Preprocessing and Cox regularisation selection were repeated within {OUTER_FOLDS}-fold cross-validation for {repeats} repeats. Each participant therefore received {repeats} out-of-fold predictions, which were averaged. Confidence intervals used {bootstrap:,} participant bootstrap samples. No synthetic augmentation was used.",
        "",
        "## Observed and predicted progression",
        "",
        "| Horizon | New conversions in interval | Cumulative conversions | Observed cumulative (95% CI) | Mean OOF predicted | Time-dependent AUC (95% CI) | Brier score |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in horizon.itertuples(index=False):
        lines.append(
            f"| {row.horizon_months} months | {row.new_conversions_in_interval} | {row.observed_conversions} | "
            f"{row.observed_cumulative_percent:.2f}% ({row.observed_95ci_low_percent:.2f}-{row.observed_95ci_high_percent:.2f}%) | "
            f"{row.mean_oof_predicted_percent:.2f}% | {row.time_dependent_auc_percent:.2f}% "
            f"({row.auc_95ci_low_percent:.2f}-{row.auc_95ci_high_percent:.2f}%) | {row.brier_score:.3f} |"
        )
    cindex = overall.iloc[0]
    lines.extend(
        [
            "",
            f"Overall Harrell C-index was {cindex['value_percent']:.2f}% (95% CI {cindex['ci_95_low_percent']:.2f}-{cindex['ci_95_high_percent']:.2f}%). This is the probability that, for a comparable pair, the participant ranked at higher predicted risk converted earlier.",
            "",
            "## Risk-group illustration",
            "",
            "The table below is a descriptive split at the median repeated out-of-fold 24-month probability. It illustrates risk stratification and is not a clinical cut-off.",
            "",
            "| Group | Participants | 6 months | 12 months | 18 months | 24 months |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group_name, group in risk_groups.groupby("risk_group", sort=False):
        values = dict(zip(group["horizon_months"], group["observed_cumulative_percent"]))
        lines.append(
            f"| {group_name} | {int(group['participants'].iloc[0])} | {values[6]:.2f}% | {values[12]:.2f}% | {values[18]:.2f}% | {values[24]:.2f}% |"
        )
    selection_counts = Counter(
        f"penalizer={row.penalizer:g}, l1_ratio={row.l1_ratio:g}" for row in selections.itertuples(index=False)
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The observed cumulative percentage describes how many participants had received an AD diagnosis by each checkpoint. The model probability estimates the same cumulative endpoint for an individual from their index MRI, age and MMSE. AUC measures ranking, while the Brier score measures probability error; neither proves causation or clinical utility.",
            "",
            "This analysis supports a thesis-safe claim of internally validated, MRI-based short-term risk stratification among already-MCI participants. It does not predict disease in cognitively normal people, establish biological mechanisms, or constitute prospective/external clinical validation. PET was not included because the available PET cohort did not have compatible dated prognostic outcomes.",
            "",
            "## Reproducibility",
            "",
            f"- Random seed: {SEED}.",
            f"- Selected full-cohort parameters: {json.dumps(artifact['selected_parameters'])}.",
            f"- Full-cohort inner-CV C-index used only for final tuning: {100.0 * float(artifact['full_cohort_inner_c_index']):.2f}%.",
            f"- Outer-fold parameter selections: {dict(selection_counts)}.",
            "- All participant-level probabilities are saved in `participant_six_month_probabilities.csv`.",
        ]
    )
    (output_dir / "SIX_MONTH_PROGRESSION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the full six-month progression workflow."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir)
    logger.info("Loading and validating progression cohort")
    table, audit = load_analysis_data(args.cohort, args.clinical, args.adni2_features)
    duration, event = prepare_survival_outcomes(table)
    x, feature_names = build_feature_matrix(table, FEATURE_SET)
    logger.info("Cohort: %d participants, %d events, %d features", len(table), int(event.sum()), x.shape[1])

    oof, selections = repeated_cross_validated_predictions(
        table, x, duration, event, args.outer_repeats, args.seed, logger
    )
    predictions = aggregate_participant_predictions(oof)
    expected_predictions = len(table) * args.outer_repeats
    if len(oof) != expected_predictions or len(predictions) != len(table):
        raise RuntimeError("Cross-validation did not produce the expected participant predictions.")
    if not (predictions["prediction_repetitions"] == args.outer_repeats).all():
        raise RuntimeError("Every participant must have one OOF prediction per repeat.")

    horizon, overall = bootstrap_performance(predictions, args.bootstrap, args.seed + 10000)
    horizon = add_interval_statistics(horizon)
    calibration = calibration_table(predictions)
    risk_groups = risk_group_table(predictions)

    raw_columns = required_prediction_columns(table, FEATURE_SET)
    artifact = fit_final_model(
        x,
        duration,
        event,
        feature_names,
        table[raw_columns],
        args.output_dir,
        args.seed,
    )

    oof.to_csv(args.output_dir / "repeated_oof_six_month_predictions.csv", index=False)
    predictions.to_csv(args.output_dir / "participant_six_month_probabilities.csv", index=False)
    horizon.to_csv(args.output_dir / "six_month_progression_summary.csv", index=False)
    overall.to_csv(args.output_dir / "overall_survival_performance.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_by_risk_quartile.csv", index=False)
    risk_groups.to_csv(args.output_dir / "progression_by_risk_group.csv", index=False)
    selections.to_csv(args.output_dir / "cox_parameter_selections.csv", index=False)
    pd.DataFrame({"feature": feature_names}).to_csv(args.output_dir / "model_features.csv", index=False)
    (args.output_dir / "cohort_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    plot_progression(horizon, args.output_dir / "six_month_cumulative_progression.png")
    plot_risk_groups(risk_groups, args.output_dir / "six_month_progression_by_risk_group.png")
    plot_model_performance(horizon, args.output_dir / "six_month_model_performance.png")
    write_report(
        args.output_dir,
        horizon,
        overall,
        risk_groups,
        selections,
        audit,
        artifact,
        args.outer_repeats,
        args.bootstrap,
    )
    logger.info("Analysis complete. Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
