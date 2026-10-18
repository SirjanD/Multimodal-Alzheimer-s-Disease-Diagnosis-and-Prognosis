"""Nested calibrated logistic models for AD-versus-CN diagnosis."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .regions import feature_columns, region_feature_map


MODEL_ORDER = ["MRI_only", "PET_only", "MRI_PET_raw", "MRI_PET_GNN_selected"]


def _estimator(c_value: float) -> Pipeline:
    """Create the complete leakage-safe logistic preprocessing pipeline."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=0,
                ),
            ),
        ]
    )


def fit_calibrated_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: list[str],
    c_values: list[float],
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Tune regularisation and calibrate sigmoid probabilities within training data."""

    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        _estimator(1.0),
        {"logistic__C": c_values},
        scoring="roc_auc",
        cv=inner,
        refit=True,
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    best_c = float(search.best_params_["logistic__C"])
    calibration_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + 7919)
    calibrated = CalibratedClassifierCV(
        estimator=_estimator(best_c),
        method="sigmoid",
        cv=calibration_cv,
    )
    calibrated.fit(x_train, y_train)
    probabilities = calibrated.predict_proba(x_test)[:, 1]
    coefficients = np.vstack(
        [
            classifier.estimator.named_steps["logistic"].coef_[0]
            for classifier in calibrated.calibrated_classifiers_
        ]
    ).mean(axis=0)
    if len(coefficients) != len(feature_names):
        raise RuntimeError("Coefficient and feature dimensions do not match.")
    return probabilities, best_c, coefficients


def calibration_parameters(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Estimate logistic recalibration intercept and slope."""

    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    predictor = logit(clipped)

    def objective(parameters: np.ndarray) -> float:
        fitted = expit(parameters[0] + parameters[1] * predictor)
        return -float(
            np.sum(y_true * np.log(fitted + 1e-12) + (1 - y_true) * np.log(1 - fitted + 1e-12))
        )

    result = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    return (float(result.x[0]), float(result.x[1])) if result.success else (np.nan, np.nan)


def metric_values(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute discrimination, threshold, and calibration metrics."""

    predicted = (probability >= threshold).astype(int)
    intercept, slope = calibration_parameters(y_true, probability)
    sensitivity = recall_score(y_true, predicted, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, predicted, pos_label=0, zero_division=0)
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "brier_score": brier_score_loss(y_true, probability),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Return participant-level stratified percentile confidence intervals."""

    estimates = metric_values(y_true, probability, threshold)
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in estimates}
    for _ in range(repetitions):
        indices = np.concatenate(
            [rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)]
        )
        values = metric_values(y_true[indices], probability[indices], threshold)
        for metric, value in values.items():
            samples[metric].append(value)
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": estimate,
                "ci_lower": float(np.nanpercentile(samples[metric], 2.5)),
                "ci_upper": float(np.nanpercentile(samples[metric], 97.5)),
            }
            for metric, estimate in estimates.items()
        ]
    )


def decision_curve(y_true: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Compute model, treat-all, and treat-none net benefit."""

    n = len(y_true)
    prevalence = float(y_true.mean())
    rows = []
    for threshold in thresholds:
        predicted = probability >= threshold
        true_positive = int(np.sum(predicted & (y_true == 1)))
        false_positive = int(np.sum(predicted & (y_true == 0)))
        odds = threshold / (1 - threshold)
        rows.append(
            {
                "threshold": threshold,
                "net_benefit": true_positive / n - false_positive / n * odds,
                "treat_all_net_benefit": prevalence - (1 - prevalence) * odds,
                "treat_none_net_benefit": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _comparison_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Return metrics used in paired model-difference inference."""

    predicted = (probability >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "brier_score": brier_score_loss(y_true, probability),
    }


def paired_model_differences(
    averaged: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    threshold: float,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Compute paired participant-bootstrap differences between diagnostic models."""

    wide = averaged.pivot(index=["subject_id", "target_ad"], columns="model", values="probability_ad").reset_index()
    y_true = wide["target_ad"].to_numpy(dtype=int)
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    rng = np.random.default_rng(seed)
    rows = []
    for comparator, reference in comparisons:
        comparator_probability = wide[comparator].to_numpy(dtype=float)
        reference_probability = wide[reference].to_numpy(dtype=float)
        comparator_values = _comparison_metrics(y_true, comparator_probability, threshold)
        reference_values = _comparison_metrics(y_true, reference_probability, threshold)
        samples = {metric: [] for metric in comparator_values}
        for _ in range(repetitions):
            indices = np.concatenate(
                [rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)]
            )
            left = _comparison_metrics(y_true[indices], comparator_probability[indices], threshold)
            right = _comparison_metrics(y_true[indices], reference_probability[indices], threshold)
            for metric in samples:
                samples[metric].append(left[metric] - right[metric])
        for metric in samples:
            values = np.asarray(samples[metric])
            rows.append(
                {
                    "comparator": comparator,
                    "reference": reference,
                    "metric": metric,
                    "difference": comparator_values[metric] - reference_values[metric],
                    "ci_lower": float(np.percentile(values, 2.5)),
                    "ci_upper": float(np.percentile(values, 97.5)),
                    "paired_bootstrap_p": float(
                        min(1.0, 2 * min((np.sum(values <= 0) + 1) / (repetitions + 1), (np.sum(values >= 0) + 1) / (repetitions + 1)))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _fold_feature_sets(
    table: pd.DataFrame,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    selected_regions: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """Build raw and fold-specific GNN-selected regional feature sets."""

    mapping = region_feature_map()
    selected_features = [feature for region in selected_regions for feature in mapping[region]]
    train_indices = train_rows["row_index"].to_numpy(dtype=int)
    test_indices = test_rows["row_index"].to_numpy(dtype=int)
    return {
        "MRI_only": (
            table.iloc[train_indices][feature_columns("MRI")].to_numpy(),
            table.iloc[test_indices][feature_columns("MRI")].to_numpy(),
            feature_columns("MRI"),
        ),
        "PET_only": (
            table.iloc[train_indices][feature_columns("PET")].to_numpy(),
            table.iloc[test_indices][feature_columns("PET")].to_numpy(),
            feature_columns("PET"),
        ),
        "MRI_PET_raw": (
            table.iloc[train_indices][feature_columns()].to_numpy(),
            table.iloc[test_indices][feature_columns()].to_numpy(),
            feature_columns(),
        ),
        "MRI_PET_GNN_selected": (
            table.iloc[train_indices][selected_features].to_numpy(),
            table.iloc[test_indices][selected_features].to_numpy(),
            selected_features,
        ),
    }


def run_diagnostic_models(config: PipelineConfig, table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Evaluate four diagnostic feature sets on matched repeated outer folds."""

    output = config.output_dir
    assignments = pd.read_csv(output / "gnn_fold_assignments.csv")
    fold_importance = pd.read_csv(output / "gnn_fold_region_importance.csv")
    cv = config.values["cross_validation"]
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    fold_keys = assignments[["repeat", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
    for key in fold_keys.itertuples(index=False):
        fold_data = assignments[
            (assignments["repeat"] == key.repeat) & (assignments["fold"] == key.fold)
        ]
        train_rows = fold_data[fold_data["role"] == "train"].sort_values("row_index")
        test_rows = fold_data[fold_data["role"] == "test"].sort_values("row_index")
        importance = fold_importance[
            (fold_importance["repeat"] == key.repeat) & (fold_importance["fold"] == key.fold)
        ]
        selected_regions = importance[importance["selected_in_training_fold"].astype(bool)]["region"].tolist()
        feature_sets = _fold_feature_sets(table, train_rows, test_rows, selected_regions)
        y_train = train_rows["target_ad"].to_numpy(dtype=int)
        y_test = test_rows["target_ad"].to_numpy(dtype=int)
        for model_name in MODEL_ORDER:
            x_train, x_test, names = feature_sets[model_name]
            probabilities, best_c, coefficients = fit_calibrated_logistic(
                x_train,
                y_train,
                x_test,
                names,
                [float(value) for value in cv["logistic_C"]],
                int(cv["inner_folds"]),
                config.seed + int(key.repeat) * 100 + int(key.fold),
            )
            selection_rows.append(
                {
                    "repeat": key.repeat,
                    "fold": key.fold,
                    "model": model_name,
                    "best_C": best_c,
                    "features": len(names),
                    "training_participants": len(train_rows),
                    "test_participants": len(test_rows),
                }
            )
            coefficient_rows.extend(
                {
                    "repeat": key.repeat,
                    "fold": key.fold,
                    "model": model_name,
                    "feature": name,
                    "standardized_coefficient": float(coefficient),
                }
                for name, coefficient in zip(names, coefficients)
            )
            for row, probability, truth in zip(test_rows.itertuples(index=False), probabilities, y_test):
                prediction_rows.append(
                    {
                        "repeat": key.repeat,
                        "fold": key.fold,
                        "model": model_name,
                        "row_index": int(row.row_index),
                        "subject_id": row.subject_id,
                        "target_ad": int(truth),
                        "probability_ad": float(probability),
                    }
                )

    outer_predictions = pd.DataFrame(prediction_rows)
    averaged = (
        outer_predictions.groupby(["model", "subject_id", "target_ad"], as_index=False)
        .agg(probability_ad=("probability_ad", "mean"), prediction_sd=("probability_ad", "std"), repeats=("probability_ad", "size"))
    )
    summaries = []
    calibration_rows = []
    decision_rows = []
    for model_index, model_name in enumerate(MODEL_ORDER):
        subset = averaged[averaged["model"] == model_name]
        y_true = subset["target_ad"].to_numpy(dtype=int)
        probability = subset["probability_ad"].to_numpy(dtype=float)
        metrics = bootstrap_metrics(
            y_true,
            probability,
            float(cv["classification_threshold"]),
            int(cv["bootstrap_repetitions"]),
            config.seed + model_index,
        )
        metrics.insert(0, "model", model_name)
        summaries.append(metrics)
        observed, predicted = calibration_curve(y_true, probability, n_bins=5, strategy="quantile")
        calibration_rows.extend(
            {
                "model": model_name,
                "risk_bin": index + 1,
                "mean_predicted_risk": float(mean_probability),
                "observed_prevalence": float(prevalence),
            }
            for index, (prevalence, mean_probability) in enumerate(zip(observed, predicted))
        )
        curve = decision_curve(y_true, probability, np.linspace(0.1, 0.9, 33))
        curve.insert(0, "model", model_name)
        decision_rows.extend(curve.to_dict("records"))

    results = pd.concat(summaries, ignore_index=True)
    calibration_data = pd.DataFrame(calibration_rows)
    decision_data = pd.DataFrame(decision_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    paired_differences = paired_model_differences(
        averaged,
        [
            ("MRI_PET_raw", "MRI_only"),
            ("MRI_PET_GNN_selected", "MRI_PET_raw"),
            ("PET_only", "MRI_only"),
        ],
        float(cv["classification_threshold"]),
        int(cv["bootstrap_repetitions"]),
        config.seed + 1000,
    )
    outer_predictions.to_csv(output / "diagnostic_outer_predictions.csv", index=False)
    averaged.to_csv(output / "diagnostic_participant_predictions.csv", index=False)
    results.to_csv(output / "diagnostic_results_summary.csv", index=False)
    calibration_data.to_csv(output / "diagnostic_calibration_plots_data.csv", index=False)
    decision_data.to_csv(output / "diagnostic_decision_curve_data.csv", index=False)
    coefficients.to_csv(output / "diagnostic_fold_coefficients.csv", index=False)
    paired_differences.to_csv(output / "diagnostic_paired_model_differences.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "diagnostic_model_selection.csv", index=False)
    (output / "diagnostic_methodology.json").write_text(
        json.dumps(
            {
                "outer_validation": f"{cv['outer_folds']}-fold repeated {cv['outer_repeats']} times",
                "inner_validation": f"{cv['inner_folds']}-fold C tuning and training-only sigmoid calibration",
                "participant_averaging": "one probability per participant averaged across outer-test repeats",
                "confidence_intervals": f"stratified participant bootstrap, {cv['bootstrap_repetitions']} repetitions",
                "classification_threshold": cv["classification_threshold"],
                "gnn_selected_panel": "selected independently inside each outer training fold",
                "gnn_role": "training-fold regional feature ranking; pooled embeddings are not evaluated as a diagnostic model",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "results": results,
        "predictions": averaged,
        "calibration": calibration_data,
        "decision": decision_data,
        "coefficients": coefficients,
        "paired_differences": paired_differences,
    }


if __name__ == "__main__":
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    arguments = parser.parse_args()
    cfg = load_config(arguments.config)
    features = pd.read_csv(cfg.output_dir / "multimodal_features_long.csv")
    run_diagnostic_models(cfg, features)
    print(f"Saved diagnostic model outputs to {cfg.output_dir}")
