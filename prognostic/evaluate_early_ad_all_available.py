"""Use all compatible labelled data for 24-month MCI-to-AD prediction.

The outcome is conversion to a recorded AD diagnosis within 730 days after a
participant's second pre-outcome MCI MRI. The analysis uses all 159 eligible
ADNI participants and never mixes the separate ADNI2 or OASIS outcome schemas.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
DEFAULT_COHORT = ROOT / "longitudinal_mri_change_prediction_results" / "eligible_two_mri_cohort.csv"
DEFAULT_CLINICAL = ROOT / "adni_mri_clinical_conversion_fusion_results" / "matched_clinical_feature_table.csv"
DEFAULT_GNN_REGIONS = ROOT / "adni2_multimodal_gnn_pipeline" / "outputs" / "gnn_selected_regions.csv"
DEFAULT_ADNI2_FEATURES = ROOT / "adni2_multimodal_gnn_pipeline" / "outputs" / "multimodal_features_long.csv"
DEFAULT_OASIS = ROOT / "oasis_future_conversion_results" / "survival_performance_percent.csv"
DEFAULT_OUTPUT = ROOT / "early_ad_prediction_all_available"

OUTER_FOLDS = 5
OUTER_REPEATS = 5
INNER_FOLDS = 3
BOOTSTRAP_REPETITIONS = 5000
SEED = 20260815
THRESHOLD = 0.5

GNN_TRANSFER_REGIONS = [
    "lh_precuneus_volume",
    "rh_precuneus_volume",
    "lh_middletemporal_volume",
    "rh_middletemporal_volume",
    "rh_parahippocampal_volume",
    "lh_inferiortemporal_volume",
    "lh_parahippocampal_volume",
    "Right-Hippocampus",
    "rh_inferiortemporal_volume",
    "Left-Hippocampus",
    "Right-Amygdala",
    "Left-Amygdala",
    "lh_fusiform_volume",
]

AD_VULNERABLE_20 = [
    "lh_entorhinal_volume",
    "rh_entorhinal_volume",
    "lh_parahippocampal_volume",
    "rh_parahippocampal_volume",
    "Left-Hippocampus",
    "Right-Hippocampus",
    "Left-Amygdala",
    "Right-Amygdala",
    "lh_fusiform_volume",
    "rh_fusiform_volume",
    "lh_inferiortemporal_volume",
    "rh_inferiortemporal_volume",
    "lh_middletemporal_volume",
    "rh_middletemporal_volume",
    "lh_temporalpole_volume",
    "rh_temporalpole_volume",
    "lh_posteriorcingulate_volume",
    "rh_posteriorcingulate_volume",
    "lh_precuneus_volume",
    "rh_precuneus_volume",
]

NESTED_BEST_FEATURE_SET = "nested_best_available"
BASE_FEATURE_SET_ORDER = [
    "index_GNN13_MRI_age_MMSE",
    "longitudinal_GNN13_MRI",
    "longitudinal_GNN13_MRI_age_MMSE",
    "longitudinal_AD20_MRI_age_MMSE",
    "longitudinal_all82_MRI_age_MMSE",
]
FEATURE_SET_ORDER = BASE_FEATURE_SET_ORDER + [NESTED_BEST_FEATURE_SET]
PRIMARY_FEATURE_SET = NESTED_BEST_FEATURE_SET
FEATURE_SET_LABELS = {
    "index_GNN13_MRI_age_MMSE": "Index MRI GNN13 + age/MMSE",
    "longitudinal_GNN13_MRI": "Two-MRI GNN13",
    "longitudinal_GNN13_MRI_age_MMSE": "Two-MRI GNN13 + age/MMSE",
    "longitudinal_AD20_MRI_age_MMSE": "Two-MRI AD20 + age/MMSE",
    "longitudinal_all82_MRI_age_MMSE": "Two-MRI all82 + age/MMSE",
    NESTED_BEST_FEATURE_SET: "Nested best available",
}


def parse_args() -> argparse.Namespace:
    """Parse reproducible command-line settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--clinical", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--gnn-regions", type=Path, default=DEFAULT_GNN_REGIONS)
    parser.add_argument("--adni2-features", type=Path, default=DEFAULT_ADNI2_FEATURES)
    parser.add_argument("--oasis-results", type=Path, default=DEFAULT_OASIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-repeats", type=int, default=OUTER_REPEATS)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPETITIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    """Create a terminal and file logger."""

    logger = logging.getLogger("early_ad_all_available")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def canonical_adni_id(value: str) -> str:
    """Normalise ADNI identifiers for overlap auditing only."""

    return str(value).upper().replace("SUB-ADNI", "").replace("_", "")


def load_analysis_data(
    cohort_path: Path,
    clinical_path: Path,
    adni2_features_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, merge, and validate the labelled ADNI progression cohort."""

    cohort = pd.read_csv(cohort_path)
    clinical = pd.read_csv(clinical_path)
    if cohort["participant_id"].duplicated().any() or clinical["participant_id"].duplicated().any():
        raise ValueError("Participant IDs must be unique in cohort and clinical tables.")
    merged = cohort.merge(clinical.drop(columns=["partition", "label"]), on="participant_id", how="left", validate="one_to_one")
    required = {
        "participant_id", "label", "scan_interval_days", "baseline_eTIV", "followup_eTIV",
        "followup_age", "baseline_mmse", "followup_mmse", "annualized_mmse_change",
    }
    missing = sorted(required.difference(merged.columns))
    if missing:
        raise ValueError(f"Merged progression cohort is missing columns: {missing}")
    if len(merged) != 159 or int(merged["label"].sum()) != 84:
        raise ValueError("Expected 159 participants with 84 future converters.")
    clinical_columns = ["followup_age", "baseline_mmse", "followup_mmse", "annualized_mmse_change"]
    if merged[clinical_columns].isna().any().any():
        raise ValueError("Primary age/MMSE values must be complete for all participants.")

    adni2 = pd.read_csv(adni2_features_path)
    development_ids = {canonical_adni_id(value) for value in merged["participant_id"]}
    diagnostic_ids = {canonical_adni_id(value) for value in adni2["subject_id"]}
    overlap = sorted(development_ids.intersection(diagnostic_ids))
    audit = {
        "progression_participants": len(merged),
        "future_converters": int(merged["label"].sum()),
        "stable_mci": int((1 - merged["label"]).sum()),
        "adni2_diagnostic_participants": int(adni2["subject_id"].nunique()),
        "participant_overlap_with_gnn_selection_cohort": len(overlap),
        "overlapping_ids": overlap,
    }
    if overlap:
        raise ValueError("Transferred GNN region selection cohort overlaps the progression cohort.")
    return merged.sort_values("participant_id").reset_index(drop=True), audit


def all_region_features(table: pd.DataFrame) -> list[str]:
    """Return every region having both baseline and follow-up FreeSurfer volumes."""

    baseline = [column.removeprefix("baseline__") for column in table if column.startswith("baseline__")]
    followup = {column.removeprefix("followup__") for column in table if column.startswith("followup__")}
    return [feature for feature in baseline if feature in followup]


def longitudinal_matrix(table: pd.DataFrame, regions: list[str]) -> tuple[np.ndarray, list[str]]:
    """Create eTIV-normalised baseline, index, and annualised change features."""

    baseline = table[[f"baseline__{region}" for region in regions]].to_numpy(dtype=float)
    followup = table[[f"followup__{region}" for region in regions]].to_numpy(dtype=float)
    baseline_etiv = table["baseline_eTIV"].to_numpy(dtype=float).reshape(-1, 1)
    followup_etiv = table["followup_eTIV"].to_numpy(dtype=float).reshape(-1, 1)
    years = table["scan_interval_days"].to_numpy(dtype=float).reshape(-1, 1) / 365.25
    baseline = baseline / np.where(baseline_etiv > 0, baseline_etiv, np.nan)
    followup = followup / np.where(followup_etiv > 0, followup_etiv, np.nan)
    annual_change = (followup - baseline) / years
    annual_percent = 100.0 * (followup - baseline) / np.where(np.abs(baseline) > 1e-12, np.abs(baseline), np.nan) / years
    matrix = np.column_stack([baseline, followup, annual_change, annual_percent])
    names = (
        [f"baseline_{region}_icv_fraction" for region in regions]
        + [f"index_{region}_icv_fraction" for region in regions]
        + [f"annual_change_{region}" for region in regions]
        + [f"annual_percent_change_{region}" for region in regions]
    )
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, names


def index_matrix(table: pd.DataFrame, regions: list[str]) -> tuple[np.ndarray, list[str]]:
    """Create one current/index MRI feature vector for deployment-style comparison."""

    followup = table[[f"followup__{region}" for region in regions]].to_numpy(dtype=float)
    etiv = table["followup_eTIV"].to_numpy(dtype=float).reshape(-1, 1)
    matrix = followup / np.where(etiv > 0, etiv, np.nan)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, [f"index_{region}_icv_fraction" for region in regions]


def build_feature_sets(table: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Build all prespecified compatible feature panels."""

    available = set(all_region_features(table))
    for name, regions in (("GNN transfer", GNN_TRANSFER_REGIONS), ("AD20", AD_VULNERABLE_20)):
        missing = sorted(set(regions).difference(available))
        if missing:
            raise ValueError(f"{name} regions are unavailable: {missing}")
    all_regions = all_region_features(table)
    if len(all_regions) != 82:
        raise ValueError(f"Expected 82 paired FreeSurfer regions, found {len(all_regions)}")
    clinical = table[["followup_age", "baseline_mmse", "followup_mmse", "annualized_mmse_change"]].to_numpy(float)
    clinical_names = ["index_age", "baseline_mmse", "index_mmse", "annualized_mmse_change"]
    interval = (table["scan_interval_days"].to_numpy(float) / 365.25).reshape(-1, 1)

    index_gnn, index_gnn_names = index_matrix(table, GNN_TRANSFER_REGIONS)
    longitudinal_gnn, longitudinal_gnn_names = longitudinal_matrix(table, GNN_TRANSFER_REGIONS)
    longitudinal_ad20, longitudinal_ad20_names = longitudinal_matrix(table, AD_VULNERABLE_20)
    longitudinal_all, longitudinal_all_names = longitudinal_matrix(table, all_regions)
    matrices = {
        "index_GNN13_MRI_age_MMSE": np.column_stack([index_gnn, clinical[:, [0, 2]]]),
        "longitudinal_GNN13_MRI": np.column_stack([longitudinal_gnn, interval]),
        "longitudinal_GNN13_MRI_age_MMSE": np.column_stack([longitudinal_gnn, interval, clinical]),
        "longitudinal_AD20_MRI_age_MMSE": np.column_stack([longitudinal_ad20, interval, clinical]),
        "longitudinal_all82_MRI_age_MMSE": np.column_stack([longitudinal_all, interval, clinical]),
    }
    names = {
        "index_GNN13_MRI_age_MMSE": index_gnn_names + ["index_age", "index_mmse"],
        "longitudinal_GNN13_MRI": longitudinal_gnn_names + ["scan_interval_years"],
        "longitudinal_GNN13_MRI_age_MMSE": longitudinal_gnn_names + ["scan_interval_years"] + clinical_names,
        "longitudinal_AD20_MRI_age_MMSE": longitudinal_ad20_names + ["scan_interval_years"] + clinical_names,
        "longitudinal_all82_MRI_age_MMSE": longitudinal_all_names + ["scan_interval_years"] + clinical_names,
    }
    return matrices, names


def build_feature_matrix(table: pd.DataFrame, feature_set: str) -> tuple[np.ndarray, list[str]]:
    """Build only the feature panel required by one fitted deployment artifact."""

    if feature_set == "index_GNN13_MRI_age_MMSE":
        mri, names = index_matrix(table, GNN_TRANSFER_REGIONS)
        clinical = table[["followup_age", "followup_mmse"]].to_numpy(float)
        return np.column_stack([mri, clinical]), names + ["index_age", "index_mmse"]
    region_lookup = {
        "longitudinal_GNN13_MRI": GNN_TRANSFER_REGIONS,
        "longitudinal_GNN13_MRI_age_MMSE": GNN_TRANSFER_REGIONS,
        "longitudinal_AD20_MRI_age_MMSE": AD_VULNERABLE_20,
        "longitudinal_all82_MRI_age_MMSE": all_region_features(table),
    }
    if feature_set not in region_lookup:
        raise ValueError(f"Unsupported fitted feature set: {feature_set}")
    mri, names = longitudinal_matrix(table, region_lookup[feature_set])
    interval = (table["scan_interval_days"].to_numpy(float) / 365.25).reshape(-1, 1)
    matrix = np.column_stack([mri, interval])
    names = names + ["scan_interval_years"]
    if feature_set != "longitudinal_GNN13_MRI":
        clinical_names = ["index_age", "baseline_mmse", "index_mmse", "annualized_mmse_change"]
        clinical = table[["followup_age", "baseline_mmse", "followup_mmse", "annualized_mmse_change"]].to_numpy(float)
        matrix = np.column_stack([matrix, clinical])
        names += clinical_names
    return matrix, names


def required_prediction_columns(table: pd.DataFrame, feature_set: str) -> list[str]:
    """Return raw columns needed to reproduce one fitted feature panel."""

    common = ["participant_id"]
    if feature_set == "index_GNN13_MRI_age_MMSE":
        return common + ["followup_eTIV", "followup_age", "followup_mmse"] + [
            f"followup__{region}" for region in GNN_TRANSFER_REGIONS
        ]
    if feature_set == "longitudinal_GNN13_MRI":
        regions = GNN_TRANSFER_REGIONS
        clinical: list[str] = []
    elif feature_set == "longitudinal_GNN13_MRI_age_MMSE":
        regions = GNN_TRANSFER_REGIONS
        clinical = ["followup_age", "baseline_mmse", "followup_mmse", "annualized_mmse_change"]
    elif feature_set == "longitudinal_AD20_MRI_age_MMSE":
        regions = AD_VULNERABLE_20
        clinical = ["followup_age", "baseline_mmse", "followup_mmse", "annualized_mmse_change"]
    elif feature_set == "longitudinal_all82_MRI_age_MMSE":
        regions = all_region_features(table)
        clinical = ["followup_age", "baseline_mmse", "followup_mmse", "annualized_mmse_change"]
    else:
        raise ValueError(f"Unsupported fitted feature set: {feature_set}")
    return (
        common
        + ["scan_interval_days", "baseline_eTIV", "followup_eTIV"]
        + clinical
        + [f"baseline__{region}" for region in regions]
        + [f"followup__{region}" for region in regions]
    )


def candidate_models(seed: int) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Return compact, prespecified small-cohort model grids."""

    linear = [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler(quantile_range=(5, 95)))]
    return {
        "L2 logistic": (
            Pipeline(
                linear
                + [
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced", solver="liblinear", max_iter=10000, random_state=seed
                        ),
                    )
                ]
            ),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Elastic-net logistic": (
            Pipeline(
                linear
                + [
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced", solver="saga", max_iter=10000, random_state=seed
                        ),
                    )
                ]
            ),
            {"classifier__C": [0.1, 1.0], "classifier__l1_ratio": [0.5, 1.0]},
        ),
        "RBF SVM": (
            Pipeline(
                linear
                + [
                    (
                        "classifier",
                        SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=seed),
                    )
                ]
            ),
            {"classifier__C": [0.1, 1.0], "classifier__gamma": ["scale", 0.01]},
        ),
        "Random forest": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=150,
                            class_weight="balanced_subsample",
                            max_features="sqrt",
                            n_jobs=1,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {"classifier__max_depth": [3, None], "classifier__min_samples_leaf": [3, 8]},
        ),
    }


def select_model(x: np.ndarray, y: np.ndarray, seed: int, inner_folds: int = INNER_FOLDS) -> dict[str, Any]:
    """Select model family and hyperparameters using training data only."""

    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    candidates = []
    for order, (name, (estimator, grid)) in enumerate(candidate_models(seed).items()):
        search = GridSearchCV(
            estimator,
            grid,
            scoring="roc_auc",
            cv=inner,
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(x, y)
        candidates.append(
            {
                "model": name,
                "order": order,
                "best_score": float(search.best_score_),
                "best_params": search.best_params_,
                "estimator": clone(search.best_estimator_),
            }
        )
    return sorted(candidates, key=lambda row: (-row["best_score"], row["order"]))[0]


def fit_calibrated(selection: dict[str, Any], x: np.ndarray, y: np.ndarray, seed: int, folds: int) -> CalibratedClassifierCV:
    """Fit sigmoid calibration within the supplied development data."""

    calibrator = CalibratedClassifierCV(
        estimator=clone(selection["estimator"]),
        method="sigmoid",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
    )
    calibrator.fit(x, y)
    return calibrator


def calibration_parameters(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Estimate logistic calibration intercept and slope."""

    predictor = logit(np.clip(probability, 1e-6, 1 - 1e-6))

    def objective(parameters: np.ndarray) -> float:
        fitted = expit(parameters[0] + parameters[1] * predictor)
        return -float(np.sum(y_true * np.log(fitted + 1e-12) + (1 - y_true) * np.log(1 - fitted + 1e-12)))

    result = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    return (float(result.x[0]), float(result.x[1])) if result.success else (np.nan, np.nan)


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = THRESHOLD) -> dict[str, float | int]:
    """Calculate classification, discrimination, and calibration values."""

    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    intercept, slope = calibration_parameters(y_true, probability)
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "roc_auc": roc_auc_score(y_true, probability),
        "sensitivity": recall_score(y_true, prediction, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, prediction, pos_label=0, zero_division=0),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "brier_score": brier_score_loss(y_true, probability),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _bootstrap_metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Return metrics that are stable and efficient for bootstrap inference."""

    prediction = (probability >= THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "roc_auc": roc_auc_score(y, probability),
        "sensitivity": recall_score(y, prediction, pos_label=1, zero_division=0),
        "specificity": recall_score(y, prediction, pos_label=0, zero_division=0),
        "precision": precision_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "brier_score": brier_score_loss(y, probability),
    }


def bootstrap_summary(y: np.ndarray, probability: np.ndarray, repetitions: int, seed: int) -> pd.DataFrame:
    """Calculate stratified participant-bootstrap confidence intervals."""

    point = metrics(y, probability)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    boot = {metric: [] for metric in _bootstrap_metric_values(y, probability)}
    for _ in range(repetitions):
        sample = np.concatenate(
            [rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)]
        )
        values = _bootstrap_metric_values(y[sample], probability[sample])
        for metric, value in values.items():
            boot[metric].append(value)
    rows = [
        {
            "metric": metric,
            "estimate": float(point[metric]),
            "ci_lower": float(np.percentile(values, 2.5)),
            "ci_upper": float(np.percentile(values, 97.5)),
        }
        for metric, values in boot.items()
    ]
    rows.extend(
        [
            {"metric": "calibration_intercept", "estimate": point["calibration_intercept"], "ci_lower": np.nan, "ci_upper": np.nan},
            {"metric": "calibration_slope", "estimate": point["calibration_slope"], "ci_lower": np.nan, "ci_upper": np.nan},
        ]
    )
    return pd.DataFrame(rows)


def paired_differences(
    averaged: pd.DataFrame,
    comparator: str,
    reference: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Compute paired bootstrap differences on the same participants."""

    wide = averaged.pivot(index=["participant_id", "label"], columns="feature_set", values="probability").reset_index()
    y = wide["label"].to_numpy(int)
    left = wide[comparator].to_numpy(float)
    right = wide[reference].to_numpy(float)
    point_left = _bootstrap_metric_values(y, left)
    point_right = _bootstrap_metric_values(y, right)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in point_left}
    for _ in range(repetitions):
        indices = np.concatenate(
            [rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)]
        )
        left_values = _bootstrap_metric_values(y[indices], left[indices])
        right_values = _bootstrap_metric_values(y[indices], right[indices])
        for metric in samples:
            samples[metric].append(left_values[metric] - right_values[metric])
    return pd.DataFrame(
        [
            {
                "comparator": comparator,
                "reference": reference,
                "metric": metric,
                "difference": point_left[metric] - point_right[metric],
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
                "paired_bootstrap_p": float(
                    min(1.0, 2 * min((np.sum(np.asarray(values) <= 0) + 1) / (repetitions + 1), (np.sum(np.asarray(values) >= 0) + 1) / (repetitions + 1)))
                ),
            }
            for metric, values in samples.items()
        ]
    )


def run_nested_validation(
    table: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    repeats: int,
    seed: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run matched repeated nested validation for every prespecified feature set."""

    labels = table["label"].to_numpy(int)
    splitter = RepeatedStratifiedKFold(n_splits=OUTER_FOLDS, n_repeats=repeats, random_state=seed)
    prediction_rows = []
    selection_rows = []
    for feature_index, feature_set in enumerate(BASE_FEATURE_SET_ORDER):
        logger.info("Evaluating %s with %s features", feature_set, matrices[feature_set].shape[1])
        for split_index, (train, test) in enumerate(splitter.split(matrices[feature_set], labels)):
            repeat = split_index // OUTER_FOLDS + 1
            fold = split_index % OUTER_FOLDS + 1
            fold_seed = seed + feature_index * 1000 + split_index
            selection = select_model(matrices[feature_set][train], labels[train], fold_seed)
            model = fit_calibrated(selection, matrices[feature_set][train], labels[train], fold_seed + 100_000, INNER_FOLDS)
            probability = model.predict_proba(matrices[feature_set][test])[:, 1]
            selection_rows.append(
                {
                    "feature_set": feature_set,
                    "selected_input_feature_set": feature_set,
                    "repeat": repeat,
                    "fold": fold,
                    "selected_model": selection["model"],
                    "inner_roc_auc": selection["best_score"],
                    "selected_parameters": json.dumps(selection["best_params"], sort_keys=True),
                    "training_participants": len(train),
                    "test_participants": len(test),
                }
            )
            prediction_rows.extend(
                {
                    "feature_set": feature_set,
                    "repeat": repeat,
                    "fold": fold,
                    "participant_id": table.iloc[index]["participant_id"],
                    "label": int(labels[index]),
                    "probability": float(probability[offset]),
                }
                for offset, index in enumerate(test)
            )
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    composite_predictions = []
    composite_selections = []
    feature_priority = {name: index for index, name in enumerate(BASE_FEATURE_SET_ORDER)}
    for (repeat, fold), candidates in selections.groupby(["repeat", "fold"], sort=True):
        choice = (
            candidates.assign(priority=candidates["feature_set"].map(feature_priority))
            .sort_values(["inner_roc_auc", "priority"], ascending=[False, True])
            .iloc[0]
        )
        selected_predictions = predictions[
            (predictions["repeat"] == repeat)
            & (predictions["fold"] == fold)
            & (predictions["feature_set"] == choice["feature_set"])
        ].copy()
        selected_predictions["feature_set"] = NESTED_BEST_FEATURE_SET
        composite_predictions.extend(selected_predictions.to_dict("records"))
        composite_selections.append(
            {
                "feature_set": NESTED_BEST_FEATURE_SET,
                "selected_input_feature_set": choice["feature_set"],
                "repeat": repeat,
                "fold": fold,
                "selected_model": choice["selected_model"],
                "inner_roc_auc": choice["inner_roc_auc"],
                "selected_parameters": choice["selected_parameters"],
                "training_participants": choice["training_participants"],
                "test_participants": choice["test_participants"],
            }
        )
    predictions = pd.concat([predictions, pd.DataFrame(composite_predictions)], ignore_index=True)
    selections = pd.concat([selections, pd.DataFrame(composite_selections)], ignore_index=True)
    return predictions, selections


def summarise_predictions(
    predictions: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Average repeats and calculate metrics, calibration bins, and paired contrasts."""

    averaged = (
        predictions.groupby(["feature_set", "participant_id", "label"], as_index=False)
        .agg(probability=("probability", "mean"), probability_sd=("probability", "std"), outer_predictions=("probability", "size"))
    )
    summaries = []
    calibration_rows = []
    for feature_index, feature_set in enumerate(FEATURE_SET_ORDER):
        subset = averaged[averaged["feature_set"] == feature_set]
        summary = bootstrap_summary(
            subset["label"].to_numpy(int), subset["probability"].to_numpy(float), bootstrap, seed + feature_index
        )
        summary.insert(0, "feature_set", feature_set)
        summary.insert(1, "feature_set_label", FEATURE_SET_LABELS[feature_set])
        summaries.append(summary)
        observed, predicted = calibration_curve(subset["label"], subset["probability"], n_bins=5, strategy="quantile")
        calibration_rows.extend(
            {
                "feature_set": feature_set,
                "risk_bin": index + 1,
                "mean_predicted_probability": float(mean_probability),
                "observed_conversion_proportion": float(prevalence),
            }
            for index, (prevalence, mean_probability) in enumerate(zip(observed, predicted))
        )
    comparisons = []
    for index, reference in enumerate(feature_set for feature_set in FEATURE_SET_ORDER if feature_set != PRIMARY_FEATURE_SET):
        comparisons.append(paired_differences(averaged, PRIMARY_FEATURE_SET, reference, bootstrap, seed + 100 + index))
    return averaged, pd.concat(summaries, ignore_index=True), pd.concat(comparisons, ignore_index=True), pd.DataFrame(calibration_rows)


def fit_final_model(
    table: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    names: dict[str, list[str]],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Fit a research-use calibrated primary model on all labelled participants."""

    labels = table["label"].to_numpy(int)
    candidates = []
    for feature_index, feature_set in enumerate(BASE_FEATURE_SET_ORDER):
        selection = select_model(
            matrices[feature_set], labels, seed + 900_000 + feature_index * 1000, inner_folds=5
        )
        candidates.append(selection | {"feature_set": feature_set, "priority": feature_index})
    selection = sorted(candidates, key=lambda row: (-row["best_score"], row["priority"]))[0]
    selected_feature_set = str(selection["feature_set"])
    model = fit_calibrated(selection, matrices[selected_feature_set], labels, seed + 910_000, folds=5)
    artifact = {
        "model": model,
        "feature_set": selected_feature_set,
        "feature_names": names[selected_feature_set],
        "regions": GNN_TRANSFER_REGIONS,
        "selected_model_family": selection["model"],
        "selected_parameters": selection["best_params"],
        "inner_roc_auc": selection["best_score"],
        "training_participants": len(table),
        "future_converters": int(labels.sum()),
        "outcome": "AD diagnosis within 730 days after the second pre-outcome MCI MRI",
        "classification_threshold": THRESHOLD,
        "warning": "Research-use internal model; not externally validated or suitable for clinical decisions.",
    }
    joblib.dump(artifact, output_dir / "final_research_early_ad_model.joblib")
    required = required_prediction_columns(table, selected_feature_set)
    pd.DataFrame(columns=required).to_csv(output_dir / "prediction_input_template.csv", index=False)
    return artifact


def save_final_interpretability(artifact: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    """Save standardized linear coefficients when the selected model exposes them."""

    coefficient_vectors = []
    for calibrated in artifact["model"].calibrated_classifiers_:
        classifier = calibrated.estimator.named_steps["classifier"]
        if not hasattr(classifier, "coef_"):
            return pd.DataFrame()
        coefficient_vectors.append(classifier.coef_[0])
    coefficients = np.vstack(coefficient_vectors)
    summary = pd.DataFrame(
        {
            "feature": artifact["feature_names"],
            "mean_robust_scaled_coefficient": coefficients.mean(axis=0),
            "mean_absolute_coefficient": np.abs(coefficients).mean(axis=0),
            "sign_stability": np.maximum((coefficients > 0).mean(axis=0), (coefficients < 0).mean(axis=0)),
        }
    ).sort_values("mean_absolute_coefficient", ascending=False)
    summary["interpretation_warning"] = (
        "Conditional regularized association with fitted conversion log-odds; not causal or biological effect."
    )
    summary.to_csv(output_dir / "final_model_coefficients.csv", index=False)
    top = summary.head(15).sort_values("mean_robust_scaled_coefficient")
    fig, axis = plt.subplots(figsize=(10, 8))
    axis.barh(
        top["feature"].str.replace("_", " ", regex=False),
        top["mean_robust_scaled_coefficient"],
        color=np.where(top["mean_robust_scaled_coefficient"] >= 0, "#a23e48", "#1f5673"),
    )
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set(
        xlabel="Mean coefficient after robust scaling",
        ylabel="Input feature",
        title="Final Research Model: Conditional Feature Associations",
    )
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "final_model_coefficients.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def save_figures(averaged: pd.DataFrame, summary: pd.DataFrame, calibration: pd.DataFrame, output_dir: Path) -> None:
    """Save thesis-quality model-comparison figures."""

    colors = ["#1f5673", "#755c9c", "#a23e48", "#2f7d5b", "#d27c2c", "#bf9b30"]
    fig, axis = plt.subplots(figsize=(9, 7))
    for color, feature_set in zip(colors, FEATURE_SET_ORDER):
        subset = averaged[averaged["feature_set"] == feature_set]
        false_positive, true_positive, _ = roc_curve(subset["label"], subset["probability"])
        auc = roc_auc_score(subset["label"], subset["probability"])
        axis.plot(false_positive, true_positive, color=color, linewidth=2, label=f"{FEATURE_SET_LABELS[feature_set]} ({auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Chance")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Future MCI-to-AD Conversion: Repeated Outer-Test ROC")
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "early_ad_roc_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    key_metrics = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    plot_data = summary[summary["metric"].isin(key_metrics)]
    fig, axis = plt.subplots(figsize=(13, 7))
    x = np.arange(len(key_metrics))
    width = 0.16
    for model_index, (color, feature_set) in enumerate(zip(colors, FEATURE_SET_ORDER)):
        values = plot_data[plot_data["feature_set"] == feature_set].set_index("metric").loc[key_metrics]
        position = x + (model_index - (len(FEATURE_SET_ORDER) - 1) / 2) * width
        axis.bar(position, values["estimate"], width, color=color, label=FEATURE_SET_LABELS[feature_set])
        axis.errorbar(
            position,
            values["estimate"],
            yerr=[values["estimate"] - values["ci_lower"], values["ci_upper"] - values["estimate"]],
            fmt="none",
            color="#333333",
            capsize=2,
            linewidth=0.8,
        )
    axis.set_xticks(x, ["ROC-AUC", "Balanced accuracy", "Sensitivity", "Specificity", "F1"])
    axis.set_ylim(0, 1.02)
    axis.set(ylabel="Estimate", title="Future Conversion Performance with 95% Bootstrap Intervals")
    axis.legend(frameon=False, ncol=2, fontsize=9)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "early_ad_metric_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 7))
    for color, feature_set in zip(colors, FEATURE_SET_ORDER):
        subset = calibration[calibration["feature_set"] == feature_set]
        axis.plot(
            subset["mean_predicted_probability"],
            subset["observed_conversion_proportion"],
            marker="o",
            linewidth=2,
            color=color,
            label=FEATURE_SET_LABELS[feature_set],
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Ideal")
    axis.set(xlabel="Mean predicted conversion probability", ylabel="Observed conversion proportion", title="Internal Conversion-Probability Calibration")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "early_ad_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    table: pd.DataFrame,
    audit: dict[str, Any],
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    selections: pd.DataFrame,
    artifact: dict[str, Any],
    gnn_region_path: Path,
    oasis_path: Path,
) -> None:
    """Write a cautious report separating prediction, diagnosis, and sensitivity evidence."""

    metric_names = ["accuracy", "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "precision", "f1", "brier_score"]
    table_rows = []
    for feature_set in FEATURE_SET_ORDER:
        values = summary[summary["feature_set"] == feature_set].set_index("metric")
        cells = [
            f"{100 * values.loc[metric, 'estimate']:.2f}% ({100 * values.loc[metric, 'ci_lower']:.2f}-{100 * values.loc[metric, 'ci_upper']:.2f}%)"
            if metric != "brier_score"
            else f"{values.loc[metric, 'estimate']:.3f} ({values.loc[metric, 'ci_lower']:.3f}-{values.loc[metric, 'ci_upper']:.3f})"
            for metric in metric_names
        ]
        table_rows.append(f"| {FEATURE_SET_LABELS[feature_set]} | " + " | ".join(cells) + " |")
    raw_gnn = pd.read_csv(gnn_region_path)["region"].tolist()
    excluded = [region for region in raw_gnn if "angular" in region.lower()]
    difference_auc = differences[(differences["reference"] == "longitudinal_AD20_MRI_age_MMSE") & (differences["metric"] == "roc_auc")].iloc[0]
    selection_counts = Counter(selections[selections["feature_set"] == PRIMARY_FEATURE_SET]["selected_model"])
    oasis_text = "OASIS-2 sensitivity results were unavailable."
    if oasis_path.exists():
        oasis = pd.read_csv(oasis_path)
        mri = oasis[oasis["model"] == "mri_only_nwbv"].iloc[0]
        oasis_text = (
            f"A separate OASIS-2 analysis used 83 baseline-normal participants with 13 future dementia events. "
            f"Its MRI-only nWBV C-index was {mri.c_index_percent:.2f}% "
            f"(95% CI {mri.ci_95_lower_percent:.2f}-{mri.ci_95_upper_percent:.2f}%). "
            "This is contextual sensitivity evidence, not pooled training or external validation of the ADNI model."
        )
    coefficient_text = "The selected final model does not expose linear coefficients."
    coefficient_path = output_dir / "final_model_coefficients.csv"
    if coefficient_path.exists():
        coefficient_table = pd.read_csv(coefficient_path).head(10)
        coefficient_text = "\n".join(
            f"- `{row.feature}`: coefficient {row.mean_robust_scaled_coefficient:+.3f}; sign stability {row.sign_stability:.0%}."
            for row in coefficient_table.itertuples(index=False)
        )
    report = f"""# All-Compatible-Data Early Alzheimer Prediction Report

## What Was Predicted

The primary target is a **future recorded AD diagnosis within 730 days after the second pre-outcome MCI MRI**. This differs from the AD-versus-CN diagnostic model, which recognises an existing diagnosis.

## Data Used

- ADNI MCI participants: {len(table)}.
- Future converters: {int(table['label'].sum())}.
- Stable MCI controls with sufficient follow-up: {int((1 - table['label']).sum())}.
- All eligible participants were used through five-fold outer validation repeated five times; no participant was discarded into one small permanent test set.
- Participant overlap with the independent ADNI2 AD-versus-CN GNN region-selection cohort: {audit['participant_overlap_with_gnn_selection_cohort']}.
- Synthetic augmentation: none.
- PET prognosis input: not used, because the available ADNI2 MCI PET cases do not have dated future conversion/censoring outcomes.

## Transferred GNN Panel

The ADNI2 multimodal diagnostic GNN selected 15 regions without access to these progression labels. Thirteen had exact FreeSurfer matches and were transferred: precuneus, middle temporal, parahippocampal, inferior temporal, hippocampal, amygdala, and left fusiform regions. The unmatched AAL angular nodes ({', '.join(excluded)}) were not approximated with a different anatomical label.

## Repeated Nested Internal Results

| Feature set | Accuracy | Balanced accuracy | ROC-AUC | Sensitivity | Specificity | Precision | F1 | Brier score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

For the primary nested-best pipeline, outer-fold model-family selection frequencies were: {', '.join(f'{name}: {count}' for name, count in selection_counts.items())}. Feature panels were also selected inside each outer training fold.

Nested best available minus AD20 plus age/MMSE ROC-AUC was {100 * difference_auc.difference:+.2f} percentage points (95% paired interval {100 * difference_auc.ci_lower:+.2f} to {100 * difference_auc.ci_upper:+.2f}; p={difference_auc.paired_bootstrap_p:.3f}). Superiority is not claimed when this interval includes zero.

## Final Research Model

- Full-cohort selected feature set: {FEATURE_SET_LABELS[artifact['feature_set']]}.
- Selected full-cohort model family: {artifact['selected_model_family']}.
- Full-cohort internal selection ROC-AUC: {100 * artifact['inner_roc_auc']:.2f}% (model-selection estimate, not an external test result).
- Output: a calibrated research probability of conversion within 730 days, plus a thresholded research classification at 0.5.

### Conditional Feature Associations

{coefficient_text}

These regularized coefficients describe conditional model behaviour after robust scaling. They are not causal effects, and correlated predictors can produce counterintuitive signs.

## Separate OASIS Evidence

{oasis_text}

## Thesis-Safe Conclusion

This analysis constitutes **internal future-conversion prediction** because MRI and clinical inputs precede the recorded 24-month outcome. It does not predict neuropathologically confirmed Alzheimer disease, does not establish clinical utility, and has no compatible external PET/MRI conversion cohort. The ADNI2 multimodal data strengthen region selection and current-diagnosis modelling, but cannot validate future PET-based conversion until dated longitudinal outcomes are supplied.
"""
    (output_dir / "EARLY_AD_PREDICTION_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Run feature transfer, nested validation, final fitting, and reporting."""

    args = parse_args()
    if args.outer_repeats < 1 or args.bootstrap < 100:
        raise ValueError("Use at least one outer repeat and 100 bootstrap samples.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir)
    table, audit = load_analysis_data(args.cohort, args.clinical, args.adni2_features)
    gnn_regions = pd.read_csv(args.gnn_regions)
    expected_transfer_names = {
        "Left precuneus", "Right precuneus", "Left middle temporal gyrus", "Right middle temporal gyrus",
        "Right parahippocampal gyrus", "Left inferior temporal gyrus", "Left parahippocampal gyrus",
        "Right hippocampus", "Right inferior temporal gyrus", "Left hippocampus", "Right amygdala",
        "Left amygdala", "Left fusiform gyrus",
    }
    if not expected_transfer_names.issubset(set(gnn_regions["region"])):
        raise ValueError("The saved GNN-selected panel changed; review the transfer mapping.")
    matrices, names = build_feature_sets(table)
    audit["all_mri_regions"] = len(all_region_features(table))
    audit["transferred_exact_regions"] = len(GNN_TRANSFER_REGIONS)
    audit["feature_dimensions"] = {feature_set: matrix.shape[1] for feature_set, matrix in matrices.items()}
    (args.output_dir / "data_and_overlap_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    feature_dictionary_rows = [
            {
                "feature_set": feature_set,
                "feature_set_label": FEATURE_SET_LABELS[feature_set],
                "feature_count": len(names[feature_set]),
                "features": json.dumps(names[feature_set]),
            }
            for feature_set in BASE_FEATURE_SET_ORDER
        ]
    feature_dictionary_rows.append(
        {
            "feature_set": NESTED_BEST_FEATURE_SET,
            "feature_set_label": FEATURE_SET_LABELS[NESTED_BEST_FEATURE_SET],
            "feature_count": "selected inside each training fold",
            "features": json.dumps(BASE_FEATURE_SET_ORDER),
        }
    )
    pd.DataFrame(feature_dictionary_rows).to_csv(args.output_dir / "feature_set_dictionary.csv", index=False)
    pd.DataFrame(
        {
            "transfer_order": np.arange(1, len(GNN_TRANSFER_REGIONS) + 1),
            "freesurfer_feature": GNN_TRANSFER_REGIONS,
        }
    ).to_csv(args.output_dir / "gnn_transferred_regions.csv", index=False)

    predictions, selections = run_nested_validation(table, matrices, args.outer_repeats, args.seed, logger)
    averaged, summary, differences, calibration = summarise_predictions(
        predictions, args.bootstrap, args.seed
    )
    predictions.to_csv(args.output_dir / "outer_fold_predictions.csv", index=False)
    selections.to_csv(args.output_dir / "outer_fold_model_selection.csv", index=False)
    averaged.to_csv(args.output_dir / "participant_averaged_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "performance_with_confidence_intervals.csv", index=False)
    differences.to_csv(args.output_dir / "paired_feature_set_differences.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_plot_data.csv", index=False)
    confusion_rows = []
    for feature_set in FEATURE_SET_ORDER:
        subset = averaged[averaged["feature_set"] == feature_set]
        values = metrics(subset["label"].to_numpy(int), subset["probability"].to_numpy(float))
        confusion_rows.append(
            {
                "feature_set": feature_set,
                "true_negative": values["true_negative"],
                "false_positive": values["false_positive"],
                "false_negative": values["false_negative"],
                "true_positive": values["true_positive"],
            }
        )
    pd.DataFrame(confusion_rows).to_csv(args.output_dir / "confusion_matrices.csv", index=False)

    artifact = fit_final_model(table, matrices, names, args.output_dir, args.seed)
    save_final_interpretability(artifact, args.output_dir)
    save_figures(averaged, summary, calibration, args.output_dir)
    write_report(
        args.output_dir,
        table,
        audit,
        summary,
        differences,
        selections,
        artifact,
        args.gnn_regions,
        args.oasis_results,
    )
    protocol = {
        "outcome": artifact["outcome"],
        "outer_folds": OUTER_FOLDS,
        "outer_repeats": args.outer_repeats,
        "inner_folds": INNER_FOLDS,
        "bootstrap_repetitions": args.bootstrap,
        "probability_calibration": "sigmoid calibration inside outer training data",
        "classification_threshold": THRESHOLD,
        "primary_feature_set": PRIMARY_FEATURE_SET,
        "primary_selection": "feature panel and model family selected inside each outer training fold",
        "participant_level_splits": True,
        "augmentation": False,
        "pet_exclusion": "No dated future outcomes for the paired ADNI2 MCI PET cohort",
        "random_seed": args.seed,
    }
    (args.output_dir / "evaluation_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    logger.info("Completed all-compatible-data early AD prediction outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
