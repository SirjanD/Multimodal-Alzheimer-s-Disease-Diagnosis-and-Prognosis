"""Evaluate MRI-plus-clinical prediction of MCI-to-AD conversion in ADNI.

The primary fusion model combines the fixed 20-region longitudinal MRI panel
with age and MMSE measures available at the two pre-outcome MCI visits. CDR and
CDR Sum of Boxes are retained only as a sensitivity feature set because these
clinical ratings may overlap with diagnostic decision-making. The original
participant split is preserved, and repeated nested cross-validation provides
a separate internal stability estimate. This remains exploratory research, not
clinical validation.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
for thread_variable in (
    "LOKY_MAX_CPU_COUNT",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
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
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate_affected_region_mci_conversion import (
    load_partitions,
    panel_region_specs,
)
from evaluate_affected_region_robust_alternatives import (
    METRICS,
    binary_metrics,
    bootstrap_intervals,
    choose_validation_threshold,
)
from evaluate_longitudinal_mri_change_prediction import raw_dynamic_matrices


FEATURE_SET_LABELS = {
    "mri_only": "MRI only",
    "mri_age_mmse": "MRI + age + MMSE",
    "mri_age_mmse_cdr_sensitivity": "MRI + age + MMSE + CDR sensitivity",
}


def parse_args() -> argparse.Namespace:
    """Parse reproducible fusion experiment settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=PROJECT_ROOT
        / "longitudinal_mri_change_prediction_results"
        / "eligible_two_mri_cohort.csv",
    )
    parser.add_argument(
        "--diagnosis-files",
        type=Path,
        nargs=3,
        default=[
            PROJECT_ROOT / "Train_diagnosis_ADNI.tsv",
            PROJECT_ROOT / "Val_diagnosis_ADNI.tsv",
            PROJECT_ROOT / "Test_diagnosis_ADNI.tsv",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "adni_mri_clinical_conversion_fusion_results",
    )
    parser.add_argument("--outer-repeats", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_visit_clinical_data(paths: list[Path]) -> pd.DataFrame:
    """Load and validate visit-level age and cognitive measurements."""

    required = {
        "participant_id",
        "session_id",
        "examination_date",
        "age",
        "mmse",
        "cdr",
        "cdr_sb",
    }
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Diagnosis table does not exist: {path}")
        frame = pd.read_csv(path, sep="\t")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing clinical fields: {missing}")
        frames.append(frame[list(required)].copy())
    clinical = pd.concat(frames, ignore_index=True)
    clinical["examination_date"] = pd.to_datetime(
        clinical["examination_date"], errors="coerce"
    )
    value_columns = ["age", "mmse", "cdr", "cdr_sb"]
    for column in value_columns:
        clinical[column] = pd.to_numeric(clinical[column], errors="coerce")

    key = ["participant_id", "session_id"]
    conflicts = clinical.groupby(key, dropna=False)[value_columns].nunique(dropna=False)
    if (conflicts > 1).any(axis=None):
        raise ValueError("Conflicting clinical values exist for a participant visit")
    return (
        clinical.sort_values("examination_date")
        .drop_duplicates(key, keep="last")
        .reset_index(drop=True)
    )


def attach_clinical_features(
    cohort: pd.DataFrame, clinical: pd.DataFrame
) -> pd.DataFrame:
    """Match clinical values to each participant's two selected MRI visits."""

    original_ids = cohort["participant_id"].tolist()
    fields = ["age", "mmse", "cdr", "cdr_sb"]
    baseline = clinical[["participant_id", "session_id", *fields]].rename(
        columns={
            "session_id": "baseline_session_id",
            **{field: f"baseline_{field}" for field in fields},
        }
    )
    followup = clinical[["participant_id", "session_id", *fields]].rename(
        columns={
            "session_id": "followup_session_id",
            **{field: f"followup_{field}" for field in fields},
        }
    )
    enriched = cohort.merge(
        baseline,
        on=["participant_id", "baseline_session_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).merge(
        followup,
        on=["participant_id", "followup_session_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if enriched["participant_id"].tolist() != original_ids:
        raise ValueError("Clinical matching changed participant order")
    required_clinical = [
        f"{visit}_{field}"
        for visit in ("baseline", "followup")
        for field in fields
    ]
    if enriched[required_clinical].isna().any().any():
        missing = enriched.loc[
            enriched[required_clinical].isna().any(axis=1), "participant_id"
        ].tolist()
        raise ValueError(f"Clinical fields are incomplete for participants: {missing}")

    years = enriched["scan_interval_days"].to_numpy(float) / 365.25
    if np.any(~np.isfinite(years)) or np.any(years <= 0):
        raise ValueError("Scan intervals must be positive for clinical change features")
    for field in ("mmse", "cdr", "cdr_sb"):
        enriched[f"annualized_{field}_change"] = (
            enriched[f"followup_{field}"] - enriched[f"baseline_{field}"]
        ) / years
    return enriched


def feature_matrices(
    cohort: pd.DataFrame, region_specs: list
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Create prespecified MRI-only and MRI-plus-clinical matrices."""

    raw_mri = raw_dynamic_matrices(cohort, region_specs)
    mri = raw_mri.reshape(len(cohort), -1)
    channel_names = [
        "baseline_normalized_volume",
        "followup_normalized_volume",
        "annualized_normalized_volume_change",
        "annualized_percent_change",
    ]
    mri_names = [
        f"{channel}__{spec.region_name}"
        for channel in channel_names
        for spec in region_specs
    ]
    primary_clinical_names = [
        "followup_age",
        "baseline_mmse",
        "followup_mmse",
        "annualized_mmse_change",
    ]
    cdr_names = [
        "baseline_cdr",
        "followup_cdr",
        "annualized_cdr_change",
        "baseline_cdr_sb",
        "followup_cdr_sb",
        "annualized_cdr_sb_change",
    ]
    primary_clinical = cohort[primary_clinical_names].to_numpy(float)
    cdr = cohort[cdr_names].to_numpy(float)
    matrices = {
        "mri_only": mri,
        "mri_age_mmse": np.concatenate([mri, primary_clinical], axis=1),
        "mri_age_mmse_cdr_sensitivity": np.concatenate(
            [mri, primary_clinical, cdr], axis=1
        ),
    }
    names = {
        "mri_only": mri_names,
        "mri_age_mmse": mri_names + primary_clinical_names,
        "mri_age_mmse_cdr_sensitivity": mri_names
        + primary_clinical_names
        + cdr_names,
    }
    return matrices, names


def candidate_models(
    seed: int, compact: bool = False
) -> dict[str, tuple[object, dict[str, list]]]:
    """Return conventional small-sample classifiers and compact tuning grids."""

    linear_steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    elastic_c = [0.1, 1.0] if compact else [0.01, 0.1, 1.0, 10.0]
    elastic_ratio = [0.0, 0.5] if compact else [0.0, 0.5, 1.0]
    svm_c = [0.1, 1.0] if compact else [0.1, 1.0, 10.0]
    svm_gamma = [0.01, "scale"] if compact else [0.001, 0.01, "scale"]
    boosting_leaves = [5] if compact else [5, 9]
    return {
        "Elastic-net logistic": (
            Pipeline(
                linear_steps
                + [
                    (
                        "classifier",
                        LogisticRegression(
                            solver="saga",
                            class_weight="balanced",
                            max_iter=10000,
                            random_state=seed,
                        ),
                    )
                ]
            ),
            {
                "classifier__C": elastic_c,
                "classifier__l1_ratio": elastic_ratio,
            },
        ),
        "RBF SVM": (
            Pipeline(
                linear_steps
                + [
                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            class_weight="balanced",
                            probability=True,
                            random_state=seed,
                        ),
                    )
                ]
            ),
            {
                "classifier__C": svm_c,
                "classifier__gamma": svm_gamma,
            },
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
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "classifier__max_depth": [2, None],
                "classifier__min_samples_leaf": [2, 8],
                "classifier__max_features": ["sqrt"],
            },
        ),
        "Histogram gradient boosting": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            class_weight="balanced",
                            max_iter=150,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                "classifier__learning_rate": [0.03, 0.08],
                "classifier__max_leaf_nodes": boosting_leaves,
                "classifier__l2_regularization": [0.5, 2.0],
            },
        ),
    }


def percent_metrics_row(
    feature_set: str,
    model: str,
    training_cv_auc: float,
    parameters: dict,
    metrics: dict[str, float | int],
) -> dict[str, object]:
    """Format model-selection metrics as review-friendly percentages."""

    row: dict[str, object] = {
        "feature_set": feature_set,
        "feature_set_label": FEATURE_SET_LABELS[feature_set],
        "model": model,
        "training_cv_auc_percent": 100.0 * training_cv_auc,
        "selected_hyperparameters": json.dumps(parameters, sort_keys=True),
        "validation_threshold_percent": 100.0 * float(metrics["threshold"]),
        "participants": metrics["participants"],
        "converters": metrics["converters"],
        "stable_mci": metrics["stable_mci"],
        "true_negative": metrics["true_negative"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "true_positive": metrics["true_positive"],
    }
    for metric in METRICS:
        row[f"{metric}_percent"] = 100.0 * float(metrics[metric])
    return row


def tune_feature_sets(
    matrices: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Tune candidates on training and select one per feature set on validation."""

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    rows = []
    records: dict[str, dict[str, object]] = {}
    for feature_set in FEATURE_SET_LABELS:
        print(f"Tuning {FEATURE_SET_LABELS[feature_set]}", flush=True)
        for model_name, (estimator, grid) in candidate_models(seed).items():
            search = GridSearchCV(
                estimator,
                grid,
                scoring="roc_auc",
                cv=cv,
                refit=True,
                n_jobs=1,
                error_score="raise",
            )
            search.fit(matrices["train"][feature_set], labels["train"])
            probability = search.predict_proba(
                matrices["validation"][feature_set]
            )[:, 1]
            threshold = choose_validation_threshold(labels["validation"], probability)
            metrics = binary_metrics(labels["validation"], probability, threshold)
            candidate_id = f"{feature_set}__{model_name}"
            records[candidate_id] = {
                "feature_set": feature_set,
                "model": model_name,
                "estimator": clone(search.best_estimator_),
                "parameters": search.best_params_,
                "training_cv_auc": float(search.best_score_),
                "validation_probability": probability,
                "threshold": threshold,
                "metrics": metrics,
            }
            rows.append(
                percent_metrics_row(
                    feature_set,
                    model_name,
                    float(search.best_score_),
                    search.best_params_,
                    metrics,
                )
            )

    validation = pd.DataFrame(rows)
    selected: dict[str, dict[str, object]] = {}
    validation["selected_within_feature_set"] = False
    for feature_set in FEATURE_SET_LABELS:
        candidates = validation.loc[validation["feature_set"].eq(feature_set)]
        chosen = candidates.sort_values(
            [
                "roc_auc_percent",
                "balanced_accuracy_percent",
                "accuracy_percent",
                "training_cv_auc_percent",
            ],
            ascending=False,
            kind="mergesort",
        ).iloc[0]
        chosen_id = f"{feature_set}__{chosen['model']}"
        selected[feature_set] = records[chosen_id]
        validation.loc[chosen.name, "selected_within_feature_set"] = True
    return validation, selected


def fit_selected_models(
    selected: dict[str, dict[str, object]],
    development_matrices: dict[str, np.ndarray],
    development_labels: np.ndarray,
    test_matrices: dict[str, np.ndarray],
    test_labels: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Refit frozen feature-set selections and evaluate the unchanged test set."""

    fitted: dict[str, object] = {}
    metric_rows = []
    prediction_columns: dict[str, np.ndarray] = {}
    for feature_set, selection in selected.items():
        estimator = clone(selection["estimator"])
        estimator.fit(development_matrices[feature_set], development_labels)
        probability = estimator.predict_proba(test_matrices[feature_set])[:, 1]
        threshold = float(selection["threshold"])
        metrics = binary_metrics(test_labels, probability, threshold)
        row = percent_metrics_row(
            feature_set,
            str(selection["model"]),
            float(selection["training_cv_auc"]),
            dict(selection["parameters"]),
            metrics,
        )
        row["analysis_role"] = (
            "primary_fusion"
            if feature_set == "mri_age_mmse"
            else "mri_reference"
            if feature_set == "mri_only"
            else "cdr_sensitivity_not_primary"
        )
        metric_rows.append(row)
        fitted[feature_set] = estimator
        prediction_columns[f"{feature_set}_probability"] = probability
        prediction_columns[f"{feature_set}_prediction"] = (
            probability >= threshold
        ).astype(int)
    predictions = pd.DataFrame(prediction_columns)
    return fitted, pd.DataFrame(metric_rows), predictions


def paired_bootstrap_comparison(
    labels: np.ndarray,
    new_probability: np.ndarray,
    new_threshold: float,
    reference_probability: np.ndarray,
    reference_threshold: float,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare fusion with MRI-only using paired participant resampling."""

    new_point = binary_metrics(labels, new_probability, new_threshold)
    reference_point = binary_metrics(labels, reference_probability, reference_threshold)
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    samples = {metric: np.empty(iterations) for metric in METRICS}
    for iteration in range(iterations):
        selected = np.concatenate(
            [
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            ]
        )
        new_values = binary_metrics(
            labels[selected], new_probability[selected], new_threshold
        )
        reference_values = binary_metrics(
            labels[selected], reference_probability[selected], reference_threshold
        )
        for metric in METRICS:
            samples[metric][iteration] = float(new_values[metric]) - float(
                reference_values[metric]
            )
    rows = []
    for metric in METRICS:
        lower, upper = np.percentile(samples[metric], [2.5, 97.5])
        rows.append(
            {
                "metric": metric,
                "fusion_percent": 100.0 * float(new_point[metric]),
                "mri_only_percent": 100.0 * float(reference_point[metric]),
                "difference_percentage_points": 100.0
                * (float(new_point[metric]) - float(reference_point[metric])),
                "ci_95_lower_percentage_points": 100.0 * lower,
                "ci_95_upper_percentage_points": 100.0 * upper,
            }
        )

    new_correct = (new_probability >= new_threshold).astype(int) == labels
    reference_correct = (
        reference_probability >= reference_threshold
    ).astype(int) == labels
    new_only = int(np.sum(new_correct & ~reference_correct))
    reference_only = int(np.sum(~new_correct & reference_correct))
    discordant = new_only + reference_only
    p_value = (
        float(binomtest(new_only, discordant, 0.5).pvalue) if discordant else 1.0
    )
    mcnemar = pd.DataFrame(
        [
            {
                "fusion_only_correct": new_only,
                "mri_only_only_correct": reference_only,
                "discordant_participants": discordant,
                "exact_two_sided_p_value": p_value,
            }
        ]
    )
    return pd.DataFrame(rows), mcnemar


def direct_metrics(
    labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray
) -> dict[str, float | int]:
    """Calculate metrics when repeated outer folds have fold-specific thresholds."""

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "participants": int(len(labels)),
        "converters": int(labels.sum()),
        "stable_mci": int((1 - labels).sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def repeated_nested_cv(
    cohort: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run repeated nested CV for the primary MRI-age-MMSE feature set."""

    outer = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=repeats, random_state=seed
    )
    prediction_rows = []
    fold_rows = []
    for outer_index, (train_index, test_index) in enumerate(
        outer.split(features, labels)
    ):
        repeat = outer_index // 5 + 1
        fold = outer_index % 5 + 1
        print(f"Nested CV repeat {repeat}/{repeats}, fold {fold}/5", flush=True)
        inner = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=seed + outer_index
        )
        best: dict[str, object] | None = None
        for model_name, (estimator, grid) in candidate_models(
            seed + outer_index, compact=True
        ).items():
            search = GridSearchCV(
                estimator,
                grid,
                scoring="roc_auc",
                cv=inner,
                refit=True,
                n_jobs=1,
                error_score="raise",
            )
            search.fit(features[train_index], labels[train_index])
            candidate = {
                "model": model_name,
                "estimator": clone(search.best_estimator_),
                "cv_auc": float(search.best_score_),
                "parameters": search.best_params_,
            }
            if best is None or float(candidate["cv_auc"]) > float(best["cv_auc"]):
                best = candidate
        if best is None:
            raise RuntimeError("Nested CV failed to select a candidate")

        oof_probability = cross_val_predict(
            clone(best["estimator"]),
            features[train_index],
            labels[train_index],
            cv=inner,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = choose_validation_threshold(labels[train_index], oof_probability)
        estimator = clone(best["estimator"])
        estimator.fit(features[train_index], labels[train_index])
        probability = estimator.predict_proba(features[test_index])[:, 1]
        prediction = (probability >= threshold).astype(int)
        fold_metrics = binary_metrics(labels[test_index], probability, threshold)
        fold_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "selected_model": best["model"],
                "selected_hyperparameters": json.dumps(
                    best["parameters"], sort_keys=True
                ),
                "inner_cv_auc_percent": 100.0 * float(best["cv_auc"]),
                "threshold_percent": 100.0 * threshold,
                **{
                    f"{metric}_percent": 100.0 * float(fold_metrics[metric])
                    for metric in METRICS
                },
            }
        )
        for local_index, participant_index in enumerate(test_index):
            prediction_rows.append(
                {
                    "participant_id": cohort.iloc[participant_index]["participant_id"],
                    "label": int(labels[participant_index]),
                    "repeat": repeat,
                    "fold": fold,
                    "selected_model": best["model"],
                    "conversion_probability_percent": 100.0
                    * float(probability[local_index]),
                    "threshold_percent": 100.0 * threshold,
                    "predicted_class": int(prediction[local_index]),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    counts = predictions.groupby("participant_id").size()
    if not (counts == repeats).all():
        raise ValueError("Each participant must receive one outer prediction per repeat")
    participant_average = (
        predictions.groupby("participant_id", sort=False)
        .agg(
            label=("label", "first"),
            mean_conversion_probability_percent=(
                "conversion_probability_percent",
                "mean",
            ),
            mean_binary_vote=("predicted_class", "mean"),
            mean_threshold_percent=("threshold_percent", "mean"),
            outer_predictions=("predicted_class", "size"),
        )
        .reset_index()
    )
    participant_average["aggregated_predicted_class"] = (
        participant_average["mean_conversion_probability_percent"]
        >= participant_average["mean_threshold_percent"]
    ).astype(int)
    aggregate = direct_metrics(
        participant_average["label"].to_numpy(int),
        participant_average["mean_conversion_probability_percent"].to_numpy(float)
        / 100.0,
        participant_average["aggregated_predicted_class"].to_numpy(int),
    )
    summary = {
        "analysis": "repeated_nested_5_fold_internal_cross_validation",
        "outer_repeats": repeats,
        "outer_folds_per_repeat": 5,
        "inner_folds": 3,
        "participants": len(participant_average),
        "outer_predictions": len(predictions),
    }
    summary.update({f"{metric}_percent": 100.0 * float(aggregate[metric]) for metric in METRICS})
    summary.update(
        {
            "true_negative": aggregate["true_negative"],
            "false_positive": aggregate["false_positive"],
            "false_negative": aggregate["false_negative"],
            "true_positive": aggregate["true_positive"],
        }
    )
    frequencies = pd.DataFrame(
        Counter(predictions.drop_duplicates(["repeat", "fold"])["selected_model"]).items(),
        columns=["model", "outer_fold_selection_count"],
    ).sort_values("outer_fold_selection_count", ascending=False)
    return predictions, participant_average, pd.DataFrame(fold_rows), pd.DataFrame([summary]), frequencies


def save_figures(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    test_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save publication-friendly ROC, confusion, and metric comparison figures."""

    colors = {
        "mri_only": "#176b73",
        "mri_age_mmse": "#b84a2b",
        "mri_age_mmse_cdr_sensitivity": "#c8952e",
    }
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    for feature_set in FEATURE_SET_LABELS:
        fpr, tpr, _ = roc_curve(labels, probabilities[feature_set])
        auc = roc_auc_score(labels, probabilities[feature_set])
        ax.plot(
            fpr,
            tpr,
            linewidth=2.4,
            color=colors[feature_set],
            label=f"{FEATURE_SET_LABELS[feature_set]}: AUC {100 * auc:.2f}%",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Chance")
    ax.set(
        title="Post Hoc ADNI MCI-to-AD Conversion ROC Comparison",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "fusion_vs_mri_test_roc.png", dpi=300)
    plt.close(fig)

    primary_probability = probabilities["mri_age_mmse"]
    primary_prediction = (
        primary_probability >= thresholds["mri_age_mmse"]
    ).astype(int)
    matrix = confusion_matrix(labels, primary_prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(7.6, 6.5))
    image = ax.imshow(matrix, cmap="YlGn", vmin=0)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=22)
    ax.set_xticks([0, 1], labels=["Stable MCI", "Converted to AD"])
    ax.set_yticks([0, 1], labels=["Stable MCI", "Converted to AD"])
    ax.set(
        title="MRI + Age + MMSE: Post Hoc Test Confusion Matrix",
        xlabel="Predicted outcome",
        ylabel="Observed outcome",
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Participants")
    fig.tight_layout()
    fig.savefig(output_dir / "primary_fusion_test_confusion.png", dpi=300)
    plt.close(fig)

    metrics_to_plot = ["accuracy", "roc_auc", "sensitivity", "specificity", "f1_score"]
    rows = test_metrics.set_index("feature_set")
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6.8))
    for index, feature_set in enumerate(FEATURE_SET_LABELS):
        values = [float(rows.loc[feature_set, f"{metric}_percent"]) for metric in metrics_to_plot]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            color=colors[feature_set],
            label=FEATURE_SET_LABELS[feature_set],
        )
    ax.set_xticks(x, labels=["Accuracy", "ROC-AUC", "Sensitivity", "Specificity", "F1"])
    ax.set(
        title="Post Hoc Test Metrics by ADNI Feature Set",
        ylabel="Performance (%)",
        ylim=(0, 100),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_set_test_metric_comparison.png", dpi=300)
    plt.close(fig)


def write_report(
    validation: pd.DataFrame,
    test_metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    comparison: pd.DataFrame,
    mcnemar: pd.DataFrame,
    nested_summary: pd.DataFrame,
    nested_frequencies: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write a complete academically cautious fusion report."""

    selected_validation = validation.loc[validation["selected_within_feature_set"]]
    validation_lookup = selected_validation.set_index("feature_set")
    test_lookup = test_metrics.set_index("feature_set")
    fusion = test_lookup.loc["mri_age_mmse"]
    mri = test_lookup.loc["mri_only"]
    cdr = test_lookup.loc["mri_age_mmse_cdr_sensitivity"]
    interval_lookup = intervals.set_index("metric")
    difference_lookup = comparison.set_index("metric")
    nested = nested_summary.iloc[0]
    model_frequency_text = ", ".join(
        f"{row.model}: {int(row.outer_fold_selection_count)} folds"
        for row in nested_frequencies.itertuples()
    )
    report = f"""# ADNI MRI-Plus-Clinical MCI Conversion Analysis

## Question

Does adding age and MMSE to longitudinal MRI measurements improve internal
prediction of conversion from MCI to AD within 730 days after the second
pre-outcome MCI MRI?

## Participants and Inputs

- Eligible ADNI participants: 159, comprising 84 converters and 75 stable MCI
  participants.
- Fixed split: 111 training, 24 validation, and 24 test participants.
- MRI inputs: baseline eTIV-normalized volume, follow-up normalized volume,
  annualized normalized-volume change, and annualized percentage change for 20
  bilateral Alzheimer-vulnerable regions.
- Primary clinical inputs: age at the second MRI, baseline MMSE, second-visit
  MMSE, and annualized MMSE change.
- CDR and CDR Sum of Boxes were evaluated only as a sensitivity feature set
  because they can overlap with clinical diagnostic decision-making.
- Clinical measurements were complete at both MRI visits for all 159 eligible
  participants. No augmentation or synthetic participants were used.

## Development-Only Selection

- MRI-only selection: **{validation_lookup.loc['mri_only', 'model']}**;
  validation ROC-AUC {validation_lookup.loc['mri_only', 'roc_auc_percent']:.2f}%.
- Primary MRI-age-MMSE selection:
  **{validation_lookup.loc['mri_age_mmse', 'model']}**;
  validation ROC-AUC {validation_lookup.loc['mri_age_mmse', 'roc_auc_percent']:.2f}%.
- CDR sensitivity selection:
  **{validation_lookup.loc['mri_age_mmse_cdr_sensitivity', 'model']}**;
  validation ROC-AUC
  {validation_lookup.loc['mri_age_mmse_cdr_sensitivity', 'roc_auc_percent']:.2f}%.
- Thresholds were selected from validation balanced accuracy and frozen before
  the post hoc test scoring in this script.

## Unchanged Test Partition

### Primary MRI + Age + MMSE

- Accuracy: {fusion['accuracy_percent']:.2f}% (95% bootstrap interval
  {interval_lookup.loc['accuracy', 'ci_95_lower_percent']:.2f}%-
  {interval_lookup.loc['accuracy', 'ci_95_upper_percent']:.2f}%).
- Balanced accuracy: {fusion['balanced_accuracy_percent']:.2f}%.
- ROC-AUC: {fusion['roc_auc_percent']:.2f}% (95% bootstrap interval
  {interval_lookup.loc['roc_auc', 'ci_95_lower_percent']:.2f}%-
  {interval_lookup.loc['roc_auc', 'ci_95_upper_percent']:.2f}%).
- Sensitivity: {fusion['sensitivity_percent']:.2f}%.
- Specificity: {fusion['specificity_percent']:.2f}%.
- Precision: {fusion['precision_percent']:.2f}%.
- F1: {fusion['f1_score_percent']:.2f}%.
- Confusion counts: TN={int(fusion['true_negative'])},
  FP={int(fusion['false_positive'])}, FN={int(fusion['false_negative'])},
  TP={int(fusion['true_positive'])}.

### Comparators

- MRI-only: accuracy {mri['accuracy_percent']:.2f}%, ROC-AUC
  {mri['roc_auc_percent']:.2f}%.
- CDR sensitivity: accuracy {cdr['accuracy_percent']:.2f}%, ROC-AUC
  {cdr['roc_auc_percent']:.2f}%.
- Fusion versus MRI-only accuracy difference:
  {difference_lookup.loc['accuracy', 'difference_percentage_points']:.2f}
  percentage points (95% paired interval
  {difference_lookup.loc['accuracy', 'ci_95_lower_percentage_points']:.2f} to
  {difference_lookup.loc['accuracy', 'ci_95_upper_percentage_points']:.2f}).
- Fusion versus MRI-only ROC-AUC difference:
  {difference_lookup.loc['roc_auc', 'difference_percentage_points']:.2f}
  points (95% paired interval
  {difference_lookup.loc['roc_auc', 'ci_95_lower_percentage_points']:.2f} to
  {difference_lookup.loc['roc_auc', 'ci_95_upper_percentage_points']:.2f}).
- Exact McNemar p-value: {mcnemar.iloc[0]['exact_two_sided_p_value']:.4f}.

## Repeated Nested Internal Cross-Validation

- Design: {int(nested['outer_repeats'])} repeats of five outer folds with
  three-fold inner tuning, using all 159 participants for the primary
  MRI-age-MMSE feature set.
- Participant-averaged accuracy: {nested['accuracy_percent']:.2f}%.
- Participant-averaged balanced accuracy:
  {nested['balanced_accuracy_percent']:.2f}%.
- Participant-averaged ROC-AUC: {nested['roc_auc_percent']:.2f}%.
- Sensitivity: {nested['sensitivity_percent']:.2f}%.
- Specificity: {nested['specificity_percent']:.2f}%.
- F1: {nested['f1_score_percent']:.2f}%.
- Outer-fold model selections: {model_frequency_text}.

## Thesis-Safe Interpretation

The fixed test set had already been examined during earlier MRI-only analyses,
so its fusion result is post hoc rather than a fresh confirmatory test. In this
split, MRI plus age and MMSE did not improve accuracy or ROC-AUC over the
identically trained MRI-only reference, and the paired intervals included zero.
Repeated nested cross-validation reduces dependence on one small split but
remains internal validation; without a paired nested MRI-only comparator it
describes fusion performance rather than proving a fusion benefit. The CDR
sensitivity result must not be used as the primary MRI-plus-clinical claim
because CDR contributes to clinical characterization of dementia severity.
"""
    (output_dir / "THESIS_SAFE_MRI_CLINICAL_FUSION_REPORT.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    """Run development selection, post hoc testing, and repeated nested CV."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort, partitions = load_partitions(args.cohort_csv)
    clinical = load_visit_clinical_data(args.diagnosis_files)
    panel_specs, _ = panel_region_specs()
    regions = panel_specs["extended_ad_vulnerable_20"]

    enriched_partitions = {
        name: attach_clinical_features(frame, clinical)
        for name, frame in partitions.items()
    }
    partition_matrices: dict[str, dict[str, np.ndarray]] = {}
    feature_names: dict[str, list[str]] | None = None
    for name, frame in enriched_partitions.items():
        partition_matrices[name], current_names = feature_matrices(frame, regions)
        if feature_names is None:
            feature_names = current_names
    if feature_names is None:
        raise RuntimeError("Feature construction failed")
    labels = {
        name: frame["label"].to_numpy(int)
        for name, frame in enriched_partitions.items()
    }

    validation, selected = tune_feature_sets(partition_matrices, labels, args.seed)
    development = pd.concat(
        [enriched_partitions["train"], enriched_partitions["validation"]],
        ignore_index=True,
    )
    development_matrices, _ = feature_matrices(development, regions)
    development_labels = development["label"].to_numpy(int)
    fitted, test_metrics, test_prediction_values = fit_selected_models(
        selected,
        development_matrices,
        development_labels,
        partition_matrices["test"],
        labels["test"],
    )

    test_predictions = enriched_partitions["test"][
        ["participant_id", "label", "outcome"]
    ].copy()
    for column in test_prediction_values:
        if column.endswith("_probability"):
            test_predictions[f"{column}_percent"] = 100.0 * test_prediction_values[column]
        else:
            test_predictions[column] = test_prediction_values[column]
    for feature_set in FEATURE_SET_LABELS:
        test_predictions[f"{feature_set}_threshold_percent"] = 100.0 * float(
            selected[feature_set]["threshold"]
        )

    primary_probability = test_prediction_values["mri_age_mmse_probability"]
    primary_threshold = float(selected["mri_age_mmse"]["threshold"])
    intervals = bootstrap_intervals(
        labels["test"],
        primary_probability,
        primary_threshold,
        args.bootstrap,
        args.seed,
    )
    comparison, mcnemar = paired_bootstrap_comparison(
        labels["test"],
        primary_probability,
        primary_threshold,
        test_prediction_values["mri_only_probability"],
        float(selected["mri_only"]["threshold"]),
        args.bootstrap,
        args.seed + 101,
    )

    enriched_cohort = attach_clinical_features(cohort, clinical)
    all_matrices, _ = feature_matrices(enriched_cohort, regions)
    nested_predictions, nested_average, nested_folds, nested_summary, frequencies = (
        repeated_nested_cv(
            enriched_cohort,
            all_matrices["mri_age_mmse"],
            enriched_cohort["label"].to_numpy(int),
            args.outer_repeats,
            args.seed + 1000,
        )
    )

    clinical_columns = [
        "participant_id",
        "partition",
        "label",
        "baseline_age",
        "followup_age",
        "baseline_mmse",
        "followup_mmse",
        "annualized_mmse_change",
        "baseline_cdr",
        "followup_cdr",
        "annualized_cdr_change",
        "baseline_cdr_sb",
        "followup_cdr_sb",
        "annualized_cdr_sb_change",
    ]
    enriched_cohort[clinical_columns].to_csv(
        args.output_dir / "matched_clinical_feature_table.csv", index=False
    )
    validation.to_csv(
        args.output_dir / "validation_model_selection_percent.csv", index=False
    )
    test_metrics.to_csv(
        args.output_dir / "posthoc_test_metrics_percent.csv", index=False
    )
    test_predictions.to_csv(
        args.output_dir / "posthoc_test_predictions_percent.csv", index=False
    )
    intervals.to_csv(
        args.output_dir / "primary_fusion_test_confidence_intervals_percent.csv",
        index=False,
    )
    comparison.to_csv(
        args.output_dir / "primary_fusion_vs_mri_paired_differences_percent.csv",
        index=False,
    )
    mcnemar.to_csv(
        args.output_dir / "primary_fusion_vs_mri_mcnemar.csv", index=False
    )
    nested_predictions.to_csv(
        args.output_dir / "repeated_nested_outer_predictions_percent.csv", index=False
    )
    nested_average.to_csv(
        args.output_dir / "repeated_nested_participant_average_percent.csv", index=False
    )
    nested_folds.to_csv(
        args.output_dir / "repeated_nested_fold_metrics_percent.csv", index=False
    )
    nested_summary.to_csv(
        args.output_dir / "repeated_nested_summary_percent.csv", index=False
    )
    frequencies.to_csv(
        args.output_dir / "repeated_nested_model_selection_frequency.csv", index=False
    )
    joblib.dump(
        {
            "models": fitted,
            "thresholds": {
                feature_set: float(selected[feature_set]["threshold"])
                for feature_set in FEATURE_SET_LABELS
            },
            "feature_names": feature_names,
            "selected_models": {
                feature_set: str(selected[feature_set]["model"])
                for feature_set in FEATURE_SET_LABELS
            },
        },
        args.output_dir / "fitted_fusion_models.joblib",
    )
    probabilities = {
        feature_set: test_prediction_values[f"{feature_set}_probability"]
        for feature_set in FEATURE_SET_LABELS
    }
    thresholds = {
        feature_set: float(selected[feature_set]["threshold"])
        for feature_set in FEATURE_SET_LABELS
    }
    save_figures(
        labels["test"], probabilities, thresholds, test_metrics, args.output_dir
    )
    write_report(
        validation,
        test_metrics,
        intervals,
        comparison,
        mcnemar,
        nested_summary,
        frequencies,
        args.output_dir,
    )
    protocol = {
        "task": "MCI-to-AD conversion within 730 days after second MCI MRI",
        "dataset": "ADNI-derived FreeSurfer volume and diagnosis tables",
        "participants": 159,
        "split": {"training": 111, "validation": 24, "testing": 24},
        "primary_feature_set": FEATURE_SET_LABELS["mri_age_mmse"],
        "primary_clinical_features": [
            "followup age",
            "baseline MMSE",
            "followup MMSE",
            "annualized MMSE change",
        ],
        "cdr_feature_set_role": "sensitivity only, not primary",
        "model_selection": "training 4-fold CV followed by validation ROC-AUC",
        "threshold_selection": "validation balanced accuracy, constrained 0.30-0.70",
        "nested_internal_validation": {
            "outer_folds": 5,
            "outer_repeats": args.outer_repeats,
            "inner_folds": 3,
        },
        "augmentation_used": False,
        "external_progression_validation": False,
        "test_outcomes_previously_examined": True,
        "seed": args.seed,
    }
    (args.output_dir / "evaluation_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(test_metrics.to_string(index=False))
    print(nested_summary.to_string(index=False))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
