"""Predict 24-month MCI-to-AD conversion from longitudinal MRI change.

The prediction time is the participant's second MRI acquired while the
diagnosis is still mild cognitive impairment (MCI). Both scans therefore occur
before the outcome. The positive class is an observed AD diagnosis within 730
days after the second MRI; the negative class remains MCI with follow-up that
covers the same horizon.

This is an exploratory research analysis. It does not establish clinical
diagnostic or prognostic validity.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from evaluate_mri_early_ad_gnn import (
    MODEL_FACTORIES,
    binary_metrics,
    bootstrap_intervals,
    class_weights,
    ensemble_probabilities,
    forward_model,
    set_seed,
    train_gnn_ensemble,
)
from src.adni_graph_analysis.gnn.mri_progression import (
    adjacency_to_edges,
    build_region_specs,
    build_structural_covariance_adjacency,
    load_adni_volume_splits,
    load_diagnosis_splits,
    region_metadata_frame,
    validate_region_features,
)


LOGGER = logging.getLogger("longitudinal_mri_change_prediction")
PERCENT_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "sensitivity",
    "specificity",
    "precision",
    "f1_score",
    "brier_score",
)
DYNAMIC_CHANNELS = (
    "baseline_volume",
    "followup_volume",
    "annualized_volume_change",
    "annualized_percent_change",
)


@dataclass
class ChangePreprocessor:
    """Training-derived robust imputation, clipping, and scaling parameters."""

    medians: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    interval_mean: float
    interval_scale: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Predict 24-month MCI-to-AD conversion from two MRI scans."
    )
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source-zip",
        type=Path,
        default=repo_root / "data" / "private" / "longitudinal_mri_source.zip",
    )
    parser.add_argument(
        "--diagnosis-dir",
        type=Path,
        default=repo_root / "data" / "private" / "diagnosis_tables",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "processed" / "longitudinal_mri_change",
    )
    parser.add_argument("--horizon-days", type=int, default=730)
    parser.add_argument("--minimum-scan-interval-days", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument(
        "--ensemble-seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33],
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    """Write progress to the terminal and a reproducible log file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "training.log", mode="w", encoding="utf-8"),
        ],
    )


def _dated_volume_table(volume: pd.DataFrame, diagnosis: pd.DataFrame) -> pd.DataFrame:
    """Attach diagnosis dates to MRI rows by participant and session."""

    diagnosis_dates = diagnosis[["participant_id", "session_id", "examination_date"]].copy()
    diagnosis_dates["scan_date"] = pd.to_datetime(
        diagnosis_dates["examination_date"], errors="coerce"
    )
    diagnosis_dates = diagnosis_dates.drop_duplicates(["participant_id", "session_id"])
    return volume.merge(
        diagnosis_dates[["participant_id", "session_id", "scan_date"]],
        on=["participant_id", "session_id"],
        how="left",
        validate="many_to_one",
    )


def derive_two_mri_cohort(
    volume_tables: dict[str, pd.DataFrame],
    diagnosis_tables: dict[str, pd.DataFrame],
    region_specs,
    horizon_days: int,
    minimum_interval_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one leakage-safe two-MRI row per eligible MCI participant."""

    rows: list[dict[str, object]] = []
    audits: list[dict[str, int | str]] = []
    region_features = [spec.feature for spec in region_specs]

    for source_split in ("train", "val", "test"):
        volume = _dated_volume_table(
            volume_tables[source_split], diagnosis_tables[source_split]
        )
        diagnosis = diagnosis_tables[source_split].copy()
        diagnosis["diagnosis_date"] = pd.to_datetime(
            diagnosis["examination_date"], errors="coerce"
        )
        audit: dict[str, int | str] = {
            "source_split": source_split,
            "participants_seen": 0,
            "two_preoutcome_mci_mris": 0,
            "converter": 0,
            "stable_mci": 0,
            "insufficient_followup_or_ambiguous": 0,
        }

        participant_ids = sorted(
            set(volume["participant_id"]) & set(diagnosis["participant_id"])
        )
        for participant_id in participant_ids:
            audit["participants_seen"] += 1
            participant_volume = volume[
                (volume["participant_id"] == participant_id)
                & volume["scan_date"].notna()
            ].sort_values("scan_date")
            participant_diagnosis = diagnosis[
                (diagnosis["participant_id"] == participant_id)
                & diagnosis["diagnosis_date"].notna()
            ].sort_values("diagnosis_date")

            mci_scans = participant_volume[participant_volume["diagnosis"].eq("MCI")]
            if mci_scans.empty:
                continue
            baseline = mci_scans.iloc[0]
            followup_candidates = mci_scans[
                mci_scans["scan_date"]
                >= baseline["scan_date"] + pd.Timedelta(days=minimum_interval_days)
            ]
            if followup_candidates.empty:
                continue
            followup = followup_candidates.iloc[0]

            diagnoses_through_followup = participant_diagnosis[
                participant_diagnosis["diagnosis_date"] <= followup["scan_date"]
            ]
            if diagnoses_through_followup["diagnosis"].eq("AD").any():
                continue

            audit["two_preoutcome_mci_mris"] += 1
            prediction_date = followup["scan_date"]
            horizon_date = prediction_date + pd.Timedelta(days=horizon_days)
            future = participant_diagnosis[
                participant_diagnosis["diagnosis_date"] > prediction_date
            ]
            ad_dates = future.loc[future["diagnosis"].eq("AD"), "diagnosis_date"]

            if not ad_dates.empty and ad_dates.min() <= horizon_date:
                label = 1
                outcome = "MCI_to_AD_within_24_months_after_second_MRI"
                first_ad_date = ad_dates.min()
                days_to_ad = int((first_ad_date - prediction_date).days)
                audit["converter"] += 1
            else:
                through_horizon = future[future["diagnosis_date"] <= horizon_date]
                followup_covers_horizon = (
                    not future.empty
                    and future["diagnosis_date"].max() >= horizon_date
                )
                only_mci = (
                    not through_horizon.empty
                    and through_horizon["diagnosis"].eq("MCI").all()
                )
                if not (followup_covers_horizon and only_mci):
                    audit["insufficient_followup_or_ambiguous"] += 1
                    continue
                label = 0
                outcome = "MCI_stable_at_least_24_months_after_second_MRI"
                days_to_ad = np.nan
                audit["stable_mci"] += 1

            interval_days = int((followup["scan_date"] - baseline["scan_date"]).days)
            available_followup_days = int(
                (future["diagnosis_date"].max() - prediction_date).days
            )
            row: dict[str, object] = {
                "participant_id": str(participant_id),
                "source_split": source_split,
                "baseline_session_id": str(baseline["session_id"]),
                "baseline_date": baseline["scan_date"].date().isoformat(),
                "followup_session_id": str(followup["session_id"]),
                "prediction_date": prediction_date.date().isoformat(),
                "scan_interval_days": interval_days,
                "label": label,
                "outcome": outcome,
                "days_to_ad_after_second_mri": days_to_ad,
                "available_followup_days": available_followup_days,
                "baseline_eTIV": pd.to_numeric(baseline["eTIV"], errors="coerce"),
                "followup_eTIV": pd.to_numeric(followup["eTIV"], errors="coerce"),
            }
            for feature in region_features:
                row[f"baseline__{feature}"] = pd.to_numeric(
                    baseline[feature], errors="coerce"
                )
                row[f"followup__{feature}"] = pd.to_numeric(
                    followup[feature], errors="coerce"
                )
            rows.append(row)
        audits.append(audit)

    cohort = pd.DataFrame(rows)
    if cohort.empty:
        raise ValueError("No eligible two-MRI MCI progression participants were found.")
    duplicates = cohort["participant_id"].duplicated(keep=False)
    if duplicates.any():
        duplicate_ids = cohort.loc[duplicates, "participant_id"].unique().tolist()
        raise ValueError(f"Participants occurred in multiple source splits: {duplicate_ids}")
    return cohort.sort_values("participant_id").reset_index(drop=True), pd.DataFrame(audits)


def stratified_participant_split(
    cohort: pd.DataFrame, random_state: int
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Create a reproducible 70%/15%/15% participant split."""

    train, temporary = train_test_split(
        cohort,
        test_size=0.30,
        stratify=cohort["label"],
        random_state=random_state,
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        stratify=temporary["label"],
        random_state=random_state,
    )
    partitions = {
        "train": train.sort_values("participant_id").reset_index(drop=True),
        "validation": validation.sort_values("participant_id").reset_index(drop=True),
        "test": test.sort_values("participant_id").reset_index(drop=True),
    }
    total = len(cohort)
    summary = pd.DataFrame(
        [
            {
                "partition": name,
                "participants": len(frame),
                "percentage_of_eligible": 100.0 * len(frame) / total,
                "converters": int(frame["label"].sum()),
                "stable_mci": int((1 - frame["label"]).sum()),
            }
            for name, frame in partitions.items()
        ]
    )
    return partitions, summary


def raw_dynamic_matrices(cohort: pd.DataFrame, region_specs) -> np.ndarray:
    """Return baseline, follow-up, annualized change, and percent-change arrays."""

    features = [spec.feature for spec in region_specs]
    baseline = cohort[[f"baseline__{feature}" for feature in features]].to_numpy(float)
    followup = cohort[[f"followup__{feature}" for feature in features]].to_numpy(float)
    baseline_etiv = cohort["baseline_eTIV"].to_numpy(float, copy=True).reshape(-1, 1)
    followup_etiv = cohort["followup_eTIV"].to_numpy(float, copy=True).reshape(-1, 1)
    baseline_etiv[baseline_etiv <= 0] = np.nan
    followup_etiv[followup_etiv <= 0] = np.nan
    baseline = baseline / baseline_etiv
    followup = followup / followup_etiv
    years = cohort["scan_interval_days"].to_numpy(float).reshape(-1, 1) / 365.25
    annualized_change = (followup - baseline) / years
    denominator = np.where(np.abs(baseline) > 1e-12, np.abs(baseline), np.nan)
    annualized_percent = 100.0 * (followup - baseline) / denominator / years
    return np.stack([baseline, followup, annualized_change, annualized_percent], axis=1)


def fit_change_preprocessor(raw_train: np.ndarray, interval_days: np.ndarray) -> ChangePreprocessor:
    """Fit robust preprocessing using training participants only."""

    finite = np.where(np.isfinite(raw_train), raw_train, np.nan)
    medians = np.nanmedian(finite, axis=0)
    imputed = np.where(np.isnan(finite), medians, finite)
    lower = np.quantile(imputed, 0.01, axis=0)
    upper = np.quantile(imputed, 0.99, axis=0)
    clipped = np.clip(imputed, lower, upper)
    means = clipped.mean(axis=0)
    scales = clipped.std(axis=0)
    scales[scales == 0] = 1.0
    interval_mean = float(np.mean(interval_days))
    interval_scale = float(np.std(interval_days)) or 1.0
    return ChangePreprocessor(
        medians=medians,
        lower_bounds=lower,
        upper_bounds=upper,
        means=means,
        scales=scales,
        interval_mean=interval_mean,
        interval_scale=interval_scale,
    )


def transform_dynamic_matrices(raw: np.ndarray, preprocessor: ChangePreprocessor) -> np.ndarray:
    """Apply frozen training-derived preprocessing to four dynamic channels."""

    finite = np.where(np.isfinite(raw), raw, np.nan)
    imputed = np.where(np.isnan(finite), preprocessor.medians, finite)
    clipped = np.clip(imputed, preprocessor.lower_bounds, preprocessor.upper_bounds)
    return (clipped - preprocessor.means) / preprocessor.scales


def build_change_graphs(
    cohort: pd.DataFrame,
    transformed: np.ndarray,
    adjacency: np.ndarray,
    region_specs,
    preprocessor: ChangePreprocessor,
) -> list[Data]:
    """Build one longitudinal regional graph per participant."""

    edge_index, edge_weight = adjacency_to_edges(adjacency)
    hemisphere = np.array(
        [-1.0 if spec.hemisphere == "left" else 1.0 for spec in region_specs]
    )
    cortical = np.array(
        [spec.region_type == "cortical" for spec in region_specs], dtype=float
    )
    subcortical = 1.0 - cortical
    graphs: list[Data] = []
    for position, row in cohort.reset_index(drop=True).iterrows():
        interval_z = (
            float(row["scan_interval_days"]) - preprocessor.interval_mean
        ) / preprocessor.interval_scale
        node_features = np.column_stack(
            [
                transformed[position, 0],
                transformed[position, 1],
                transformed[position, 2],
                transformed[position, 3],
                np.full(len(region_specs), interval_z),
                hemisphere,
                cortical,
                subcortical,
            ]
        ).astype(np.float32)
        graph = Data(
            x=torch.tensor(node_features),
            edge_index=edge_index.clone(),
            edge_weight=edge_weight.clone(),
            y=torch.tensor([int(row["label"])], dtype=torch.long),
        )
        graph.participant_id = str(row["participant_id"])
        graph.baseline_session_id = str(row["baseline_session_id"])
        graph.followup_session_id = str(row["followup_session_id"])
        graphs.append(graph)
    return graphs


def train_fixed_epoch_ensemble(
    model_name: str,
    graphs,
    epochs: int,
    args: argparse.Namespace,
) -> list[nn.Module]:
    """Refit a GNN ensemble on development participants for fixed epochs."""

    models: list[nn.Module] = []
    for seed in args.ensemble_seeds:
        LOGGER.info("Refitting %s seed %s for %s epochs", model_name, seed, epochs)
        set_seed(seed)
        hidden = args.hidden_channels if model_name == "GCN" else max(8, args.hidden_channels // 2)
        model = MODEL_FACTORIES[model_name](
            in_channels=graphs[0].num_node_features,
            hidden_channels=hidden,
            out_channels=2,
            dropout=args.dropout,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights(graphs))
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            graphs,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        )
        for _ in range(epochs):
            model.train()
            for batch in loader:
                optimizer.zero_grad()
                loss = criterion(forward_model(model, batch), batch.y.view(-1))
                loss.backward()
                optimizer.step()
        models.append(model)
    return models


def conventional_candidates(random_state: int):
    """Return prespecified MRI comparators and compact training-only grids."""

    return {
        "Baseline logistic": (
            "baseline",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            solver="liblinear",
                            max_iter=5000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Longitudinal elastic-net": (
            "longitudinal",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            solver="saga",
                            max_iter=10000,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            {
                "classifier__C": [0.01, 0.1, 1.0],
                "classifier__l1_ratio": [0.25, 0.5, 0.75],
            },
        ),
        "Longitudinal RBF SVM": (
            "longitudinal",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            class_weight="balanced",
                            probability=True,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__gamma": ["scale", 0.01, 0.1],
            },
        ),
        "Longitudinal shrinkage LDA": (
            "longitudinal",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LinearDiscriminantAnalysis(solver="lsqr", priors=[0.5, 0.5]),
                    ),
                ]
            ),
            {"classifier__shrinkage": ["auto", 0.1, 0.5, 0.9]},
        ),
        "Longitudinal Extra Trees": (
            "longitudinal",
            ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            {
                "max_depth": [3, 5, None],
                "min_samples_leaf": [2, 5],
                "max_features": ["sqrt", 0.5],
            },
        ),
    }


def flattened_features(transformed: np.ndarray, view: str) -> np.ndarray:
    """Flatten baseline-only or all longitudinal regional channels."""

    if view == "baseline":
        return transformed[:, 0, :]
    return transformed.reshape(len(transformed), -1)


def percent_metric_row(model: str, metrics: dict[str, float | int]) -> dict[str, object]:
    """Convert classification metrics to percentage-valued output columns."""

    row: dict[str, object] = {
        "model": model,
        "participants": metrics["n_participants"],
        "converters": metrics["n_converters"],
        "stable_mci": metrics["n_stable_mci"],
        "true_negative": metrics["true_negative"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "true_positive": metrics["true_positive"],
    }
    for metric in PERCENT_METRICS:
        row[f"{metric}_percent"] = 100.0 * float(metrics[metric])
    return row


def percentage_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    """Convert bootstrap estimates and bounds to percentages."""

    result = intervals.copy()
    result[["estimate", "ci_95_lower", "ci_95_upper"]] *= 100.0
    result["unit"] = "percent"
    return result


def compute_multichannel_saliency(models, graphs, metadata: pd.DataFrame) -> pd.DataFrame:
    """Compute gradient-times-input saliency for each dynamic MRI channel."""

    saliency = np.zeros((len(metadata), len(DYNAMIC_CHANNELS)), dtype=float)
    observations = 0
    for model in models:
        model.eval()
        for graph in graphs:
            x = graph.x.detach().clone().requires_grad_(True)
            batch = torch.zeros(graph.num_nodes, dtype=torch.long)
            logits = model(x, graph.edge_index, batch, edge_weight=graph.edge_weight)
            model.zero_grad(set_to_none=True)
            logits[0, 1].backward()
            contribution = np.abs(
                x.grad[:, : len(DYNAMIC_CHANNELS)].detach().numpy()
                * x[:, : len(DYNAMIC_CHANNELS)].detach().numpy()
            )
            saliency += contribution
            observations += 1
    result = metadata.copy()
    channel_saliency = saliency / max(observations, 1)
    for index, channel in enumerate(DYNAMIC_CHANNELS):
        result[f"{channel}_saliency"] = channel_saliency[:, index]
    result["total_dynamic_saliency"] = channel_saliency.sum(axis=1)
    result["saliency_rank"] = (
        result["total_dynamic_saliency"].rank(method="first", ascending=False).astype(int)
    )
    return result.sort_values("saliency_rank").reset_index(drop=True)


def save_figures(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    selected_model: str,
    selected_gnn: str,
    saliency: pd.DataFrame,
    raw_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Save publication-friendly ROC, confusion, saliency, and change figures."""

    shown_models = [selected_model]
    if selected_gnn not in shown_models:
        shown_models.append(selected_gnn)
    if "Baseline logistic" not in shown_models:
        shown_models.append("Baseline logistic")
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    colors = ["#a63d40", "#1f5f7a", "#5a6b4d"]
    for model, color in zip(shown_models, colors):
        fpr, tpr, _ = roc_curve(labels, probabilities[model])
        auc = binary_metrics(labels, probabilities[model])["roc_auc"] * 100.0
        ax.plot(100 * fpr, 100 * tpr, lw=2.5, color=color, label=f"{model}: AUC {auc:.2f}%")
    ax.plot([0, 100], [0, 100], "--", color="#777777", lw=1.5)
    ax.set_xlabel("False-positive rate (%)")
    ax.set_ylabel("True-positive rate (%)")
    ax.set_title("Two-MRI prediction of MCI-to-AD conversion")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "held_out_roc_percent.png", dpi=300)
    plt.close(fig)

    predictions = (probabilities[selected_model] >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).astype(float)
    row_percent = 100.0 * matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    image = ax.imshow(row_percent, cmap="Blues", vmin=0, vmax=100)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{row_percent[row, column]:.1f}%", ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], labels=["Stable MCI", "Converted to AD"])
    ax.set_yticks([0, 1], labels=["Stable MCI", "Converted to AD"])
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Observed outcome")
    ax.set_title(f"Held-out normalized confusion matrix\n{selected_model}", fontsize=13)
    fig.colorbar(image, ax=ax, label="Row percentage")
    fig.tight_layout()
    fig.savefig(output_dir / "held_out_selected_confusion_percent.png", dpi=300)
    plt.close(fig)

    top = saliency.nsmallest(15, "saliency_rank").sort_values("total_dynamic_saliency")
    fig, ax = plt.subplots(figsize=(9.0, 7.2))
    ax.barh(top["region_name"], top["total_dynamic_saliency"], color="#1f5f7a")
    ax.set_xlabel("Mean absolute gradient x dynamic MRI input")
    ax.set_ylabel("FreeSurfer anatomical region")
    ax.set_title(f"Top longitudinal regional saliency: {selected_gnn}")
    fig.tight_layout()
    fig.savefig(output_dir / "top_15_longitudinal_region_saliency.png", dpi=300)
    plt.close(fig)

    percent_change = raw_test[:, 3, :]
    group_difference = percent_change[labels == 1].mean(axis=0) - percent_change[labels == 0].mean(axis=0)
    metadata = saliency.sort_values("node_index")
    order = np.argsort(np.abs(group_difference))[-10:]
    colors = np.where(group_difference[order] < 0, "#a63d40", "#5a6b4d")
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ax.barh(metadata.iloc[order]["region_name"], group_difference[order], color=colors)
    ax.axvline(0, color="#444444", lw=1)
    ax.set_xlabel("Converter minus stable-MCI annualized volume change (% points/year)")
    ax.set_ylabel("FreeSurfer anatomical region")
    ax.set_title("Largest held-out regional change differences (descriptive)")
    fig.tight_layout()
    fig.savefig(output_dir / "top_10_held_out_change_differences.png", dpi=300)
    plt.close(fig)


def write_report(
    output_dir: Path,
    split_summary: pd.DataFrame,
    edge_count: int,
    selected_gnn: str,
    selected_model: str,
    validation_metrics: dict[str, dict[str, float | int]],
    test_metrics: dict[str, dict[str, float | int]],
    intervals: pd.DataFrame,
    saliency: pd.DataFrame,
    median_scan_interval_days: float,
) -> None:
    """Write a cautious plain-English report with percentage results."""

    selected = test_metrics[selected_model]
    ci = intervals.set_index("metric")
    split_lines = [
        f"- {row.partition}: {int(row.participants)} participants "
        f"({row.percentage_of_eligible:.2f}%; {int(row.converters)} converters, "
        f"{int(row.stable_mci)} stable MCI)."
        for row in split_summary.itertuples()
    ]
    top_regions = ", ".join(saliency.nsmallest(8, "saliency_rank")["region_name"])
    lines = [
        "# Exploratory Longitudinal MRI Change Prediction",
        "",
        "## Target",
        "",
        "The model estimates whether a participant with two MRI scans acquired while",
        "still diagnosed with MCI is observed to convert to AD within 730 days after",
        "the second MRI. It does not diagnose AD at the second scan.",
        "",
        "## Cohort",
        "",
        *split_lines,
        f"- Median interval between the two MRI scans: {median_scan_interval_days:.0f} days.",
        "",
        "## Graph",
        "",
        "- Nodes: 82 FreeSurfer regions (68 Desikan cortical and 14 subcortical).",
        "- Dynamic node inputs: baseline volume, follow-up volume, annualized volume",
        "  change, and annualized percentage volume change.",
        f"- Edges: {edge_count} weighted undirected change-covariance relationships.",
        "- Edge weights and all preprocessing parameters were learned from development",
        "  participants only; each node retained eight strongest neighbours.",
        "",
        "## Model Selection",
        "",
        f"Validation selected {selected_gnn} among the GNN architectures and",
        f"{selected_model} across all prespecified MRI models.",
        f"Selected-model validation ROC-AUC: {100*validation_metrics[selected_model]['roc_auc']:.2f}%.",
        "No synthetic augmentation was used. Class weighting and dropout were used",
        "without inventing regional trajectories.",
        "",
        "## Held-Out Test",
        "",
        f"- Accuracy: {100*selected['accuracy']:.2f}% "
        f"(95% CI {100*ci.loc['accuracy','ci_95_lower']:.2f}%-{100*ci.loc['accuracy','ci_95_upper']:.2f}%).",
        f"- ROC-AUC: {100*selected['roc_auc']:.2f}% "
        f"(95% CI {100*ci.loc['roc_auc','ci_95_lower']:.2f}%-{100*ci.loc['roc_auc','ci_95_upper']:.2f}%).",
        f"- Sensitivity: {100*selected['sensitivity']:.2f}%.",
        f"- Specificity: {100*selected['specificity']:.2f}%.",
        f"- Precision: {100*selected['precision']:.2f}%.",
        f"- F1 score: {100*selected['f1_score']:.2f}%.",
        "",
        f"## Held-Out {selected_gnn}",
        "",
        f"- Accuracy: {100*test_metrics[selected_gnn]['accuracy']:.2f}%.",
        f"- ROC-AUC: {100*test_metrics[selected_gnn]['roc_auc']:.2f}%.",
        f"- Sensitivity: {100*test_metrics[selected_gnn]['sensitivity']:.2f}%.",
        f"- Specificity: {100*test_metrics[selected_gnn]['specificity']:.2f}%.",
        f"- F1 score: {100*test_metrics[selected_gnn]['f1_score']:.2f}%.",
        "",
        "## GNN Interpretability",
        "",
        f"The highest fitted longitudinal GNN saliency occurred in: {top_regions}.",
        "Saliency is model sensitivity, not causal or biomarker proof.",
        "",
        "## Required Limitation",
        "",
        "The final test cohort is small and no independent MCI-conversion cohort is",
        "available. The source data and outcomes were also examined in earlier work.",
        "These are exploratory feasibility results, not untouched confirmatory evidence",
        "and not a clinically validated prediction system.",
    ]
    (output_dir / "THESIS_SAFE_LONGITUDINAL_CHANGE_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    """Run cohort derivation, model selection, refit, evaluation, and reporting."""

    args = parse_args()
    configure_logging(args.output_dir)
    torch.set_num_threads(args.torch_threads)
    args.balance_strategy = "weighted_loss"

    LOGGER.info("Loading MRI volume and longitudinal diagnosis tables")
    region_specs = build_region_specs()
    volume_tables = load_adni_volume_splits(args.source_zip)
    diagnosis_tables = load_diagnosis_splits(args.diagnosis_dir)
    validate_region_features(volume_tables, region_specs)
    cohort, cohort_audit = derive_two_mri_cohort(
        volume_tables,
        diagnosis_tables,
        region_specs,
        args.horizon_days,
        args.minimum_scan_interval_days,
    )
    partitions, split_summary = stratified_participant_split(cohort, args.random_state)
    LOGGER.info(
        "Eligible=%s | train=%s validation=%s test=%s",
        len(cohort),
        len(partitions["train"]),
        len(partitions["validation"]),
        len(partitions["test"]),
    )

    raw_selection = {
        name: raw_dynamic_matrices(frame, region_specs)
        for name, frame in partitions.items()
    }
    selection_preprocessor = fit_change_preprocessor(
        raw_selection["train"],
        partitions["train"]["scan_interval_days"].to_numpy(float),
    )
    transformed_selection = {
        name: transform_dynamic_matrices(raw, selection_preprocessor)
        for name, raw in raw_selection.items()
    }
    selection_adjacency = build_structural_covariance_adjacency(
        transformed_selection["train"][:, 3, :], top_k=args.top_k
    )
    selection_graphs = {
        name: build_change_graphs(
            partitions[name],
            transformed_selection[name],
            selection_adjacency,
            region_specs,
            selection_preprocessor,
        )
        for name in ("train", "validation")
    }

    validation_probabilities: dict[str, np.ndarray] = {}
    validation_metrics: dict[str, dict[str, float | int]] = {}
    gnn_training_details = []
    gnn_ensembles = {}
    for model_name in ("GCN", "GAT"):
        ensemble, details, probabilities = train_gnn_ensemble(
            model_name,
            selection_graphs["train"],
            selection_graphs["validation"],
            args,
        )
        gnn_ensembles[model_name] = ensemble
        details.insert(0, "selection_stage", "architecture_selection")
        gnn_training_details.append(details)
        validation_probabilities[model_name] = probabilities
        validation_metrics[model_name] = binary_metrics(
            partitions["validation"]["label"].to_numpy(int), probabilities
        )
    selected_gnn = max(
        ("GCN", "GAT"),
        key=lambda name: (
            validation_metrics[name]["roc_auc"],
            validation_metrics[name]["balanced_accuracy"],
        ),
    )

    candidate_estimators = {}
    candidate_views = {}
    candidate_params = {}
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.random_state)
    train_labels = partitions["train"]["label"].to_numpy(int)
    validation_labels = partitions["validation"]["label"].to_numpy(int)
    for model_name, (view, estimator, parameter_grid) in conventional_candidates(
        args.random_state
    ).items():
        LOGGER.info("Training conventional comparator: %s", model_name)
        train_features = flattened_features(transformed_selection["train"], view)
        validation_features = flattened_features(
            transformed_selection["validation"], view
        )
        search = GridSearchCV(
            estimator,
            parameter_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(train_features, train_labels)
        probabilities = search.predict_proba(validation_features)[:, 1]
        candidate_estimators[model_name] = clone(search.best_estimator_)
        candidate_views[model_name] = view
        candidate_params[model_name] = search.best_params_
        validation_probabilities[model_name] = probabilities
        validation_metrics[model_name] = binary_metrics(validation_labels, probabilities)

    selected_model = max(
        validation_metrics,
        key=lambda name: (
            validation_metrics[name]["roc_auc"],
            validation_metrics[name]["balanced_accuracy"],
        ),
    )
    LOGGER.info("Validation selected GNN=%s | overall model=%s", selected_gnn, selected_model)

    development = pd.concat(
        [partitions["train"], partitions["validation"]], ignore_index=True
    ).sort_values("participant_id").reset_index(drop=True)
    raw_final = {
        "development": raw_dynamic_matrices(development, region_specs),
        "test": raw_dynamic_matrices(partitions["test"], region_specs),
    }
    final_preprocessor = fit_change_preprocessor(
        raw_final["development"],
        development["scan_interval_days"].to_numpy(float),
    )
    transformed_final = {
        name: transform_dynamic_matrices(raw, final_preprocessor)
        for name, raw in raw_final.items()
    }
    final_adjacency = build_structural_covariance_adjacency(
        transformed_final["development"][:, 3, :], top_k=args.top_k
    )
    final_graphs = {
        "development": build_change_graphs(
            development,
            transformed_final["development"],
            final_adjacency,
            region_specs,
            final_preprocessor,
        ),
        "test": build_change_graphs(
            partitions["test"],
            transformed_final["test"],
            final_adjacency,
            region_specs,
            final_preprocessor,
        ),
    }
    selected_details = pd.concat(gnn_training_details, ignore_index=True)
    selected_epochs = selected_details.loc[
        selected_details["model"].eq(selected_gnn), "best_epoch"
    ]
    refit_epochs = max(1, int(np.median(selected_epochs)))
    final_gnn_ensemble = train_fixed_epoch_ensemble(
        selected_gnn, final_graphs["development"], refit_epochs, args
    )

    test_probabilities: dict[str, np.ndarray] = {
        selected_gnn: ensemble_probabilities(
            final_gnn_ensemble, final_graphs["test"], args.batch_size
        )
    }
    development_labels = development["label"].to_numpy(int)
    for model_name, estimator in candidate_estimators.items():
        view = candidate_views[model_name]
        estimator.fit(
            flattened_features(transformed_final["development"], view),
            development_labels,
        )
        test_probabilities[model_name] = estimator.predict_proba(
            flattened_features(transformed_final["test"], view)
        )[:, 1]
        candidate_estimators[model_name] = estimator

    # The unselected GNN is not refitted or tested; model choice was locked on validation.
    test_labels = partitions["test"]["label"].to_numpy(int)
    test_metrics = {
        name: binary_metrics(test_labels, probabilities)
        for name, probabilities in test_probabilities.items()
    }
    selected_test_probabilities = test_probabilities[selected_model]
    selected_intervals = bootstrap_intervals(
        test_labels,
        selected_test_probabilities,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
    )
    saliency = compute_multichannel_saliency(
        final_gnn_ensemble,
        final_graphs["test"],
        region_metadata_frame(region_specs),
    )

    cohort_output = cohort.copy()
    partition_lookup = {
        participant_id: partition
        for partition, frame in partitions.items()
        for participant_id in frame["participant_id"]
    }
    cohort_output.insert(
        2,
        "partition",
        cohort_output["participant_id"].map(partition_lookup),
    )
    cohort_output.to_csv(args.output_dir / "eligible_two_mri_cohort.csv", index=False)
    cohort_audit.to_csv(args.output_dir / "cohort_audit.csv", index=False)
    split_summary.to_csv(args.output_dir / "participant_split_summary_percent.csv", index=False)
    region_metadata_frame(region_specs).to_csv(
        args.output_dir / "graph_nodes_82_regions.csv", index=False
    )
    pd.DataFrame(
        final_adjacency,
        index=[spec.region_name for spec in region_specs],
        columns=[spec.region_name for spec in region_specs],
    ).to_csv(args.output_dir / "development_change_covariance_adjacency.csv")
    selected_details.to_csv(args.output_dir / "gnn_architecture_training_details.csv", index=False)
    pd.DataFrame(
        [percent_metric_row(name, metrics) for name, metrics in validation_metrics.items()]
    ).to_csv(args.output_dir / "validation_model_selection_percent.csv", index=False)
    pd.DataFrame(
        [
            {
                **percent_metric_row(name, metrics),
                "selected_from_validation": name == selected_model,
            }
            for name, metrics in test_metrics.items()
        ]
    ).to_csv(args.output_dir / "held_out_test_metrics_percent.csv", index=False)
    percentage_intervals(selected_intervals).to_csv(
        args.output_dir / "selected_model_confidence_intervals_percent.csv", index=False
    )
    saliency.to_csv(args.output_dir / "longitudinal_gnn_region_saliency.csv", index=False)

    prediction_output = partitions["test"][
        [
            "participant_id",
            "baseline_session_id",
            "followup_session_id",
            "prediction_date",
            "label",
            "outcome",
        ]
    ].copy()
    for model_name, probabilities in test_probabilities.items():
        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        prediction_output[f"{safe_name}_ad_conversion_probability_percent"] = 100.0 * probabilities
    prediction_output["selected_model"] = selected_model
    prediction_output["selected_ad_conversion_probability_percent"] = 100.0 * selected_test_probabilities
    prediction_output["selected_predicted_class"] = (
        selected_test_probabilities >= 0.5
    ).astype(int)
    prediction_output.to_csv(
        args.output_dir / "held_out_participant_predictions_percent.csv", index=False
    )

    group_change_rows = []
    raw_percent = raw_final["test"][:, 3, :]
    for node_index, spec in enumerate(region_specs):
        stable_mean = float(raw_percent[test_labels == 0, node_index].mean())
        converter_mean = float(raw_percent[test_labels == 1, node_index].mean())
        group_change_rows.append(
            {
                "node_index": node_index,
                "region_name": spec.region_name,
                "stable_mci_annualized_change_percent": stable_mean,
                "converter_annualized_change_percent": converter_mean,
                "converter_minus_stable_percentage_points": converter_mean - stable_mean,
                "interpretation": "descriptive held-out difference; not causal or multiplicity-corrected",
            }
        )
    pd.DataFrame(group_change_rows).to_csv(
        args.output_dir / "held_out_regional_change_descriptive_percent.csv", index=False
    )

    save_figures(
        test_labels,
        test_probabilities,
        selected_model,
        selected_gnn,
        saliency,
        raw_final["test"],
        args.output_dir,
    )
    edge_count = int(np.count_nonzero(np.triu(final_adjacency, k=1)))
    write_report(
        args.output_dir,
        split_summary,
        edge_count,
        selected_gnn,
        selected_model,
        validation_metrics,
        test_metrics,
        selected_intervals,
        saliency,
        float(cohort["scan_interval_days"].median()),
    )

    protocol = {
        "task": "two-MRI MCI-to-AD conversion within 730 days after second MCI MRI",
        "eligible_participants": len(cohort),
        "split_percent_target": {"training": 70, "validation": 15, "testing": 15},
        "split_unit": "participant",
        "random_state": args.random_state,
        "nodes": len(region_specs),
        "node_definition": "68 bilateral Desikan cortical plus 14 bilateral subcortical FreeSurfer regions",
        "dynamic_node_features": list(DYNAMIC_CHANNELS),
        "edges": edge_count,
        "edge_definition": "top-8 absolute Pearson covariance of annualized percentage volume change from development participants",
        "selected_gnn": selected_gnn,
        "selected_overall_model": selected_model,
        "selection_rule": "validation ROC-AUC percentage, then balanced accuracy percentage",
        "refit_epochs": refit_epochs,
        "ensemble_seeds": args.ensemble_seeds,
        "probability_threshold_percent": 50,
        "augmentation_used": False,
        "external_progression_validation": False,
        "reporting_scope": "exploratory feasibility, not clinical validation",
    }
    (args.output_dir / "evaluation_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "model_name": selected_gnn,
            "state_dicts": [model.state_dict() for model in final_gnn_ensemble],
            "adjacency": final_adjacency,
            "preprocessor": asdict(final_preprocessor),
            "region_metadata": region_metadata_frame(region_specs).to_dict(orient="records"),
            "node_feature_order": list(DYNAMIC_CHANNELS)
            + ["scan_interval", "hemisphere", "cortical", "subcortical"],
            "refit_epochs": refit_epochs,
        },
        args.output_dir / "selected_longitudinal_gnn_ensemble.pt",
    )
    joblib.dump(
        {
            "selected_model": selected_model,
            "fitted_conventional_models": candidate_estimators,
            "candidate_views": candidate_views,
            "training_selected_parameters": candidate_params,
            "change_preprocessor": asdict(final_preprocessor),
        },
        args.output_dir / "fitted_conventional_models.joblib",
    )
    LOGGER.info("Saved longitudinal MRI change outputs to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
