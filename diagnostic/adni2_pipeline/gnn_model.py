"""Small multimodal GCN representation learner with repeated fold-safe training."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from .config import PipelineConfig
from .graph_construction import (
    adjacency_edge_table,
    adjacency_to_pyg,
    build_node_feature_array,
    fit_node_scaler,
    structural_covariance_adjacency,
    transform_node_features,
)
from .regions import REGION_NODES


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible CPU runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class MultimodalGCN(nn.Module):
    """Two-channel regional GCN followed by mean pooling and a diagnostic head."""

    def __init__(self, hidden_dim: int = 24, layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        if layers not in {1, 2}:
            raise ValueError("This transparent implementation supports one or two GCN layers.")
        self.convolutions = nn.ModuleList([GCNConv(2, hidden_dim)])
        if layers == 2:
            self.convolutions.append(GCNConv(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, batch: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Return graph-level logits and pooled participant embeddings."""

        x = batch.x
        for convolution in self.convolutions:
            x = torch.relu(convolution(x, batch.edge_index, batch.edge_weight))
            x = self.dropout(x)
        embedding = global_mean_pool(x, batch.batch)
        return self.classifier(embedding).squeeze(-1), embedding


def _graphs(
    nodes: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> list[Data]:
    """Create PyG graph objects for selected participant rows."""

    return [
        Data(
            x=torch.tensor(nodes[index], dtype=torch.float32),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor([labels[index]], dtype=torch.float32),
            row_index=torch.tensor([index], dtype=torch.long),
        )
        for index in indices
    ]


def _loss(model: MultimodalGCN, loader: DataLoader, criterion: nn.Module) -> float:
    """Return mean validation loss without parameter updates."""

    model.eval()
    total = 0.0
    observations = 0
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            total += float(loss) * batch.num_graphs
            observations += batch.num_graphs
    return total / max(observations, 1)


def _fit_one_gnn(
    nodes: np.ndarray,
    labels: np.ndarray,
    outer_train: np.ndarray,
    adjacency: np.ndarray,
    settings: dict[str, float | int],
    seed: int,
) -> tuple[MultimodalGCN, dict[str, float | int]]:
    """Fit one GCN with an internal stratified early-stopping split."""

    set_global_seed(seed)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(settings["validation_fraction"]),
        random_state=seed,
    )
    local_train, local_validation = next(splitter.split(outer_train, labels[outer_train]))
    train_indices = outer_train[local_train]
    validation_indices = outer_train[local_validation]
    edge_index, edge_weight = adjacency_to_pyg(adjacency)
    batch_size = int(settings["batch_size"])
    train_loader = DataLoader(
        _graphs(nodes, labels, train_indices, edge_index, edge_weight),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        _graphs(nodes, labels, validation_indices, edge_index, edge_weight),
        batch_size=batch_size,
        shuffle=False,
    )
    model = MultimodalGCN(
        hidden_dim=int(settings["hidden_dim"]),
        layers=int(settings["layers"]),
        dropout=float(settings["dropout"]),
    )
    positives = max(int(labels[train_indices].sum()), 1)
    negatives = max(len(train_indices) - positives, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = np.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
        validation_loss = _loss(model, validation_loader, criterion)
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError("GNN training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_loss),
        "subtraining_participants": len(train_indices),
        "validation_participants": len(validation_indices),
    }


def _predict_probabilities(
    model: MultimodalGCN,
    nodes: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    adjacency: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Return exploratory GNN probabilities in the supplied index order."""

    edge_index, edge_weight = adjacency_to_pyg(adjacency)
    loader = DataLoader(
        _graphs(nodes, labels, indices, edge_index, edge_weight),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def _region_saliency(
    model: MultimodalGCN,
    nodes: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    adjacency: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Compute mean absolute gradient-times-input saliency by region and modality."""

    edge_index, edge_weight = adjacency_to_pyg(adjacency)
    loader = DataLoader(
        _graphs(nodes, labels, indices, edge_index, edge_weight),
        batch_size=batch_size,
        shuffle=False,
    )
    totals = np.zeros((len(REGION_NODES), 2), dtype=np.float64)
    observations = 0
    model.eval()
    for batch in loader:
        model.zero_grad(set_to_none=True)
        batch.x = batch.x.detach().requires_grad_(True)
        logits, _ = model(batch)
        logits.sum().backward()
        values = torch.abs(batch.x.grad * batch.x).detach().cpu().numpy()
        values = values.reshape(batch.num_graphs, len(REGION_NODES), 2)
        totals += values.sum(axis=0)
        observations += batch.num_graphs
    return totals / max(observations, 1)


def run_gnn_cross_validation(config: PipelineConfig, table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train fold-specific GNNs and save leakage-safe region rankings."""

    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    labels = table["target_ad"].to_numpy(dtype=np.int64)
    raw_nodes = build_node_feature_array(table)
    cv = config.values["cross_validation"]
    graph_settings = config.values["graph"]
    gnn_settings = config.values["gnn"]
    splitter = RepeatedStratifiedKFold(
        n_splits=int(cv["outer_folds"]),
        n_repeats=int(cv["outer_repeats"]),
        random_state=config.seed,
    )
    prediction_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    for split_index, (train_index, test_index) in enumerate(splitter.split(raw_nodes, labels)):
        repeat = split_index // int(cv["outer_folds"]) + 1
        fold = split_index % int(cv["outer_folds"]) + 1
        fold_seed = config.seed + split_index
        scaler = fit_node_scaler(raw_nodes[train_index])
        nodes = transform_node_features(raw_nodes, scaler)
        adjacency = structural_covariance_adjacency(
            nodes[train_index], top_k=int(graph_settings["top_k_neighbours"])
        )
        model, history = _fit_one_gnn(nodes, labels, train_index, adjacency, gnn_settings, fold_seed)
        saliency = _region_saliency(
            model, nodes, labels, train_index, adjacency, int(gnn_settings["batch_size"])
        )
        combined = saliency.sum(axis=1)
        selected_indices = set(np.argsort(combined)[-int(graph_settings["selected_regions"]):])
        for region_index, region in enumerate(REGION_NODES):
            importance_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "region_index": region_index,
                    "region": region.name,
                    "mri_importance": saliency[region_index, 0],
                    "pet_importance": saliency[region_index, 1],
                    "importance_score": combined[region_index],
                    "selected_in_training_fold": region_index in selected_indices,
                }
            )
        fold_edges = adjacency_edge_table(adjacency)
        fold_edges.insert(0, "fold", fold)
        fold_edges.insert(0, "repeat", repeat)
        edge_rows.extend(fold_edges.to_dict("records"))
        for role, indices in (("train", train_index), ("test", test_index)):
            probabilities = _predict_probabilities(
                model, nodes, labels, indices, adjacency, int(gnn_settings["batch_size"])
            )
            for offset, row_index in enumerate(indices):
                base = {
                    "repeat": repeat,
                    "fold": fold,
                    "role": role,
                    "row_index": int(row_index),
                    "subject_id": table.iloc[row_index]["subject_id"],
                    "diagnosis": table.iloc[row_index]["diagnosis"],
                    "target_ad": int(labels[row_index]),
                    "gnn_probability_uncalibrated": float(probabilities[offset]),
                }
                assignment_rows.append(base)
                if role == "test":
                    prediction_rows.append(base)
        training_rows.append({"repeat": repeat, "fold": fold, **history})

    predictions = pd.DataFrame(prediction_rows)
    fold_assignments = pd.DataFrame(assignment_rows)
    fold_importance = pd.DataFrame(importance_rows)
    consensus = (
        fold_importance.groupby(["region_index", "region"], as_index=False)
        .agg(
            importance_score=("importance_score", "mean"),
            importance_sd=("importance_score", "std"),
            mri_importance=("mri_importance", "mean"),
            pet_importance=("pet_importance", "mean"),
            selection_frequency=("selected_in_training_fold", "mean"),
        )
        .sort_values("importance_score", ascending=False)
        .reset_index(drop=True)
    )
    consensus["importance_rank"] = np.arange(1, len(consensus) + 1)
    selected = consensus.head(int(graph_settings["selected_regions"]))[
        ["importance_rank", "region_index", "region", "importance_score", "selection_frequency"]
    ]
    predictions.to_csv(output / "gnn_outer_predictions_uncalibrated.csv", index=False)
    fold_assignments.to_csv(output / "gnn_fold_assignments.csv", index=False)
    fold_importance.to_csv(output / "gnn_fold_region_importance.csv", index=False)
    consensus.to_csv(output / "gnn_region_importance.csv", index=False)
    selected.to_csv(output / "gnn_selected_regions.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(output / "gnn_fold_edges.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output / "gnn_training_history.csv", index=False)
    (output / "gnn_methodology.json").write_text(
        json.dumps(
            {
                "nodes": len(REGION_NODES),
                "node_features": ["MRI regional volume / ICV z-score", "PET cerebellar-reference ratio z-score"],
                "edge_definition": "mean absolute MRI/PET Pearson structural covariance; training fold only",
                "top_k_neighbours": int(graph_settings["top_k_neighbours"]),
                "symmetrisation": "element-wise maximum",
                "importance": "mean absolute gradient-times-input on outer-training participants",
                "direct_probabilities": "uncalibrated exploratory GNN outputs; not primary clinical scores",
                "outer_test_leakage": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "predictions": predictions,
        "fold_assignments": fold_assignments,
        "importance": consensus,
        "selected": selected,
    }


if __name__ == "__main__":
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    arguments = parser.parse_args()
    cfg = load_config(arguments.config)
    features = pd.read_csv(cfg.output_dir / "multimodal_features_long.csv")
    run_gnn_cross_validation(cfg, features)
    print(f"Saved repeated GNN outputs to {cfg.output_dir}")
