"""Fold-safe multimodal node tensors and structural covariance graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .regions import REGION_NODES


@dataclass(frozen=True)
class NodeScaler:
    """Training-fold mean and standard deviation for each node and modality."""

    mean: np.ndarray
    scale: np.ndarray


def build_node_feature_array(table: pd.DataFrame) -> np.ndarray:
    """Return participant x region x [MRI, PET] features in fixed node order."""

    nodes = [
        table[[region.mri_feature, region.pet_feature]].to_numpy(dtype=np.float32)
        for region in REGION_NODES
    ]
    values = np.stack(nodes, axis=1)
    if not np.isfinite(values).all():
        raise ValueError("Node features contain missing or infinite values.")
    return values


def fit_node_scaler(training_nodes: np.ndarray) -> NodeScaler:
    """Fit z-score parameters using training participants only."""

    mean = training_nodes.mean(axis=0)
    scale = training_nodes.std(axis=0, ddof=0)
    scale[scale < 1e-8] = 1.0
    return NodeScaler(mean=mean, scale=scale)


def transform_node_features(nodes: np.ndarray, scaler: NodeScaler) -> np.ndarray:
    """Apply training-derived node-wise z-score parameters."""

    return ((nodes - scaler.mean) / scaler.scale).astype(np.float32)


def structural_covariance_adjacency(training_nodes: np.ndarray, top_k: int = 6) -> np.ndarray:
    """Build a sparse fused covariance graph from training participants only.

    MRI and PET correlations are calculated separately across participants.
    Their absolute values are averaged, then each node retains its strongest
    ``top_k`` neighbours. Taking the element-wise maximum with the transpose
    produces an undirected graph.
    """

    if training_nodes.ndim != 3 or training_nodes.shape[2] != 2:
        raise ValueError("training_nodes must have shape [participants, regions, 2]")
    n_regions = training_nodes.shape[1]
    if not 1 <= top_k < n_regions:
        raise ValueError(f"top_k must be between 1 and {n_regions - 1}")
    modality_correlations = []
    for modality in range(2):
        corr = np.corrcoef(training_nodes[:, :, modality], rowvar=False)
        modality_correlations.append(np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0))
    fused = np.mean(modality_correlations, axis=0)
    np.fill_diagonal(fused, 0.0)
    sparse = np.zeros_like(fused)
    for node in range(n_regions):
        neighbours = np.argsort(fused[node])[-top_k:]
        sparse[node, neighbours] = fused[node, neighbours]
    adjacency = np.maximum(sparse, sparse.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency.astype(np.float32)


def adjacency_to_pyg(adjacency: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a dense weighted adjacency to PyG edge index and edge weight."""

    source, target = np.nonzero(adjacency)
    edge_index = torch.tensor(np.vstack([source, target]), dtype=torch.long)
    edge_weight = torch.tensor(adjacency[source, target], dtype=torch.float32)
    return edge_index, edge_weight


def adjacency_edge_table(adjacency: np.ndarray) -> pd.DataFrame:
    """Return one row per undirected anatomical graph edge."""

    rows = []
    for source in range(len(REGION_NODES)):
        for target in range(source + 1, len(REGION_NODES)):
            if adjacency[source, target] > 0:
                rows.append(
                    {
                        "source_index": source,
                        "source_region": REGION_NODES[source].name,
                        "target_index": target,
                        "target_region": REGION_NODES[target].name,
                        "absolute_correlation_weight": float(adjacency[source, target]),
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    arguments = parser.parse_args()
    cfg = load_config(arguments.config)
    data = pd.read_csv(cfg.output_dir / "multimodal_features_long.csv")
    node_values = build_node_feature_array(data)
    scaler = fit_node_scaler(node_values)
    adjacency = structural_covariance_adjacency(
        transform_node_features(node_values, scaler), cfg.values["graph"]["top_k_neighbours"]
    )
    adjacency_edge_table(adjacency).to_csv(cfg.output_dir / "full_cohort_graph_edges_descriptive.csv", index=False)
    print(f"Built a descriptive {len(REGION_NODES)}-node graph; modelling uses fold-specific graphs.")
