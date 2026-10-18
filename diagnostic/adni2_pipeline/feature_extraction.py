"""Prepare validated regional MRI/PET feature tables for modelling."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .regions import REGION_NODES, feature_columns


def regional_means(image_path: str | Path, label_path: str | Path, labels: list[int]) -> dict[int, float]:
    """Compute image means inside integer atlas labels on an identical voxel grid."""

    image = nib.load(str(image_path))
    atlas = nib.load(str(label_path))
    if image.shape[:3] != atlas.shape[:3] or not np.allclose(image.affine, atlas.affine, atol=1e-3):
        raise ValueError("Image and label map must be resampled to the same grid before extraction.")
    values = np.asarray(image.dataobj, dtype=np.float32)
    regions = np.asarray(atlas.dataobj)
    return {
        label: float(np.nanmean(values[regions == label])) if np.any(regions == label) else np.nan
        for label in labels
    }


def regional_volumes(label_path: str | Path, labels: list[int], icv_label: int | None = None) -> dict[int, float]:
    """Compute regional volumes and optionally divide them by an ICV label volume."""

    atlas = nib.load(str(label_path))
    regions = np.asarray(atlas.dataobj)
    voxel_volume = float(abs(np.linalg.det(atlas.affine[:3, :3])))
    denominator = float(np.count_nonzero(regions == icv_label) * voxel_volume) if icv_label else 1.0
    if denominator <= 0:
        raise ValueError("The requested ICV label has zero volume.")
    return {label: float(np.count_nonzero(regions == label) * voxel_volume / denominator) for label in labels}


def validate_feature_table(table: pd.DataFrame) -> None:
    """Check labels, participant uniqueness, and every matched regional feature."""

    required = {"subject_id", "diagnosis", "target_ad", *feature_columns()}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Validated feature table is missing columns: {missing}")
    if table["subject_id"].duplicated().any():
        raise ValueError("The diagnostic feature table must contain one row per participant.")
    if not set(table["diagnosis"]).issubset({"AD", "CN"}):
        raise ValueError("The primary diagnostic table must contain only AD and CN labels.")
    values = table[feature_columns()].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Regional model features contain missing or infinite values.")


def prepare_feature_tables(config: PipelineConfig) -> pd.DataFrame:
    """Split the verified wide table into the requested thesis-friendly outputs."""

    source = config.data_path("feature_table")
    table = pd.read_csv(source)
    validate_feature_table(table)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    identifiers = [
        column for column in (
            "subject_id", "diagnosis", "target_ad", "mri_acq_date", "pet_acq_date", "absolute_interval_days"
        ) if column in table
    ]
    mri = table[identifiers + feature_columns("MRI")].copy()
    pet = table[identifiers + feature_columns("PET")].copy()
    multimodal = table[identifiers + feature_columns()].copy()
    multimodal["visit_id"] = (
        multimodal["subject_id"].astype(str) + "_" + pd.to_datetime(
            multimodal["mri_acq_date"], errors="coerce"
        ).dt.strftime("%Y%m%d").fillna("baseline")
    )
    mri.to_csv(output / "mri_regional_features.csv", index=False)
    pet.to_csv(output / "pet_regional_features.csv", index=False)
    multimodal.to_csv(output / "multimodal_features_long.csv", index=False)
    pd.DataFrame(
        [
            {
                "node_index": index,
                "region": region.name,
                "hemisphere": region.hemisphere,
                "structure": region.structure,
                "mri_feature": region.mri_feature,
                "pet_feature": region.pet_feature,
            }
            for index, region in enumerate(REGION_NODES)
        ]
    ).to_csv(output / "graph_node_dictionary.csv", index=False)
    (output / "feature_preparation_provenance.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "participants": len(table),
                "matched_graph_nodes": len(REGION_NODES),
                "mri_features": len(feature_columns("MRI")),
                "pet_features": len(feature_columns("PET")),
                "normalisation": {
                    "MRI": "MUSE regional volume divided by official MUSE intracranial volume",
                    "PET": "AAL regional mean divided by the prespecified AAL cerebellar reference",
                },
                "candidate_region_policy": "Broad bilateral MUSE-AAL anatomical matches fixed before GNN fitting",
                "note": "MUSE entorhinal regions remain excluded because AAL-SPM12 has no directly matched entorhinal label.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return multimodal


if __name__ == "__main__":
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    arguments = parser.parse_args()
    prepared = prepare_feature_tables(load_config(arguments.config))
    print(f"Prepared regional features for {len(prepared)} participants")
