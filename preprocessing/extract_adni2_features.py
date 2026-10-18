"""Extract transparent MRI and AV45 PET features for ADNI2 analyses.

The script streams NIfTI members directly from ADNI2_dataset.zip so the 40 GB
archive does not need to be extracted. MRI regional volumes use supplied MUSE
segmentations and the official MUSE intracranial-volume definition. PET regional
uptake uses an AAL atlas resampled once to the normalized PET grid.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostic.adni2_pipeline.regions import REGION_NODES


DEFAULT_ARCHIVE = ROOT / "data" / "private" / "ADNI2_dataset.zip"
DEFAULT_PAIRS = ROOT / "data" / "processed" / "adni2_inventory" / "pet_mri_acq_date_pairs.csv"
DEFAULT_INVENTORY = ROOT / "data" / "processed" / "adni2_inventory" / "nifti_inventory.csv"
DEFAULT_MUSE_MAPPING = ROOT / "data" / "private" / "MUSE_mapping_consecutive_indices.csv"
DEFAULT_MUSE_COMPLETE = ROOT / "data" / "private" / "MUSE_ROI_complete_list.csv"
DEFAULT_AAL = ROOT / "data" / "private" / "ROI_MNI_V4.nii"
DEFAULT_AAL_LABELS = ROOT / "data" / "private" / "aal_labels.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "adni2_features"

# The broad candidate set is fixed independently of model outcomes. The GNN
# performs supervised ranking only inside each outer training fold.
TARGET_MUSE_REGIONS = tuple(node.muse_region for node in REGION_NODES)
TARGET_AAL_REGIONS = tuple(node.aal_region for node in REGION_NODES)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Extract MUSE MRI and AAL AV45 features directly from an ADNI ZIP."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--muse-mapping", type=Path, default=DEFAULT_MUSE_MAPPING)
    parser.add_argument("--muse-complete", type=Path, default=DEFAULT_MUSE_COMPLETE)
    parser.add_argument("--aal", type=Path, default=DEFAULT_AAL)
    parser.add_argument("--aal-labels", type=Path, default=DEFAULT_AAL_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--maximum-interval-days",
        type=int,
        default=90,
        help="Maximum absolute MRI/PET Acq Date interval.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("AD", "CN", "MCI"),
        default=("AD", "CN"),
        help="Diagnostic archive groups to extract; defaults to the primary AD/CN cohort.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for the complete cohort.",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    """Configure console and file logging."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("adni2_multimodal_feature_extraction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(output_dir / "feature_extraction.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def select_primary_cohort(
    pairs_path: Path,
    maximum_days: int,
    groups: tuple[str, ...] | list[str] = ("AD", "CN"),
) -> pd.DataFrame:
    """Select one earliest eligible AV45 pair per participant in requested groups."""

    pairs = pd.read_csv(pairs_path, dtype=str).fillna("")
    pairs = pairs[
        pairs["research_group"].isin(groups)
        & (pairs["tracer"] == "AV45_amyloid")
        & (pairs["absolute_interval_days"] != "")
    ].copy()
    pairs["absolute_interval_days"] = pairs["absolute_interval_days"].astype(int)
    pairs = pairs[pairs["absolute_interval_days"] <= maximum_days]
    pairs = pairs.sort_values(
        ["subject_id", "pet_acq_date", "absolute_interval_days", "mri_acq_date"]
    ).drop_duplicates("subject_id", keep="first")
    return pairs.reset_index(drop=True)


def load_nifti_from_zip(handle: zipfile.ZipFile, member: str) -> nib.Nifti1Image:
    """Read one uncompressed NIfTI member into memory."""

    payload = handle.read(member)
    return nib.Nifti1Image.from_bytes(payload)


def parse_derived_roi_ids(complete: pd.DataFrame, derived_id: int) -> set[int]:
    """Parse the official MUSE component IDs for one derived ROI."""

    row = complete.loc[complete["ID"] == derived_id]
    if len(row) != 1:
        raise ValueError(f"Expected one MUSE derived ROI row for ID {derived_id}")
    value = str(row.iloc[0]["Consisting_of_ROIS"])
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def create_pet_atlas(
    handle: zipfile.ZipFile,
    pet_member: str,
    aal_path: Path,
    output_path: Path,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Resample AAL labels to the common PET grid with nearest-neighbor interpolation."""

    pet_image = load_nifti_from_zip(handle, pet_member)
    atlas_image = nib.load(str(aal_path))
    atlas_pet = resample_from_to(
        atlas_image,
        (pet_image.shape, pet_image.affine),
        order=0,
    )
    atlas_data = np.rint(np.asanyarray(atlas_pet.dataobj)).astype(np.int16)
    atlas_pet = nib.Nifti1Image(atlas_data, pet_image.affine, pet_image.header)
    atlas_pet.set_data_dtype(np.int16)
    nib.save(atlas_pet, str(output_path))
    return pet_image, atlas_data


def save_atlas_qc(
    pet_image: nib.Nifti1Image,
    atlas_data: np.ndarray,
    output_path: Path,
) -> None:
    """Save central PET slices with AAL boundaries for visual overlap review."""

    pet = np.asarray(pet_image.dataobj, dtype=np.float32)
    positive = pet[pet > 0]
    scale = float(np.percentile(positive, 99)) if positive.size else 1.0
    pet = np.clip(pet / max(scale, 1e-6), 0, 1)
    center = tuple(size // 2 for size in pet.shape)
    views = (
        (pet[:, :, center[2]], atlas_data[:, :, center[2]], "Axial"),
        (pet[:, center[1], :], atlas_data[:, center[1], :], "Coronal"),
        (pet[center[0], :, :], atlas_data[center[0], :, :], "Sagittal"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (pet_slice, atlas_slice, title) in zip(axes, views):
        axis.imshow(np.rot90(pet_slice), cmap="gray")
        axis.contour(np.rot90(atlas_slice > 0), levels=[0.5], colors="cyan", linewidths=0.5)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("Normalized AV45 PET and Resampled AAL Coverage QC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def extract_mri_features(
    segmentation_image: nib.Nifti1Image,
    muse_labels: pd.DataFrame,
    icv_ids: set[int],
) -> tuple[dict[str, float], list[dict[str, object]], dict[str, float]]:
    """Compute targeted MUSE volumes and official ICV-normalized fractions."""

    segmentation = np.rint(np.asanyarray(segmentation_image.dataobj)).astype(np.int16)
    voxel_volume_ml = abs(float(np.linalg.det(segmentation_image.affine[:3, :3]))) / 1000.0
    counts = np.bincount(segmentation.ravel().clip(min=0))
    count_for = lambda roi_id: int(counts[roi_id]) if roi_id < len(counts) else 0
    icv_ml = sum(count_for(roi_id) for roi_id in icv_ids) * voxel_volume_ml
    if icv_ml <= 0:
        raise ValueError("MUSE-derived intracranial volume is zero")

    targeted = muse_labels[muse_labels["ROINameMUSE"].isin(TARGET_MUSE_REGIONS)].copy()
    found = set(targeted["ROINameMUSE"])
    missing = set(TARGET_MUSE_REGIONS).difference(found)
    if missing:
        raise ValueError(f"Missing targeted MUSE labels: {sorted(missing)}")

    features = {}
    long_rows = []
    for row in targeted.itertuples(index=False):
        roi_id = int(row.IndexMUSE)
        volume_ml = count_for(roi_id) * voxel_volume_ml
        fraction = volume_ml / icv_ml
        name = str(row.ROINameMUSE)
        feature_name = f"mri_{name}_icv_fraction"
        features[feature_name] = fraction
        long_rows.append(
            {
                "modality": "MRI",
                "atlas": "MUSE",
                "region_id": roi_id,
                "region_name": name,
                "raw_value": volume_ml,
                "normalized_value": fraction,
                "normalization": "MUSE_ICV_fraction",
            }
        )
    qc = {
        "muse_icv_ml": icv_ml,
        "muse_nonzero_labels": float(np.unique(segmentation[segmentation > 0]).size),
        "muse_voxel_volume_ml": voxel_volume_ml,
    }
    return features, long_rows, qc


def extract_pet_features(
    pet_image: nib.Nifti1Image,
    atlas_data: np.ndarray,
    aal_labels: pd.DataFrame,
) -> tuple[dict[str, float], list[dict[str, object]], dict[str, float]]:
    """Compute targeted AAL uptake ratios using an explicit cerebellar reference."""

    pet = np.asarray(pet_image.dataobj, dtype=np.float32)
    target = aal_labels[aal_labels["region_name"].isin(TARGET_AAL_REGIONS)].copy()
    found = set(target["region_name"])
    missing = set(TARGET_AAL_REGIONS).difference(found)
    if missing:
        raise ValueError(f"Missing targeted AAL labels: {sorted(missing)}")

    cerebellar = aal_labels[
        aal_labels["region_name"].str.startswith(("Cerebelum_", "Vermis_"))
    ]
    reference_mask = np.isin(atlas_data, cerebellar["region_id"].astype(int).to_numpy())
    reference_values = pet[reference_mask & np.isfinite(pet) & (pet > 0)]
    if reference_values.size == 0:
        raise ValueError("No positive PET voxels in the AAL cerebellar reference")
    reference_mean = float(np.mean(reference_values))
    if reference_mean <= 0:
        raise ValueError("AAL cerebellar PET reference mean is non-positive")

    features = {}
    long_rows = []
    weighted_sum = 0.0
    weighted_voxels = 0
    coverages = []
    for row in target.itertuples(index=False):
        roi_id = int(row.region_id)
        roi_mask = atlas_data == roi_id
        roi_voxels = int(np.count_nonzero(roi_mask))
        valid_mask = roi_mask & np.isfinite(pet) & (pet > 0)
        valid_voxels = int(np.count_nonzero(valid_mask))
        coverage = valid_voxels / roi_voxels if roi_voxels else 0.0
        mean_uptake = float(np.mean(pet[valid_mask])) if valid_voxels else np.nan
        ratio = mean_uptake / reference_mean if np.isfinite(mean_uptake) else np.nan
        name = str(row.region_name)
        feature_name = f"pet_{name}_cerebellar_ratio"
        features[feature_name] = ratio
        weighted_sum += ratio * valid_voxels if np.isfinite(ratio) else 0.0
        weighted_voxels += valid_voxels if np.isfinite(ratio) else 0
        coverages.append(coverage)
        long_rows.append(
            {
                "modality": "PET",
                "atlas": "AAL_SPM12",
                "region_id": roi_id,
                "region_name": name,
                "raw_value": mean_uptake,
                "normalized_value": ratio,
                "normalization": "AAL_cerebellar_reference_ratio",
                "positive_voxel_coverage": coverage,
            }
        )
    features["pet_target_panel_mean_cerebellar_ratio"] = (
        weighted_sum / weighted_voxels if weighted_voxels else np.nan
    )
    qc = {
        "pet_cerebellar_reference_mean": reference_mean,
        "pet_min_target_coverage": min(coverages) if coverages else 0.0,
        "pet_mean_target_coverage": float(np.mean(coverages)) if coverages else 0.0,
    }
    return features, long_rows, qc


def build_member_lookup(inventory: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    """Build a deterministic subject/date/product-to-member lookup."""

    ordered = inventory.sort_values(["subject_id", "acq_date", "product_type", "image_id"])
    lookup = {}
    for row in ordered.itertuples(index=False):
        lookup.setdefault((row.subject_id, row.acq_date, row.product_type), row.member)
    return lookup


def write_feature_dictionary(path: Path, wide: pd.DataFrame) -> None:
    """Document every primary model feature and its normalization."""

    rows = []
    for feature in wide.columns:
        if feature.startswith("mri_") and feature.endswith("_icv_fraction"):
            rows.append(
                {
                    "feature": feature,
                    "modality": "MRI",
                    "source": "MUSE segmentation",
                    "normalization": "regional volume / official MUSE ICV",
                    "primary_model_feature": True,
                }
            )
        elif feature.startswith("pet_") and feature.endswith("_cerebellar_ratio"):
            rows.append(
                {
                    "feature": feature,
                    "modality": "PET",
                    "source": "normalized AV45 PET with AAL-SPM12",
                    "normalization": "regional uptake / AAL cerebellar reference",
                    "primary_model_feature": True,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    """Extract the complete primary diagnostic feature table."""

    args = parse_args()
    logger = configure_logging(args.output_dir)
    cohort = select_primary_cohort(args.pairs, args.maximum_interval_days, args.groups)
    if args.max_subjects is not None:
        cohort = cohort.head(args.max_subjects)
    logger.info(
        "Selected %d participants: %s",
        len(cohort),
        cohort["research_group"].value_counts().to_dict(),
    )

    inventory = pd.read_csv(args.inventory, dtype=str).fillna("")
    lookup = build_member_lookup(inventory)
    muse_labels = pd.read_csv(args.muse_mapping)
    muse_complete = pd.read_csv(args.muse_complete)
    icv_ids = parse_derived_roi_ids(muse_complete, 702)
    aal_labels = pd.read_csv(args.aal_labels)

    wide_rows = []
    long_rows = []
    qc_rows = []
    failures = []
    atlas_path = args.output_dir / "aal_spm12_in_normalized_pet_space.nii.gz"
    atlas_qc_path = args.output_dir / "aal_pet_overlap_qc.png"

    with zipfile.ZipFile(args.archive) as handle:
        first = cohort.iloc[0]
        first_pet = load_nifti_from_zip(handle, first["pet_member"])
        _, atlas_data = create_pet_atlas(
            handle,
            first["pet_member"],
            args.aal,
            atlas_path,
        )
        save_atlas_qc(first_pet, atlas_data, atlas_qc_path)

        for index, row in enumerate(cohort.itertuples(index=False), start=1):
            try:
                muse_member = lookup[(row.subject_id, row.mri_acq_date, "MRI_MUSE_segmentation")]
                segmentation = load_nifti_from_zip(handle, muse_member)
                pet_image = load_nifti_from_zip(handle, row.pet_member)
                if pet_image.shape != atlas_data.shape:
                    raise ValueError(
                        f"PET shape {pet_image.shape} does not match common atlas {atlas_data.shape}"
                    )
                mri_features, mri_long, mri_qc = extract_mri_features(
                    segmentation, muse_labels, icv_ids
                )
                pet_features, pet_long, pet_qc = extract_pet_features(
                    pet_image, atlas_data, aal_labels
                )
                metadata = {
                    "subject_id": row.subject_id,
                    "diagnosis": row.research_group,
                    "target_ad": int(row.research_group == "AD"),
                    "mri_acq_date": row.mri_acq_date,
                    "pet_acq_date": row.pet_acq_date,
                    "absolute_interval_days": int(row.absolute_interval_days),
                }
                wide_rows.append({**metadata, **mri_features, **pet_features})
                for feature_row in [*mri_long, *pet_long]:
                    long_rows.append({**metadata, **feature_row})
                qc_rows.append({**metadata, **mri_qc, **pet_qc})
            except Exception as exc:
                logger.exception("Feature extraction failed for %s", row.subject_id)
                failures.append({"subject_id": row.subject_id, "error": str(exc)})
            if index % 10 == 0 or index == len(cohort):
                logger.info("Processed %d/%d participants", index, len(cohort))

    wide = pd.DataFrame(wide_rows).sort_values("subject_id")
    long = pd.DataFrame(long_rows).sort_values(["subject_id", "modality", "region_name"])
    qc = pd.DataFrame(qc_rows).sort_values("subject_id")
    failure_df = pd.DataFrame(failures, columns=["subject_id", "error"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output_dir / "primary_cohort.csv", index=False)
    wide.to_csv(args.output_dir / "multimodal_features_wide.csv", index=False)
    long.to_csv(args.output_dir / "multimodal_features_long.csv", index=False)
    qc.to_csv(args.output_dir / "feature_extraction_qc.csv", index=False)
    failure_df.to_csv(args.output_dir / "feature_extraction_failures.csv", index=False)
    write_feature_dictionary(args.output_dir / "feature_dictionary.csv", wide)

    logger.info(
        "Completed feature extraction: %d successful, %d failed, %d model features",
        len(wide),
        len(failure_df),
        sum(
            column.endswith(("_icv_fraction", "_cerebellar_ratio"))
            for column in wide.columns
        ),
    )
    return 0 if failure_df.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
