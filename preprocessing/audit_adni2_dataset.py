"""Evaluate longitudinal ADSP-PHC MRI/PET NIfTI products in ADNI2_dataset.zip."""

from __future__ import annotations

import argparse
import statistics
import zipfile
from datetime import date
from pathlib import Path

import nibabel as nib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data" / "private" / "ADNI2_dataset.zip"
DEFAULT_CATALOG = ROOT / "data" / "private" / "ADNI2_catalog.csv"
DEFAULT_SOURCE_ACQUISITIONS = ROOT / "data" / "private" / "date_level_acquisitions.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "adni2_inventory"

PRODUCT_TYPES = {
    "ADSP-PHC_Reoriented_T1_Image": "MRI_T1",
    "ADSP-PHC_T1_Brain_Mask": "MRI_brain_mask",
    "ADSP-PHC_T1_MUSE_segmentation": "MRI_MUSE_segmentation",
    "ADSP-PHC_Coregistered_Normalized_PET_Image": "PET_normalized",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Audit longitudinal MRI/PET NIfTI products without extraction."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--source-acquisitions", type=Path, default=DEFAULT_SOURCE_ACQUISITIONS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-crc", action="store_true")
    return parser.parse_args()


def path_parts(name: str) -> tuple[str, ...]:
    """Return normalized archive path components."""

    return tuple(part for part in name.replace("\\", "/").split("/") if part)


def inventory_nifti(archive: Path) -> tuple[pd.DataFrame, dict]:
    """Read path metadata and NIfTI headers directly from the ZIP."""

    rows = []
    with zipfile.ZipFile(archive) as handle:
        members = [member for member in handle.infolist() if not member.is_dir()]
        for member in members:
            parts = path_parts(member.filename)
            if len(parts) < 6 or parts[0] != "ADNI" or not parts[-1].lower().endswith(".nii"):
                continue
            with handle.open(member) as stream:
                header = nib.Nifti1Header(binaryblock=stream.read(348))
            affine = header.get_best_affine()
            rows.append(
                {
                    "subject_id": parts[1],
                    "product_description": parts[2],
                    "product_type": PRODUCT_TYPES.get(parts[2], "other"),
                    "acquisition_datetime": parts[3],
                    "acq_date": parts[3][:10],
                    "image_id": parts[4],
                    "member": member.filename,
                    "uncompressed_bytes": member.file_size,
                    "compressed_bytes": member.compress_size,
                    "shape": "x".join(str(int(value)) for value in header.get_data_shape()),
                    "voxel_size": "x".join(
                        f"{float(value):.5f}" for value in header.get_zooms()
                    ),
                    "dtype": str(header.get_data_dtype()),
                    "orientation": "".join(nib.aff2axcodes(affine)),
                    "qform_code": int(header["qform_code"]),
                    "sform_code": int(header["sform_code"]),
                }
            )
    summary = {
        "archive_bytes": archive.stat().st_size,
        "uncompressed_bytes": sum(row["uncompressed_bytes"] for row in rows),
        "files": len(rows),
    }
    return pd.DataFrame(rows).sort_values(
        ["subject_id", "acq_date", "product_type", "image_id"]
    ), summary


def load_labels(catalog: Path) -> pd.DataFrame:
    """Collapse the catalogue to one supplied Research Group per participant."""

    data = pd.read_csv(catalog, dtype=str).fillna("")
    records = []
    for subject_id, group in data.groupby("Subject ID"):
        labels = sorted({value for value in group["Research Group"] if value})
        records.append(
            {
                "subject_id": subject_id,
                "research_group": "|".join(labels),
                "label_conflict": len(labels) > 1,
            }
        )
    return pd.DataFrame(records)


def add_pet_tracer(pet: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """Match normalized PET products to source DICOM tracer by subject and Acq Date."""

    output = pet.copy()
    output["tracer"] = "unknown"
    output["tracer_match_status"] = "source acquisition inventory unavailable"
    if not source_path.is_file():
        return output
    source = pd.read_csv(source_path, dtype=str).fillna("")
    source = source[source["modality"] == "PET"]
    mapping = (
        source.groupby(["subject_id", "acquisition_date"])["tracer"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .to_dict()
    )
    keys = list(zip(output["subject_id"], output["acq_date"]))
    output["tracer"] = [mapping.get(key, "unknown") for key in keys]
    output["tracer_match_status"] = output["tracer"].map(
        lambda value: "exact subject/Acq Date match" if value != "unknown" else "unmatched"
    )
    return output


def nearest_pairs(
    nifti: pd.DataFrame, pet: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Match each normalized PET Acq Date to the nearest T1 Acq Date."""

    t1 = nifti[nifti["product_type"] == "MRI_T1"]
    label_map = labels.set_index("subject_id")["research_group"].to_dict()
    records = []
    for row in pet.itertuples(index=False):
        candidates = t1[t1["subject_id"] == row.subject_id]
        if candidates.empty:
            nearest_date = ""
            interval = ""
            mri_member = ""
        else:
            pet_date = date.fromisoformat(row.acq_date)
            scored = candidates.assign(
                interval_days=candidates["acq_date"].map(
                    lambda value: abs((pet_date - date.fromisoformat(value)).days)
                )
            ).sort_values(["interval_days", "acq_date", "image_id"])
            best = scored.iloc[0]
            nearest_date = best["acq_date"]
            interval = int(best["interval_days"])
            mri_member = best["member"]
        records.append(
            {
                "subject_id": row.subject_id,
                "research_group": label_map.get(row.subject_id, ""),
                "tracer": row.tracer,
                "pet_acq_date": row.acq_date,
                "mri_acq_date": nearest_date,
                "absolute_interval_days": interval,
                "pet_member": row.member,
                "mri_member": mri_member,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["tracer", "subject_id", "pet_acq_date"]
    )


def subject_summary(
    nifti: pd.DataFrame, pet: pd.DataFrame, labels: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    """Summarize products and longitudinal imaging coverage per participant."""

    subjects = pd.DataFrame({"subject_id": sorted(nifti["subject_id"].unique())})
    subjects = subjects.merge(labels, on="subject_id", how="left")
    for product_type, column in (
        ("MRI_T1", "t1_dates"),
        ("MRI_brain_mask", "brain_mask_dates"),
        ("MRI_MUSE_segmentation", "muse_dates"),
        ("PET_normalized", "pet_dates"),
    ):
        counts = (
            nifti[nifti["product_type"] == product_type]
            .groupby("subject_id")["acq_date"]
            .nunique()
        )
        subjects[column] = subjects["subject_id"].map(counts).fillna(0).astype(int)
    tracer_map = pet.groupby("subject_id")["tracer"].agg(
        lambda values: "|".join(sorted(set(values)))
    )
    subjects["pet_tracers"] = subjects["subject_id"].map(tracer_map).fillna("")
    valid = pairs[pairs["absolute_interval_days"] != ""].copy()
    valid["absolute_interval_days"] = valid["absolute_interval_days"].astype(int)
    for window in (0, 7, 30, 90):
        eligible = valid[valid["absolute_interval_days"] <= window]
        counts = eligible.groupby("subject_id").size()
        subjects[f"paired_visits_within_{window}_days"] = (
            subjects["subject_id"].map(counts).fillna(0).astype(int)
        )
    return subjects


def pairing_summaries(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize cross-sectional and repeated paired imaging by Acq Date window."""

    valid = pairs[pairs["absolute_interval_days"] != ""].copy()
    valid["absolute_interval_days"] = valid["absolute_interval_days"].astype(int)
    cross_rows = []
    longitudinal_rows = []
    for tracer in sorted(valid["tracer"].unique()):
        tracer_rows = valid[valid["tracer"] == tracer]
        for window in (0, 7, 30, 90, 180, 365):
            eligible = tracer_rows[tracer_rows["absolute_interval_days"] <= window]
            participants = eligible[["subject_id", "research_group"]].drop_duplicates()
            counts = participants["research_group"].value_counts()
            cross_rows.append(
                {
                    "tracer": tracer,
                    "maximum_interval_days": window,
                    "participants": participants["subject_id"].nunique(),
                    "AD": int(counts.get("AD", 0)),
                    "CN": int(counts.get("CN", 0)),
                    "MCI": int(counts.get("MCI", 0)),
                }
            )
            repeated_counts = eligible.groupby("subject_id").size()
            repeated_ids = set(repeated_counts[repeated_counts >= 2].index)
            repeated = participants[participants["subject_id"].isin(repeated_ids)]
            repeated_labels = repeated["research_group"].value_counts()
            longitudinal_rows.append(
                {
                    "tracer": tracer,
                    "maximum_interval_days": window,
                    "participants_with_at_least_2_paired_visits": len(repeated_ids),
                    "AD": int(repeated_labels.get("AD", 0)),
                    "CN": int(repeated_labels.get("CN", 0)),
                    "MCI": int(repeated_labels.get("MCI", 0)),
                }
            )
    return pd.DataFrame(cross_rows), pd.DataFrame(longitudinal_rows)


def product_summary(nifti: pd.DataFrame) -> pd.DataFrame:
    """Summarize files, participants and spatial grids for each NIfTI product."""

    return (
        nifti.groupby(
            ["product_description", "product_type", "shape", "voxel_size", "dtype"],
            as_index=False,
        )
        .agg(files=("member", "size"), participants=("subject_id", "nunique"))
        .sort_values(["product_type", "files"], ascending=[True, False])
    )


def write_report(
    path: Path,
    archive: Path,
    summary: dict,
    nifti: pd.DataFrame,
    pet: pd.DataFrame,
    subjects: pd.DataFrame,
    cross: pd.DataFrame,
    longitudinal: pd.DataFrame,
    crc_result: str,
) -> None:
    """Write a thesis-safe interpretation of the longitudinal NIfTI archive."""

    product_counts = nifti["product_type"].value_counts().to_dict()
    product_subjects = nifti.groupby("product_type")["subject_id"].nunique().to_dict()
    labels = subjects["research_group"].value_counts().to_dict()
    tracer_counts = pet["tracer"].value_counts().to_dict()
    matched_tracers = int((pet["tracer"] != "unknown").sum())
    repeated_t1 = subjects[subjects["t1_dates"] >= 2]
    repeated_pet = subjects[subjects["pet_dates"] >= 2]
    repeated_both = subjects[(subjects["t1_dates"] >= 2) & (subjects["pet_dates"] >= 2)]
    repeated_both_labels = repeated_both["research_group"].value_counts().to_dict()
    cross_90 = cross[cross["maximum_interval_days"] == 90]
    long_90 = longitudinal[longitudinal["maximum_interval_days"] == 90]
    cross_lines = [
        f"- {row.tracer}: {row.participants} participants "
        f"(AD {row.AD}, CN {row.CN}, MCI {row.MCI})."
        for row in cross_90.itertuples(index=False)
    ]
    long_lines = [
        f"- {row.tracer}: {row.participants_with_at_least_2_paired_visits} participants "
        f"(AD {row.AD}, CN {row.CN}, MCI {row.MCI})."
        for row in long_90.itertuples(index=False)
    ]
    report = f"""# Evaluation of ADNI2_dataset.zip

## Archive Integrity and Products

- Source: `{archive}`.
- Compressed size: {summary['archive_bytes'] / 1e9:.2f} GB.
- Uncompressed NIfTI content: {summary['uncompressed_bytes'] / 1e9:.2f} GB.
- NIfTI files: {summary['files']:,}; ZIP CRC verification: {crc_result}.
- Product files: {product_counts}.
- Product participant counts: {product_subjects}.
- Participant union: {len(subjects)}; supplied labels: {labels}.

This archive contains analysis-ready NIfTI products rather than the many DICOM
derivatives in `ADNI2.zip`. It includes longitudinal T1 MRI, brain masks, MUSE
segmentations and normalized PET. The files are repeated visits, not independent
participants.

## Tracer and Acq Date Linkage

{matched_tracers} of {len(pet)} normalized PET files were matched by exact
participant ID and **Acq Date** to the source DICOM acquisition inventory.
Matched PET files by tracer are {tracer_counts}. Named catalogue `Visit` values
were not used for temporal pairing.

Participants with MRI and PET within 90 Acq Date days:

{chr(10).join(cross_lines)}

## Longitudinal Imaging

Participants with at least two T1 dates: {len(repeated_t1)}. Participants with
at least two PET dates: {len(repeated_pet)}. A total of {len(repeated_both)}
participants have at least two dates in each modality; their labels are
{repeated_both_labels}.

Participants with at least two MRI/PET pairs within 90 days:

{chr(10).join(long_lines)}

The separate modalities can support longitudinal regional-change and
graph-change analysis within the supplied diagnostic groups. However, close
multimodal pairing is limited: a participant may have repeated MRI and repeated
PET without having two temporally close MRI/PET pairs. Repeated imaging does not,
by itself, provide a future conversion outcome.

## Diagnosis and Prediction

Current AD-versus-CN discrimination is feasible after quality control and
participant-level validation. This archive is preferable to the raw DICOM
archive for ROI analysis because masks and MUSE segmentations are supplied and
PET is already normalized.

Future MCI-to-AD prediction is still not labelled. Every participant has one
stable catalogue Research Group value, and no dated diagnosis-transition table
is present. A model may analyse longitudinal imaging change, but it cannot be
trained or evaluated as conversion prediction until exact participants receive
baseline and follow-up clinical diagnoses.

## Recommended Use

Use one MRI/PET pair per participant for the primary AD-versus-CN diagnostic
analysis. Use repeated pairs only in a secondary longitudinal imaging analysis,
with all visits from a participant kept in the same validation fold. For AAL-116
graphs, transform both modalities into one documented atlas space; do not apply
native MUSE labels directly to normalized PET without registration or mapping.
"""
    path.write_text(report, encoding="utf-8")


def main() -> int:
    """Run the complete longitudinal NIfTI archive evaluation."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nifti, summary = inventory_nifti(args.archive)
    labels = load_labels(args.catalog)
    pet = add_pet_tracer(
        nifti[nifti["product_type"] == "PET_normalized"], args.source_acquisitions
    )
    pairs = nearest_pairs(nifti, pet, labels)
    subjects = subject_summary(nifti, pet, labels, pairs)
    cross, longitudinal = pairing_summaries(pairs)
    products = product_summary(nifti)

    crc_result = "not requested"
    if args.verify_crc:
        with zipfile.ZipFile(args.archive) as handle:
            failing_member = handle.testzip()
        crc_result = "passed" if failing_member is None else f"failed at {failing_member}"

    nifti.to_csv(args.output_dir / "nifti_inventory.csv", index=False)
    pet.to_csv(args.output_dir / "pet_tracer_inventory.csv", index=False)
    pairs.to_csv(args.output_dir / "pet_mri_acq_date_pairs.csv", index=False)
    subjects.to_csv(args.output_dir / "subject_longitudinal_summary.csv", index=False)
    cross.to_csv(args.output_dir / "pairing_window_summary.csv", index=False)
    longitudinal.to_csv(
        args.output_dir / "longitudinal_pairing_summary.csv", index=False
    )
    products.to_csv(args.output_dir / "product_spatial_summary.csv", index=False)
    write_report(
        args.output_dir / "ADNI2_DATASET_ZIP_EVALUATION.md",
        args.archive,
        summary,
        nifti,
        pet,
        subjects,
        cross,
        longitudinal,
        crc_result,
    )
    print(
        f"Evaluated {len(nifti)} NIfTI files from {len(subjects)} participants; "
        f"matched {int((pet['tracer'] != 'unknown').sum())}/{len(pet)} PET tracers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
