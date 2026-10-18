"""Derive thesis-safe longitudinal diagnosis and progression labels from ADNI TSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "participant_id",
    "session_id",
    "diagnosis",
    "examination_date",
]


def load_diagnosis_tables(tsv_paths: list[Path]) -> pd.DataFrame:
    """Load and combine ADNI diagnosis TSV files."""

    frames = []
    for path in tsv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Diagnosis TSV does not exist: {path}")
        df = pd.read_csv(path, sep="\t")
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["participant_id", "session_id", "diagnosis", "examination_date"]
    ).copy()
    combined["examination_date"] = pd.to_datetime(combined["examination_date"], errors="coerce")
    combined["subject_id"] = combined["participant_id"].str.replace("sub-ADNI", "", regex=False)
    return combined.sort_values(["subject_id", "examination_date", "session_id"]).reset_index(drop=True)


def derive_progression_labels(diagnosis_df: pd.DataFrame) -> pd.DataFrame:
    """Derive one summary row per subject with longitudinal progression labels."""

    rows: list[dict[str, object]] = []

    for subject_id, subject_df in diagnosis_df.groupby("subject_id", sort=True):
        ordered = subject_df.sort_values(["examination_date", "session_id"]).reset_index(drop=True)
        ordered = ordered.dropna(subset=["diagnosis"]).reset_index(drop=True)
        if ordered.empty:
            continue
        diagnoses = ordered["diagnosis"].astype(str).tolist()
        first_diagnosis = diagnoses[0]
        last_diagnosis = diagnoses[-1]

        progression_label = classify_progression(first_diagnosis, last_diagnosis, diagnoses)
        rows.append(
            {
                "subject_id": subject_id,
                "n_timepoints": int(len(ordered)),
                "first_session_id": str(ordered.iloc[0]["session_id"]),
                "last_session_id": str(ordered.iloc[-1]["session_id"]),
                "first_exam_date": ordered.iloc[0]["examination_date"].date().isoformat()
                if pd.notna(ordered.iloc[0]["examination_date"])
                else "",
                "last_exam_date": ordered.iloc[-1]["examination_date"].date().isoformat()
                if pd.notna(ordered.iloc[-1]["examination_date"])
                else "",
                "first_diagnosis": first_diagnosis,
                "last_diagnosis": last_diagnosis,
                "diagnosis_sequence": " -> ".join(diagnoses),
                "progression_label": progression_label,
            }
        )

    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def classify_progression(
    first_diagnosis: str,
    last_diagnosis: str,
    diagnosis_sequence: list[str],
) -> str:
    """Assign a simple thesis-safe progression label."""

    if all(label == "CN" for label in diagnosis_sequence):
        return "CN_stable"
    if all(label == "AD" for label in diagnosis_sequence):
        return "AD_stable"
    if first_diagnosis == "MCI" and "AD" in diagnosis_sequence[1:]:
        return "MCI_to_AD"
    if first_diagnosis == "MCI" and all(label == "MCI" for label in diagnosis_sequence):
        return "MCI_stable"
    if first_diagnosis == "CN" and "MCI" in diagnosis_sequence[1:] and "AD" not in diagnosis_sequence[1:]:
        return "CN_to_MCI"
    if first_diagnosis == "CN" and "AD" in diagnosis_sequence[1:]:
        return "CN_to_AD"
    if first_diagnosis == "MCI" and "CN" in diagnosis_sequence[1:] and "AD" not in diagnosis_sequence:
        return "MCI_reversion_to_CN"
    if first_diagnosis == last_diagnosis:
        return f"{first_diagnosis}_stable"
    return "mixed_or_other_progression"


def build_overlap_table(
    progression_df: pd.DataFrame,
    graph_subject_ids: set[str],
) -> pd.DataFrame:
    """Mark which labeled subjects currently overlap with this repository's outputs."""

    overlap_df = progression_df.copy()
    overlap_df["present_in_current_outputs"] = overlap_df["subject_id"].isin(graph_subject_ids)
    return overlap_df


def collect_current_subject_ids(project_root: Path) -> set[str]:
    """Collect subject IDs from current CSV outputs when available."""

    subject_ids: set[str] = set()
    candidate_csvs = [
        project_root / "outputs_mri_aal" / "graph_metrics.csv",
        project_root / "outputs_pet_aal" / "graph_metrics.csv",
        project_root / "roi_outputs" / "053_S_1044_2007-06-04_roi_features.csv",
        project_root / "roi_outputs" / "053_S_1044_2007-11-21_roi_features.csv",
    ]
    for csv_path in candidate_csvs:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "subject_id" in df.columns:
            subject_ids.update(df["subject_id"].dropna().astype(str).tolist())
    return subject_ids


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Derive ADNI progression labels from diagnosis TSV files."
    )
    parser.add_argument(
        "--tsv-files",
        type=Path,
        nargs="+",
        required=True,
        help="Diagnosis TSV files, for example train/val/test ADNI diagnosis tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("adni_progression_labels"),
        help="Directory for derived label tables.",
    )
    return parser.parse_args()


def main() -> None:
    """Create merged diagnosis and progression-label outputs."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnosis_df = load_diagnosis_tables(args.tsv_files)
    progression_df = derive_progression_labels(diagnosis_df)
    overlap_df = build_overlap_table(
        progression_df,
        graph_subject_ids=collect_current_subject_ids(Path.cwd()),
    )

    merged_output = args.output_dir / "combined_diagnosis_longitudinal.csv"
    progression_output = args.output_dir / "derived_progression_labels.csv"
    overlap_output = args.output_dir / "derived_progression_labels_with_overlap.csv"

    diagnosis_df.to_csv(merged_output, index=False)
    progression_df.to_csv(progression_output, index=False)
    overlap_df.to_csv(overlap_output, index=False)

    print(f"Saved merged diagnosis table: {merged_output}")
    print(f"Saved progression labels: {progression_output}")
    print(f"Saved overlap table: {overlap_output}")


if __name__ == "__main__":
    main()
