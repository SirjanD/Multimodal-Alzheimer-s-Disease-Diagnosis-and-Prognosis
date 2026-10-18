"""Build participant/visit tables and derive survival outcomes when labels permit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig


class OutcomeUnavailableError(RuntimeError):
    """Raised when time-to-event labels cannot be derived without invention."""


def _normalise_diagnosis(values: pd.Series) -> pd.Series:
    """Map common diagnosis spellings to CN, MCI, and AD."""

    mapping = {
        "CN": "CN",
        "NORMAL": "CN",
        "COGNITIVELY NORMAL": "CN",
        "MCI": "MCI",
        "LMCI": "MCI",
        "EMCI": "MCI",
        "AD": "AD",
        "ALZHEIMER'S DISEASE": "AD",
        "ALZHEIMERS DISEASE": "AD",
    }
    return values.astype(str).str.strip().str.upper().map(mapping)


def build_master_from_pair_inventory(
    pair_inventory: str | Path,
    pairing_window_days: int = 90,
) -> pd.DataFrame:
    """Adapt the verified ZIP acquisition inventory to one row per paired visit.

    The archive's ``research_group`` is a static folder label, not a longitudinal
    diagnosis assessment. This distinction is retained explicitly so the table
    cannot silently be used to invent conversion events.
    """

    pairs = pd.read_csv(pair_inventory)
    required = {
        "subject_id", "research_group", "pet_acq_date", "mri_acq_date",
        "absolute_interval_days", "pet_member", "mri_member",
    }
    missing = sorted(required.difference(pairs.columns))
    if missing:
        raise ValueError(f"Pair inventory is missing columns: {missing}")

    pairs["MRI_date"] = pd.to_datetime(pairs["mri_acq_date"], errors="coerce")
    pairs["PET_date"] = pd.to_datetime(pairs["pet_acq_date"], errors="coerce")
    pairs["visit_date"] = pairs[["MRI_date", "PET_date"]].min(axis=1)
    pairs["mri_pet_within_90d"] = pairs["absolute_interval_days"].le(pairing_window_days)
    unique_mri_visits = pairs.groupby("subject_id")["MRI_date"].transform("nunique")
    unique_pet_visits = pairs.groupby("subject_id")["PET_date"].transform("nunique")
    pairs["has_longitudinal"] = unique_mri_visits.ge(2) & unique_pet_visits.ge(2)
    pairs["visit_id"] = (
        pairs["subject_id"].astype(str) + "_" + pairs["visit_date"].dt.strftime("%Y%m%d").fillna("unknown")
    )
    pairs["diagnosis"] = _normalise_diagnosis(pairs["research_group"])
    pairs["diagnosis_source"] = "archive_folder_label_static"
    pairs["RID"] = pairs["subject_id"]
    pairs["MRI_path"] = pairs["mri_member"]
    pairs["PET_path"] = pairs["pet_member"]
    pairs["segmentation_path"] = pd.NA
    for column in ("age", "sex", "APOE", "MMSE"):
        pairs[column] = pd.NA

    columns = [
        "RID", "subject_id", "visit_id", "visit_date", "diagnosis", "diagnosis_source",
        "age", "sex", "APOE", "MMSE", "MRI_date", "PET_date", "MRI_path", "PET_path",
        "segmentation_path", "absolute_interval_days", "mri_pet_within_90d", "has_longitudinal",
    ]
    return pairs[columns].sort_values(["subject_id", "visit_date"]).reset_index(drop=True)


def load_optional_metadata(path: str | Path, pairing_window_days: int = 90) -> pd.DataFrame:
    """Load a conventional extracted ADNI metadata table when one is supplied."""

    table = pd.read_csv(path)
    aliases = {"RID": "subject_id", "MRI_date": "MRI_date", "PET_date": "PET_date"}
    if "subject_id" not in table and "RID" in table:
        table["subject_id"] = table["RID"].astype(str)
    required = {"subject_id", "visit_id", "diagnosis", "MRI_date", "PET_date"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Metadata table is missing columns: {missing}")
    del aliases
    table["diagnosis"] = _normalise_diagnosis(table["diagnosis"])
    table["MRI_date"] = pd.to_datetime(table["MRI_date"], errors="coerce")
    table["PET_date"] = pd.to_datetime(table["PET_date"], errors="coerce")
    table["visit_date"] = table[["MRI_date", "PET_date"]].min(axis=1)
    table["absolute_interval_days"] = (table["MRI_date"] - table["PET_date"]).abs().dt.days
    table["mri_pet_within_90d"] = table["absolute_interval_days"].le(pairing_window_days)
    has_both = table["MRI_date"].notna() & table["PET_date"].notna()
    paired_visit_counts = has_both.groupby(table["subject_id"]).transform("sum")
    table["has_longitudinal"] = paired_visit_counts.ge(2)
    table["diagnosis_source"] = "visit_level_metadata"
    return table.sort_values(["subject_id", "visit_date"]).reset_index(drop=True)


def build_participants_master(config: PipelineConfig) -> pd.DataFrame:
    """Build and save the best available participant master table."""

    optional = config.data_path("optional_metadata")
    if optional.exists():
        master = load_optional_metadata(optional, config.values["pairing_window_days"])
        source = str(optional)
    else:
        inventory = config.data_path("pair_inventory")
        master = build_master_from_pair_inventory(inventory, config.values["pairing_window_days"])
        source = str(inventory)
        nifti_inventory = config.data_path("nifti_inventory")
        if nifti_inventory.exists():
            nifti = pd.read_csv(nifti_inventory, usecols=["subject_id", "product_type", "acq_date", "member"])
            segmentations = nifti[nifti["product_type"] == "MRI_MUSE_segmentation"].copy()
            segmentations["MRI_date"] = pd.to_datetime(segmentations["acq_date"], errors="coerce")
            segmentations = segmentations.drop_duplicates(["subject_id", "MRI_date"])[
                ["subject_id", "MRI_date", "member"]
            ].rename(columns={"member": "segmentation_member"})
            master = master.merge(segmentations, on=["subject_id", "MRI_date"], how="left")
            master["segmentation_path"] = master["segmentation_member"].combine_first(master["segmentation_path"])
            master = master.drop(columns="segmentation_member")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(config.output_dir / "participants_master.csv", index=False)
    (config.output_dir / "participants_master_provenance.json").write_text(
        json.dumps(
            {
                "source": source,
                "rows": len(master),
                "participants": int(master["subject_id"].nunique()),
                "diagnosis_source_values": sorted(master["diagnosis_source"].dropna().unique()),
                "warning": (
                    "Static archive folder labels are not visit-level outcomes and must not be used "
                    "to derive MCI-to-AD conversion."
                ),
                "path_note": "MRI, PET, and segmentation paths are reproducible ZIP member paths, not extracted filesystem paths.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return master


def derive_mci_survival_cohort(master: pd.DataFrame) -> pd.DataFrame:
    """Derive MCI-to-AD event time from genuine visit-level diagnosis histories."""

    required = {"subject_id", "visit_date", "diagnosis", "diagnosis_source"}
    missing = sorted(required.difference(master.columns))
    if missing:
        raise OutcomeUnavailableError(f"Cannot derive survival outcomes; missing columns: {missing}")
    if set(master["diagnosis_source"].dropna()) != {"visit_level_metadata"}:
        raise OutcomeUnavailableError(
            "The available diagnoses are static archive folder labels rather than dated visit-level "
            "assessments. Conversion and censoring times cannot be derived validly."
        )

    visits = master.copy()
    visits["visit_date"] = pd.to_datetime(visits["visit_date"], errors="coerce")
    visits = visits.dropna(subset=["subject_id", "visit_date", "diagnosis"])
    rows: list[dict[str, object]] = []
    for subject_id, history in visits.groupby("subject_id", sort=True):
        history = history.sort_values("visit_date")
        baseline = history.iloc[0]
        if baseline["diagnosis"] != "MCI" or len(history) < 2:
            continue
        later = history.iloc[1:]
        conversions = later[later["diagnosis"] == "AD"]
        event = int(not conversions.empty)
        end_date = conversions.iloc[0]["visit_date"] if event else later.iloc[-1]["visit_date"]
        time_years = (end_date - baseline["visit_date"]).days / 365.25
        if time_years <= 0:
            continue
        row = baseline.to_dict()
        row.update(
            {
                "baseline_date": baseline["visit_date"],
                "outcome_date": end_date,
                "time_to_conversion": time_years,
                "event": event,
                "followup_visits": len(history),
            }
        )
        rows.append(row)
    cohort = pd.DataFrame(rows)
    if cohort.empty or int(cohort.get("event", pd.Series(dtype=int)).sum()) == 0:
        raise OutcomeUnavailableError("No valid MCI-to-AD events were found in dated visit-level histories.")
    return cohort


def save_survival_feasibility(config: PipelineConfig, master: pd.DataFrame) -> pd.DataFrame:
    """Save a cohort when possible, otherwise save a machine-readable blocked status."""

    output = config.output_dir
    try:
        cohort = derive_mci_survival_cohort(master)
        cohort.to_csv(output / "mci_survival_cohort.csv", index=False)
        status = {
            "status": "ready",
            "participants": len(cohort),
            "events": int(cohort["event"].sum()),
        }
    except OutcomeUnavailableError as error:
        cohort = pd.DataFrame(columns=["subject_id", "baseline_date", "outcome_date", "time_to_conversion", "event"])
        cohort.to_csv(output / "mci_survival_cohort.csv", index=False)
        status = {"status": "blocked", "reason": str(error), "fabricated_results": False}
    (output / "prognostic_feasibility.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return cohort


if __name__ == "__main__":
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    arguments = parser.parse_args()
    cfg = load_config(arguments.config)
    participant_master = build_participants_master(cfg)
    save_survival_feasibility(cfg, participant_master)
    print(f"Saved {len(participant_master)} master rows to {cfg.output_dir}")
