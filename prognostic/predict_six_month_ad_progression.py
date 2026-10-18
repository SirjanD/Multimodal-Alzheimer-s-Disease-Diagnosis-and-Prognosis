"""Apply the final research Cox model to index-MCI MRI feature rows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import joblib
import numpy as np
import pandas as pd

from evaluate_early_ad_all_available import build_feature_matrix


DEFAULT_MODEL = ROOT / "six_month_ad_progression_results" / "final_six_month_progression_model.joblib"
DEFAULT_TEMPLATE = ROOT / "six_month_ad_progression_results" / "six_month_prediction_input_template.csv"
DEFAULT_OUTPUT = ROOT / "six_month_ad_progression_results" / "new_participant_six_month_probabilities.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Load raw features and save cumulative progression probabilities."""

    args = parse_args()
    artifact = joblib.load(args.model)
    table = pd.read_csv(args.input)
    missing = sorted(set(artifact["required_raw_columns"]).difference(table.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")
    matrix, feature_names = build_feature_matrix(table, artifact["feature_set"])
    if feature_names != artifact["feature_names"]:
        raise ValueError("Input feature order does not match the fitted model.")

    transformed = artifact["scaler"].transform(artifact["imputer"].transform(matrix))
    frame = pd.DataFrame(transformed, columns=artifact["feature_columns"])
    survival = artifact["cox_model"].predict_survival_function(
        frame, times=artifact["horizon_days"]
    ).T.to_numpy()
    probabilities = np.maximum.accumulate(np.clip(1.0 - survival, 0.0, 1.0), axis=1)
    output = pd.DataFrame({"participant_id": table["participant_id"].astype(str)})
    for index, month in enumerate(artifact["horizon_months"]):
        output[f"probability_by_{month}_months_percent"] = 100.0 * probabilities[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} participant prediction(s) to {args.output}")


if __name__ == "__main__":
    main()
