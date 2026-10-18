"""Penalised Cox prognosis with nested validation and censoring-aware metrics."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .data_loading import OutcomeUnavailableError
from .regions import feature_columns, region_feature_map


def _fit_cox(
    x: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    penalizer: float,
) -> tuple[CoxPHFitter, SimpleImputer, StandardScaler, list[str]]:
    """Fit preprocessing on training data and a ridge-penalised Cox model."""

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    transformed = scaler.fit_transform(imputer.fit_transform(x))
    names = [f"x_{index:03d}" for index in range(transformed.shape[1])]
    frame = pd.DataFrame(transformed, columns=names)
    frame["duration"] = durations
    frame["event"] = events
    model = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, duration_col="duration", event_col="event", show_progress=False)
    return model, imputer, scaler, names


def _transform(x: np.ndarray, imputer: SimpleImputer, scaler: StandardScaler, names: list[str]) -> pd.DataFrame:
    """Apply training-derived preprocessing and return Cox feature columns."""

    return pd.DataFrame(scaler.transform(imputer.transform(x)), columns=names)


def _select_penalizer(
    x: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    candidates: list[float],
    folds: int,
    seed: int,
) -> float:
    """Select ridge strength by inner-fold validation concordance."""

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = {candidate: [] for candidate in candidates}
    for train, validation in splitter.split(x, events):
        for candidate in candidates:
            try:
                model, imputer, scaler, names = _fit_cox(
                    x[train], durations[train], events[train], candidate
                )
                risk = model.predict_partial_hazard(
                    _transform(x[validation], imputer, scaler, names)
                ).to_numpy()
                score = concordance_index(durations[validation], -risk, events[validation])
                scores[candidate].append(float(score))
            except Exception:
                scores[candidate].append(np.nan)
    means = {candidate: np.nanmean(values) for candidate, values in scores.items()}
    if all(np.isnan(value) for value in means.values()):
        raise RuntimeError("All inner Cox fits failed; reduce features or increase the event count.")
    return float(max(means, key=lambda candidate: np.nan_to_num(means[candidate], nan=-np.inf)))


def _censor_survival(training_time: np.ndarray, training_event: np.ndarray) -> KaplanMeierFitter:
    """Fit the censoring distribution on an outer training fold."""

    return KaplanMeierFitter().fit(training_time, event_observed=1 - training_event)


def _g(km: KaplanMeierFitter, time: float) -> float:
    """Return a numerically safe censoring-survival estimate."""

    return max(float(km.predict(max(time - 1e-8, 0.0))), 1e-6)


def ipcw_auc(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    horizon: float,
    censoring_km: KaplanMeierFitter,
) -> float:
    """Compute cumulative/dynamic AUC with inverse censoring weights."""

    cases = np.flatnonzero((event == 1) & (time <= horizon))
    controls = np.flatnonzero(time > horizon)
    if not len(cases) or not len(controls):
        return np.nan
    numerator = 0.0
    denominator = 0.0
    control_weight = 1.0 / _g(censoring_km, horizon)
    for case in cases:
        case_weight = 1.0 / _g(censoring_km, float(time[case]))
        for control in controls:
            weight = case_weight * control_weight
            numerator += weight * (float(risk[case] > risk[control]) + 0.5 * float(risk[case] == risk[control]))
            denominator += weight
    return numerator / denominator if denominator else np.nan


def ipcw_brier(
    time: np.ndarray,
    event: np.ndarray,
    predicted_survival: np.ndarray,
    horizon: float,
    censoring_km: KaplanMeierFitter,
) -> float:
    """Compute an inverse-censoring-weighted Brier score at one horizon."""

    total = 0.0
    for subject_time, subject_event, survival in zip(time, event, predicted_survival):
        if subject_time <= horizon and subject_event == 1:
            weight = 1.0 / _g(censoring_km, float(subject_time))
            outcome = 0.0
        elif subject_time > horizon:
            weight = 1.0 / _g(censoring_km, horizon)
            outcome = 1.0
        else:
            weight = 0.0
            outcome = 0.0
        total += weight * (outcome - survival) ** 2
    return total / len(time)


def _feature_sets(cohort: pd.DataFrame, selected_regions: list[str]) -> dict[str, list[str]]:
    """Return available baseline and optional longitudinal Cox feature panels."""

    mapping = region_feature_map()
    selected = [feature for region in selected_regions for feature in mapping[region]]
    panels = {
        "MRI_only_baseline": feature_columns("MRI"),
        "PET_only_baseline": feature_columns("PET"),
        "MRI_PET_baseline_raw": feature_columns(),
        "MRI_PET_baseline_GNN_selected": selected,
    }
    slopes = sorted(column for column in cohort if column.endswith("_annualised_change"))
    if slopes:
        panels["MRI_PET_baseline_longitudinal"] = feature_columns() + slopes
    return {name: columns for name, columns in panels.items() if set(columns).issubset(cohort.columns)}


def run_prognostic_models(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Run Cox models when valid outcomes exist, otherwise fail transparently."""

    output = config.output_dir
    feasibility_path = output / "prognostic_feasibility.json"
    if feasibility_path.exists():
        feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
        if feasibility.get("status") == "blocked":
            raise OutcomeUnavailableError(str(feasibility["reason"]))
    cohort_path = output / "mci_survival_cohort.csv"
    if not cohort_path.exists():
        raise OutcomeUnavailableError("mci_survival_cohort.csv has not been created.")
    cohort = pd.read_csv(cohort_path)
    minimum_events = int(config.values["survival"]["minimum_events"])
    if cohort.empty or "event" not in cohort or int(cohort["event"].sum()) < minimum_events:
        raise OutcomeUnavailableError(
            f"Penalised Cox modelling requires at least {minimum_events} genuine dated conversion events; "
            f"the available table contains {int(cohort['event'].sum()) if 'event' in cohort else 0}."
        )
    selected = pd.read_csv(output / "gnn_selected_regions.csv")["region"].tolist()
    panels = _feature_sets(cohort, selected)
    if not panels:
        raise OutcomeUnavailableError("The survival cohort does not contain matched baseline regional features.")

    durations = cohort["time_to_conversion"].to_numpy(dtype=float)
    events = cohort["event"].to_numpy(dtype=int)
    cv = config.values["cross_validation"]
    horizons = np.asarray(config.values["survival"]["horizons_years"], dtype=float)
    splitter = RepeatedStratifiedKFold(
        n_splits=int(cv["outer_folds"]),
        n_repeats=int(cv["outer_repeats"]),
        random_state=config.seed,
    )
    metric_rows = []
    calibration_rows = []
    selection_rows = []
    for model_name, columns in panels.items():
        x = cohort[columns].to_numpy(dtype=float)
        for split_index, (train, test) in enumerate(splitter.split(x, events)):
            repeat = split_index // int(cv["outer_folds"]) + 1
            fold = split_index % int(cv["outer_folds"]) + 1
            penalizer = _select_penalizer(
                x[train],
                durations[train],
                events[train],
                [float(value) for value in cv["cox_penalizer"]],
                int(cv["inner_folds"]),
                config.seed + split_index,
            )
            model, imputer, scaler, names = _fit_cox(
                x[train], durations[train], events[train], penalizer
            )
            test_frame = _transform(x[test], imputer, scaler, names)
            risk = model.predict_partial_hazard(test_frame).to_numpy()
            survival = model.predict_survival_function(test_frame, times=horizons).T.to_numpy()
            censoring = _censor_survival(durations[train], events[train])
            c_index = concordance_index(durations[test], -risk, events[test])
            metric_rows.append(
                {"model": model_name, "repeat": repeat, "fold": fold, "metric": "c_index", "value": c_index}
            )
            brier_values = []
            for horizon_index, horizon in enumerate(horizons):
                auc = ipcw_auc(durations[test], events[test], risk, horizon, censoring)
                brier = ipcw_brier(
                    durations[test], events[test], survival[:, horizon_index], horizon, censoring
                )
                brier_values.append(brier)
                metric_rows.extend(
                    [
                        {"model": model_name, "repeat": repeat, "fold": fold, "metric": f"auc_{horizon:g}y", "value": auc},
                        {"model": model_name, "repeat": repeat, "fold": fold, "metric": f"brier_{horizon:g}y", "value": brier},
                    ]
                )
                quartiles = pd.qcut(risk, q=min(4, len(np.unique(risk))), labels=False, duplicates="drop")
                for quartile in sorted(pd.Series(quartiles).dropna().unique()):
                    mask = quartiles == quartile
                    km = KaplanMeierFitter().fit(durations[test][mask], event_observed=events[test][mask])
                    calibration_rows.append(
                        {
                            "model": model_name,
                            "repeat": repeat,
                            "fold": fold,
                            "horizon": horizon,
                            "risk_quartile": int(quartile) + 1,
                            "predicted_survival": float(survival[mask, horizon_index].mean()),
                            "observed_survival": float(km.predict(horizon)),
                        }
                    )
            integrated = float(np.trapezoid(brier_values, horizons) / (horizons[-1] - horizons[0]))
            metric_rows.append(
                {"model": model_name, "repeat": repeat, "fold": fold, "metric": "integrated_brier_score", "value": integrated}
            )
            selection_rows.append(
                {"model": model_name, "repeat": repeat, "fold": fold, "penalizer": penalizer, "features": len(columns)}
            )

    fold_metrics = pd.DataFrame(metric_rows)
    summary = (
        fold_metrics.groupby(["model", "metric"])["value"]
        .agg(mean="mean", ci_lower=lambda x: np.nanpercentile(x, 2.5), ci_upper=lambda x: np.nanpercentile(x, 97.5))
        .reset_index()
    )
    calibration = pd.DataFrame(calibration_rows)
    summary.to_csv(output / "prognostic_results_summary.csv", index=False)
    calibration.to_csv(output / "prognostic_calibration_plots_data.csv", index=False)
    fold_metrics.to_csv(output / "prognostic_fold_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "prognostic_model_selection.csv", index=False)
    (output / "prognostic_model_status.json").write_text(
        json.dumps({"status": "completed", "participants": len(cohort), "events": int(events.sum())}, indent=2),
        encoding="utf-8",
    )
    return {"results": summary, "calibration": calibration, "fold_metrics": fold_metrics}


def save_blocked_prognostic_outputs(config: PipelineConfig, reason: str) -> None:
    """Save empty schema-correct outputs and an explicit scientific validity note."""

    output = config.output_dir
    pd.DataFrame(columns=["model", "metric", "mean", "ci_lower", "ci_upper"]).to_csv(
        output / "prognostic_results_summary.csv", index=False
    )
    pd.DataFrame(
        columns=["model", "horizon", "risk_quartile", "predicted_survival", "observed_survival"]
    ).to_csv(output / "prognostic_calibration_plots_data.csv", index=False)
    (output / "prognostic_model_status.json").write_text(
        json.dumps({"status": "blocked", "reason": reason, "fabricated_results": False}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    arguments = parser.parse_args()
    cfg = load_config(arguments.config)
    try:
        run_prognostic_models(cfg)
    except OutcomeUnavailableError as error:
        save_blocked_prognostic_outputs(cfg, str(error))
        print(f"Prognostic analysis blocked safely: {error}")
