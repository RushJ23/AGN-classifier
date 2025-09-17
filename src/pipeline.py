"""Data preparation and modelling pipeline for AGN classification.

This module consolidates the exploratory work that originally lived in the
Jupyter notebooks into a repeatable workflow.  Running the module as a script
will load the SIMBAD cross-matched catalogues, perform the filtering steps
used in the publication, train baseline machine-learning models and export a
set of summary tables under :mod:`reports`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

# Columns that were treated as categorical in the analysis notebooks and should
# therefore not be forced into numeric types during preprocessing.
_CATEGORICAL_COLUMNS: set[str] = {
    "ERO_Name",
    "CTP_Classification",
    "CTP_Source_type",
    "main_type",
}

# SIMBAD source types that were considered Active Galactic Nuclei in the
# project.
_AGN_TYPES: set[str] = {
    "QSO",
    "Seyfert_1",
    "Seyfert_2",
    "BLLac",
    "Blazar",
    "RadioG",
    "AGN",
}


@dataclass
class ModelResult:
    """Container for the evaluation metrics of a trained classifier."""

    model_name: str
    accuracy: float
    precision: float
    recall: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def load_efeds_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the eFEDS × VLASS × SIMBAD cross-match used in the paper."""

    dataset_path = path or RAW_DATA_DIR / "eFEDS_VLASS_Simbad.xlsx"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "The eFEDS dataset was not found. Expected to locate it at"
            f" {dataset_path}."
        )
    return pd.read_excel(dataset_path)


def sanitise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace blank cells with nulls and cast numeric columns."""

    cleaned = df.replace("", np.nan)
    numeric_candidates = [
        col for col in cleaned.columns if col not in _CATEGORICAL_COLUMNS
    ]
    for column in numeric_candidates:
        try:
            cleaned[column] = pd.to_numeric(cleaned[column])
        except (ValueError, TypeError):
            # Columns such as identifiers occasionally contain non-numeric
            # placeholders; leave them untouched in that case.
            continue
    return cleaned


def filter_quality_flags(df: pd.DataFrame, minimum_quality: float = 2) -> pd.DataFrame:
    """Discard sources with low cross-match quality."""

    if "CTP_quality" not in df:
        return df
    return df[df["CTP_quality"] > minimum_quality]


def is_agn(
    classification: str | float, reference_count: float | int | float
) -> str | bool:
    """Replicate the SIMBAD-based labelling heuristic from the analysis notebooks."""

    if (
        isinstance(classification, str)
        and classification in _AGN_TYPES
        and reference_count >= 3
    ):
        return True
    if pd.isna(classification):
        return "Unknown"
    if reference_count < 3:
        return "Unknown"
    if isinstance(classification, str) and "Candidate" in classification:
        return "Unknown"
    return False


def label_agn_sources(df: pd.DataFrame) -> pd.DataFrame:
    labelled = df.copy()
    labelled["is_AGN"] = labelled.apply(
        lambda row: is_agn(row.get("main_type"), row.get("nbref")), axis=1
    )
    return labelled


def filter_gaia_measurements(
    df: pd.DataFrame, snr_threshold: float = 3.0
) -> pd.DataFrame:
    """Filter sources with inconsistent Gaia astrometry measurements."""

    filtered = df.copy()
    for value_col, error_col in (
        ("GaiaEDR3_parallax", "GaiaEDR3_parallax_error"),
        ("GaiaEDR3_pmra", "GaiaEDR3_pmra_error"),
        ("GaiaEDR3_pmdec", "GaiaEDR3_pmdec_error"),
    ):
        if value_col not in filtered or error_col not in filtered:
            continue
        error = filtered[error_col].replace(0, np.nan)
        ratio = (filtered[value_col] / error).abs()
        mask = ratio.le(snr_threshold) | filtered[value_col].isna() | error.isna()
        filtered = filtered[mask]
    return filtered


def encode_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the CTP classification flag as an ordinal feature."""

    mapping = {
        "SECURE GALACTIC": 0,
        "UNCERTAIN": 1,
        "UNCERTAIN ": 1,
        "SECURE EXTRAGALACTIC": 2,
    }
    encoded = df.copy()
    encoded["classification_flag"] = (
        encoded["CTP_Classification"].map(mapping).fillna(1).astype(int)
    )
    return encoded


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    derived = df.copy()
    if "ERO_ML_FLUX" in derived:
        positive_flux = derived["ERO_ML_FLUX"].replace({0: np.nan}).dropna()
        derived.loc[positive_flux.index, "log_FLUX"] = np.log10(positive_flux)
    return derived


def prepare_training_frame(
    df: pd.DataFrame, feature_columns: Sequence[str], label_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    subset = df[df[label_column].isin([True, False])]
    training_df = subset[list(feature_columns) + [label_column]].dropna()
    y = training_df[label_column].astype(int)
    X = training_df.drop(columns=[label_column])
    return X, y


def evaluate_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> ModelResult:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return ModelResult(
        model_name=model_name,
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions),
        recall=recall_score(y_test, predictions),
    )


def run_efeds_pipeline(
    test_size: float = 0.2, random_state: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, list[ModelResult]]:
    df = load_efeds_dataset()
    df = sanitise_columns(df)
    df = filter_quality_flags(df)
    df = label_agn_sources(df)
    df = filter_gaia_measurements(df)
    df = encode_classifications(df)
    df = add_derived_features(df)

    features = [
        "LS8_g",
        "LS8_r",
        "LS8_z",
        "W1",
        "W2",
        "W3",
        "W4",
        "log_FLUX",
        "classification_flag",
    ]

    X, y = prepare_training_frame(df, features, "is_AGN")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models = [
        RandomForestClassifier(n_estimators=500, random_state=random_state),
        HistGradientBoostingClassifier(random_state=random_state, early_stopping=True),
    ]

    results = [
        evaluate_model(
            model.__class__.__name__, model, X_train, X_test, y_train, y_test
        )
        for model in models
    ]

    feature_importances = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": models[0].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    summary = df[features + ["is_AGN"]].describe(include="all")
    return summary, feature_importances, results


def export_results(
    summary: pd.DataFrame,
    feature_importances: pd.DataFrame,
    results: Iterable[ModelResult],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_DIR / "efeds_feature_summary.csv")
    feature_importances.to_csv(
        REPORT_DIR / "efeds_feature_importances.csv", index=False
    )
    metrics = pd.DataFrame([result.to_dict() for result in results])
    metrics.to_csv(REPORT_DIR / "efeds_model_metrics.csv", index=False)


def main() -> None:
    summary, importances, results = run_efeds_pipeline()
    export_results(summary, importances, results)


if __name__ == "__main__":
    main()
