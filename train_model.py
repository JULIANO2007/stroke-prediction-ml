from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import shutil
import tempfile

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "healthcare-dataset-stroke-data.csv"
NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "work_Govt_job",
    "work_Never_worked",
    "work_Private",
    "work_Self-employed",
    "work_children",
    "smoking_Unknown",
    "smoking_formerly smoked",
    "smoking_never smoked",
    "smoking_smokes",
]
ARTIFACT_NAMES = [
    "median_imputer_stroke.pkl",
    "scaler_stroke.pkl",
    "feature_columns_stroke.pkl",
    "random_forest_model.pkl",
]


def convert_numeric_column(values: pd.Series, column: str) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    cleaned = cleaned.str.replace(
        r"^([+-]?\d+(?:[.,]\d+)?)\.00$", r"\1", regex=True
    )
    cleaned = cleaned.str.replace(",", ".", regex=False)
    converted = pd.to_numeric(cleaned, errors="coerce")
    invalid = cleaned.notna() & converted.isna()
    if invalid.any():
        examples = cleaned[invalid].head(5).tolist()
        raise ValueError(f"Nilai {column} tidak dapat dibaca: {examples}")
    return converted


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, sep=";")
    required_columns = {
        "id",
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Kolom dataset tidak ditemukan: {sorted(missing_columns)}")

    for column in NUMERIC_FEATURES:
        data[column] = convert_numeric_column(data[column], column)

    data["stroke"] = pd.to_numeric(data["stroke"], errors="raise").astype(int)
    target_values = set(data["stroke"].unique())
    if target_values != {0, 1}:
        raise ValueError(f"Target stroke harus 0/1, ditemukan: {target_values}")

    category_mappings = {
        "gender": {"Female": 0, "Male": 1, "Other": 2},
        "ever_married": {"No": 0, "Yes": 1},
        "Residence_type": {"Rural": 0, "Urban": 1},
    }
    for column, mapping in category_mappings.items():
        mapped = data[column].map(mapping)
        if mapped.isna().any():
            unknown = data.loc[mapped.isna(), column].unique().tolist()
            raise ValueError(f"Kategori {column} tidak dikenal: {unknown}")
        data[column] = mapped.astype(int)

    category_values = {
        "work_type": {
            "Govt_job",
            "Never_worked",
            "Private",
            "Self-employed",
            "children",
        },
        "smoking_status": {
            "Unknown",
            "formerly smoked",
            "never smoked",
            "smokes",
        },
    }
    for column, allowed_values in category_values.items():
        unknown = set(data[column].dropna().unique()).difference(allowed_values)
        if unknown:
            raise ValueError(f"Kategori {column} tidak dikenal: {sorted(unknown)}")

    data = pd.get_dummies(
        data,
        columns=["work_type", "smoking_status"],
        prefix=["work", "smoking"],
        dtype=int,
    )
    data = data.drop(columns=["id"])
    features = data.drop(columns=["stroke"]).reindex(
        columns=FEATURE_COLUMNS, fill_value=0
    )
    target = data["stroke"]
    non_numeric_features = features.drop(columns=NUMERIC_FEATURES)
    if non_numeric_features.isna().any().any():
        raise ValueError("Ada nilai kosong di fitur nonnumerik.")
    return features, target


def save_artifacts(
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: list[str],
    model: RandomForestClassifier,
    sample_input: pd.DataFrame,
) -> None:
    with tempfile.TemporaryDirectory(prefix="stroke_artifacts_", dir=PROJECT_DIR) as temp_dir:
        staging_dir = Path(temp_dir)
        artifacts = {
            ARTIFACT_NAMES[0]: imputer,
            ARTIFACT_NAMES[1]: scaler,
            ARTIFACT_NAMES[2]: feature_columns,
            ARTIFACT_NAMES[3]: model,
        }
        for filename, artifact in artifacts.items():
            joblib.dump(artifact, staging_dir / filename)

        loaded_imputer = joblib.load(staging_dir / ARTIFACT_NAMES[0])
        loaded_scaler = joblib.load(staging_dir / ARTIFACT_NAMES[1])
        loaded_columns = joblib.load(staging_dir / ARTIFACT_NAMES[2])
        loaded_model = joblib.load(staging_dir / ARTIFACT_NAMES[3])
        sample_numeric = sample_input[NUMERIC_FEATURES].copy()
        sample_numeric = pd.DataFrame(
            loaded_imputer.transform(sample_numeric),
            columns=NUMERIC_FEATURES,
            index=sample_input.index,
        )
        sample_numeric = pd.DataFrame(
            loaded_scaler.transform(sample_numeric),
            columns=NUMERIC_FEATURES,
            index=sample_input.index,
        )
        checked_input = sample_input.copy()
        checked_input[NUMERIC_FEATURES] = sample_numeric
        if loaded_columns != FEATURE_COLUMNS:
            raise ValueError("Urutan kolom artifact tidak sesuai schema training.")
        if list(loaded_model.feature_names_in_) != loaded_columns:
            raise ValueError("Fitur model tidak cocok dengan feature_columns artifact.")
        loaded_model.predict(checked_input)
        loaded_model.predict_proba(checked_input)

        existing = [name for name in ARTIFACT_NAMES if (PROJECT_DIR / name).exists()]
        if existing:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = PROJECT_DIR / f"artifact_backup_{timestamp}"
            backup_dir.mkdir()
            for filename in existing:
                shutil.copy2(PROJECT_DIR / filename, backup_dir / filename)
            print(f"Artifact lama disalin ke: {backup_dir.name}")

        for filename in ARTIFACT_NAMES:
            (staging_dir / filename).replace(PROJECT_DIR / filename)


def main() -> int:
    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train[NUMERIC_FEATURES])
    X_train_processed[NUMERIC_FEATURES] = imputer.transform(
        X_train[NUMERIC_FEATURES]
    )
    X_test_processed[NUMERIC_FEATURES] = imputer.transform(
        X_test[NUMERIC_FEATURES]
    )

    scaler = StandardScaler()
    scaler.fit(X_train_processed[NUMERIC_FEATURES])
    X_train_processed[NUMERIC_FEATURES] = scaler.transform(
        X_train_processed[NUMERIC_FEATURES]
    )
    X_test_processed[NUMERIC_FEATURES] = scaler.transform(
        X_test_processed[NUMERIC_FEATURES]
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_processed, y_train)

    sample_input = pd.DataFrame(
        [
            {
                "gender": 0,
                "age": 24,
                "hypertension": 0,
                "heart_disease": 0,
                "ever_married": 1,
                "Residence_type": 1,
                "avg_glucose_level": 89.0,
                "bmi": 22.5,
                "work_Govt_job": 0,
                "work_Never_worked": 0,
                "work_Private": 1,
                "work_Self-employed": 0,
                "work_children": 0,
                "smoking_Unknown": 0,
                "smoking_formerly smoked": 0,
                "smoking_never smoked": 1,
                "smoking_smokes": 0,
            }
        ]
    ).reindex(columns=FEATURE_COLUMNS, fill_value=0)

    save_artifacts(imputer, scaler, FEATURE_COLUMNS, model, sample_input)

    print(f"Dataset: {len(X)} rows; train: {len(X_train)}; test: {len(X_test)}")
    print(f"Target counts: {y.value_counts().sort_index().to_dict()} (0=No Stroke, 1=Stroke)")
    print(f"Feature count/order validated: {len(FEATURE_COLUMNS)} columns")
    print(f"Imputer medians: {imputer.statistics_.tolist()}")
    print(f"Scaler means: {scaler.mean_.tolist()}")
    print(f"Scaler scales: {scaler.scale_.tolist()}")
    print(f"Model classes: {model.classes_.tolist()}")
    print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test_processed)):.2%}")
    print("Semua artifact baru lulus uji load, schema, predict, dan predict_proba.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())