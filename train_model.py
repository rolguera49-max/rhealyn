# train_model.py
# Retrieve the UCI Student Performance data (Mathematics file only),
# preprocess it, train a Random Forest, evaluate it and save the model.
# Run directly:  python train_model.py
# app.py also imports train() and calls it automatically if files are missing.
import io
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE = Path(__file__).parent
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
MATH_FILE = "student-mat.csv"
DATA_PATH = BASE / "student_data.csv"
MODEL_PATH = BASE / "student_model.pkl"


def _read_math_from_zip(zf):
    """Find student-mat.csv in a zip (or in a zip nested inside it)."""
    for name in zf.namelist():
        if name.endswith(MATH_FILE):
            return pd.read_csv(zf.open(name), sep=";")
    for name in zf.namelist():
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(zf.read(name))) as inner:
                return _read_math_from_zip(inner)
    raise FileNotFoundError(MATH_FILE + " not found in the downloaded archive")


def load_raw_data():
    """Use a local student-mat.csv if present, otherwise download from UCI."""
    local = BASE / MATH_FILE
    if local.exists():
        return pd.read_csv(local, sep=None, engine="python")  # auto-detect ; or ,
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    return _read_math_from_zip(archive)


def train():
    # ---------- Activity A: retrieve and inspect ----------
    df = load_raw_data()
    df = df.drop_duplicates()  # before saving, so app and model use same records
    df.to_csv(DATA_PATH, index=False)

    print("Rows and columns:", df.shape)
    print(df.head())
    print(df.dtypes)
    print("Missing values per column:")
    print(df.isnull().sum())

    # ---------- Activity B: preprocess, train, evaluate ----------
    y = (df["G3"] < 10).astype(int)          # 1 = at risk, 0 = not at risk
    X = df.drop(columns=["G3"])              # never use G3 as a predictor

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print(classification_report(
        y_test, predictions,
        target_names=["Not at risk", "At risk"],
        zero_division=0,
    ))

    joblib.dump(model, MODEL_PATH)
    print("Saved", MODEL_PATH.name)
    return model


if __name__ == "__main__":
    train()
