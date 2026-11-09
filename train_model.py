# train_model.py
# Part 1: retrieve the UCI Student Performance data (Mathematics file only)
# Part 2: preprocess, train, evaluate and save a Random Forest model
import io
import os
import zipfile
import urllib.request

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------
# PART 1 - Retrieve and inspect the dataset (Activity A)
# ---------------------------------------------------------------
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
MATH_FILE = "student-mat.csv"


def read_math_csv(zf):
    """Find student-mat.csv in a zip (or in a zip nested inside it)."""
    for name in zf.namelist():
        if name.endswith(MATH_FILE):
            return pd.read_csv(zf.open(name), sep=";")
    for name in zf.namelist():
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(zf.read(name))) as inner:
                return read_math_csv(inner)
    raise FileNotFoundError(MATH_FILE + " not found in the downloaded archive")


if os.path.exists(MATH_FILE):
    # Optional offline fallback: put student-mat.csv next to this script
    df = pd.read_csv(MATH_FILE, sep=";")
else:
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    df = read_math_csv(archive)

# Remove duplicates BEFORE saving so the app and the model use the same records
df = df.drop_duplicates()
df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())

# ---------------------------------------------------------------
# PART 2 - Preprocess, train and evaluate (Activity B)
# ---------------------------------------------------------------
# Target: 1 = at risk (G3 < 10); 0 = not at risk
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor
X = df.drop(columns=["G3"])

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
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
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

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")
