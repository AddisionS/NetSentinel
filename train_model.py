
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

frames = []

for name in FILES:
    path = next(DATA_DIR.rglob(name), None)
    if path is None:
        raise FileNotFoundError(f"Missing dataset file: {name}")

    print(f"Loading {name}...")
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()

    # Normalize labels into benign vs attack
    df["Label"] = (
        df["Label"].astype(str).str.strip().str.upper()
        .eq("BENIGN").map({True: 0, False: 1})
    )
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
del frames

print("\nCombined shape:", data.shape)
print("Labels:\n", data["Label"].value_counts())

# Remove duplicate rows
data = data.drop_duplicates()

# Separate labels from input features
y = data.pop("Label")

# Remove columns that are not useful model inputs
for col in ["Flow ID", "Source IP", "Destination IP", "Timestamp"]:
    if col in data.columns:
        data.drop(columns=col, inplace=True)

# Convert all features to numeric and replace invalid values
data = data.apply(pd.to_numeric, errors="coerce")
data = data.replace([np.inf, -np.inf], np.nan)

# Drop completely empty columns
data = data.dropna(axis=1, how="all")

# Stratified split keeps both classes represented in each set
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )),
])

print("\nTraining Random Forest...")
pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("F1 score:", round(f1_score(y_test, predictions), 4))
print("\nClassification report:")
print(classification_report(
    y_test, predictions,
    target_names=["BENIGN", "ATTACK"],
    zero_division=0,
))
print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

joblib.dump(
    {"pipeline": pipeline, "features": list(X_train.columns)},
    MODEL_DIR / "netsentinel.joblib",
)

print("\nSaved model to models/netsentinel.joblib")
