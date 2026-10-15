
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Reuse the feature schema from the existing model.
saved = joblib.load(MODEL_DIR / "netsentinel.joblib")
features = saved["features"]


def load_day(filename):
    path = next(DATA_DIR.rglob(filename), None)
    if path is None:
        raise FileNotFoundError(f"Dataset not found: {filename}")

    print("Loading:", filename)
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()

    labels = (
        df["Label"].astype(str).str.strip().str.upper()
        .ne("BENIGN").astype(int)
    )

    X = df.reindex(columns=features)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Reproducible samples to keep runtime manageable.
    data = X.copy()
    data["_target"] = labels.to_numpy()

    benign = data[data["_target"] == 0]
    attacks = data[data["_target"] == 1]

    sampled = pd.concat([
        benign.sample(
            n=min(len(benign), 30000), random_state=42
        ),
        attacks.sample(
            n=min(len(attacks), 15000), random_state=42
        ),
    ], ignore_index=True)

    return sampled[features], sampled["_target"].to_numpy()


# Monday: fit only on benign flows.
X_monday, y_monday = load_day("Monday-WorkingHours.pcap_ISCX.csv")
X_train = X_monday[y_monday == 0]

# Tuesday: calibrate the threshold using benign flows only.
X_tuesday, y_tuesday = load_day("Tuesday-WorkingHours.pcap_ISCX.csv")
X_calibration = X_tuesday[y_tuesday == 0]

# Wednesday: final held-out evaluation.
X_wednesday, y_wednesday = load_day(
    "Wednesday-workingHours.pcap_ISCX.csv"
)

print("\nTraining benign flows:", len(X_train))
print("Calibration benign flows:", len(X_calibration))
print("Held-out Wednesday flows:", len(X_wednesday))
print("Wednesday attack flows:", int(y_wednesday.sum()))

model = make_pipeline(
    SimpleImputer(strategy="median"),
    IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    ),
)

print("\nTraining on Monday...")
model.fit(X_train)

# Set threshold using Tuesday's benign traffic only.
calibration_scores = model.decision_function(X_calibration)
threshold = float(np.percentile(calibration_scores, 5))

print("Selected threshold:", round(threshold, 6))

# Evaluate once on Wednesday.
test_scores = model.decision_function(X_wednesday)
predictions = (test_scores < threshold).astype(int)

print("\n========== WEDNESDAY TEST ==========")
print(classification_report(
    y_wednesday,
    predictions,
    labels=[0, 1],
    target_names=["BENIGN", "ATTACK"],
    zero_division=0,
))
print("Confusion matrix:")
print(confusion_matrix(y_wednesday, predictions, labels=[0, 1]))

benign_mask = y_wednesday == 0
if benign_mask.any():
    print(
        "Benign false-positive rate:",
        round(float(predictions[benign_mask].mean()), 4),
    )

joblib.dump(
    {
        "pipeline": model,
        "features": features,
        "threshold": threshold,
        "training_day": "Monday",
        "calibration_day": "Tuesday",
        "test_day": "Wednesday",
    },
    MODEL_DIR / "netsentinel_anomaly_day_test.joblib",
)

print("\nSaved models/netsentinel_anomaly_day_test.joblib")
