
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

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

# Reuse the feature schema from the supervised model.
saved = joblib.load(MODEL_DIR / "netsentinel.joblib")
features = saved["features"]

parts = []

for name in FILES:
    path = next(DATA_DIR.rglob(name), None)
    if path is None:
        raise FileNotFoundError(f"Dataset file not found: {name}")

    print("Loading:", name)
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()

    labels = (
        df["Label"].astype(str).str.strip().str.upper()
        .ne("BENIGN").astype(int)
    )

    X = df.reindex(columns=features)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    sample = X.copy()
    sample["_target"] = labels.to_numpy()

    # Sample each class separately.
    benign = sample[sample["_target"] == 0]
    attacks = sample[sample["_target"] == 1]

    sample = pd.concat([
        benign.sample(
            n=min(len(benign), 12000),
            random_state=42,
        ),
        attacks.sample(
            n=min(len(attacks), 5000),
            random_state=42,
        ),
    ], ignore_index=True)

    parts.append(sample)

data = pd.concat(parts, ignore_index=True).drop_duplicates()

X = data[features]
y = data["_target"].to_numpy()

# 60% training, 20% validation, 20% testing.
X_remaining, X_test, y_remaining, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

X_trainval, X_val, y_trainval, y_val = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.25,
    stratify=y_remaining,
    random_state=43,
)

# Train ONLY on benign traffic.
X_train = X_trainval[y_trainval == 0]

if len(X_train) == 0:
    raise ValueError("No benign training samples available.")

model = make_pipeline(
    SimpleImputer(strategy="median"),
    IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    ),
)

print("\nBenign training flows:", len(X_train))
print("Validation flows:", len(X_val))
print("Test flows:", len(X_test))

print("\nTraining Isolation Forest...")
model.fit(X_train)

# Lower decision scores mean more anomalous.
val_scores = model.decision_function(X_val)
test_scores = model.decision_function(X_test)

# Select a threshold using validation data only.
thresholds = np.unique(
    np.quantile(val_scores, np.linspace(0.01, 0.30, 60))
)

best = None

print("\nEvaluating thresholds on validation data...")

for threshold in thresholds:
    predictions = (val_scores < threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )
    false_positive_rate = predictions[y_val == 0].mean()

    if best is None or f1 > best["f1"]:
        best = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_positive_rate": float(false_positive_rate),
        }

print("\nBest validation result:")
for key, value in best.items():
    print(f"{key}: {value:.4f}")

# Evaluate once on the held-out test set.
test_predictions = (test_scores < best["threshold"]).astype(int)

print("\n========== FINAL TEST RESULTS ==========")
print(
    "Precision:",
    round(precision_score(y_test, test_predictions, zero_division=0), 4),
)
print(
    "Recall:",
    round(recall_score(y_test, test_predictions, zero_division=0), 4),
)
print(
    "F1:",
    round(f1_score(y_test, test_predictions, zero_division=0), 4),
)

benign_mask = y_test == 0
false_positive_rate = test_predictions[benign_mask].mean()

print("Benign false-positive rate:", round(float(false_positive_rate), 4))
print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))

# Save the model and the validation-selected threshold.
joblib.dump(
    {
        "pipeline": model,
        "features": features,
        "threshold": best["threshold"],
    },
    MODEL_DIR / "netsentinel_anomaly_tuned.joblib",
)

print("\nSaved models/netsentinel_anomaly_tuned.joblib")
