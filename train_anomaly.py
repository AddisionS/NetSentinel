
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Reuse the exact feature schema of our existing classifier
saved = joblib.load(MODEL_DIR / "netsentinel.joblib")
features = saved["features"]

files = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

benign_parts = []
test_parts = []

for name in files:
    path = next(DATA_DIR.rglob(name), None)
    if path is None:
        raise FileNotFoundError(name)

    print("Reading:", name)
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()

    labels = df["Label"].astype(str).str.strip().str.upper()
    X = df.reindex(columns=features)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Keep a bounded sample for manageable training
    benign = X[labels.eq("BENIGN")]
    attacks = X[~labels.eq("BENIGN")]

    if len(benign):
        benign_parts.append(benign.sample(
            n=min(20000, len(benign)), random_state=42
        ))

    if len(attacks):
        test_parts.append(
            (attacks.sample(n=min(5000, len(attacks)), random_state=42), 1)
        )

    # Sample benign flows for false-positive evaluation too
    if len(benign):
        test_parts.append(
            (benign.sample(n=min(5000, len(benign)), random_state=7), 0)
        )

train = pd.concat(benign_parts, ignore_index=True)
test_X = pd.concat(
    [part for part, label in test_parts], ignore_index=True
)
test_y = np.concatenate([
    np.full(len(part), label) for part, label in test_parts
])

print("\nBenign training flows:", len(train))
print("Evaluation flows:", len(test_X))

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("detector", IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )),
])

print("\nTraining Isolation Forest...")
model.fit(train)

# Isolation Forest returns 1 for normal, -1 for anomalous
predictions = (model.predict(test_X) == -1).astype(int)

print("\nClassification report:")
print(classification_report(
    test_y,
    predictions,
    labels=[0, 1],
    target_names=["BENIGN", "ATTACK"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(test_y, predictions, labels=[0, 1]))

joblib.dump(
    {"pipeline": model, "features": features},
    MODEL_DIR / "netsentinel_anomaly.joblib",
)

print("\nSaved models/netsentinel_anomaly.joblib")
