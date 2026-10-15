
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_DIR = Path("data")

FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

benign_parts = []
attack_parts = []

for name in FILES:
    path = next(DATA_DIR.rglob(name), None)
    if path is None:
        raise FileNotFoundError(name)

    print("Loading:", name)
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()

    labels = df["Label"].astype(str).str.strip().str.upper()
    X = df.drop(columns=["Label", "Flow ID", "Source IP",
                         "Destination IP", "Timestamp"],
                errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    benign = X[labels.eq("BENIGN")]
    attacks = X[~labels.eq("BENIGN")]

    if len(benign):
        benign_parts.append(
            benign.sample(n=min(12000, len(benign)), random_state=42)
        )
    if len(attacks):
        attack_parts.append(
            attacks.sample(n=min(3000, len(attacks)), random_state=42)
        )

benign = pd.concat(benign_parts, ignore_index=True)
attacks = pd.concat(attack_parts, ignore_index=True)

# Split benign data: training, threshold calibration, and testing.
rng = np.random.RandomState(42)
indices = rng.permutation(len(benign))
train_end = int(len(indices) * 0.6)
cal_end = int(len(indices) * 0.8)

train = benign.iloc[indices[:train_end]]
calibration = benign.iloc[indices[train_end:cal_end]]
benign_test = benign.iloc[indices[cal_end:]]

# Fixed test set: benign + attack flows.
test_X = pd.concat([benign_test, attacks], ignore_index=True)
test_y = np.concatenate([
    np.zeros(len(benign_test), dtype=int),
    np.ones(len(attacks), dtype=int),
])

scalers = {
    "No scaling": "passthrough",
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler(),
}

for name, scaler in scalers.items():
    print(f"\n{'=' * 50}\n{name}")

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        scaler,
        IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        ),
    )

    model.fit(train)

    # Calibrate a threshold using BENIGN validation data only.
    calibration_scores = model.decision_function(calibration)
    threshold = np.percentile(calibration_scores, 5)

    # Lower decision scores indicate more anomalous flows.
    test_scores = model.decision_function(test_X)
    predictions = (test_scores < threshold).astype(int)

    print("Threshold:", round(float(threshold), 6))
    print("Attack precision:", round(
        precision_score(test_y, predictions, zero_division=0), 4
    ))
    print("Attack recall:", round(
        recall_score(test_y, predictions, zero_division=0), 4
    ))
    print("Attack F1:", round(
        f1_score(test_y, predictions, zero_division=0), 4
    ))
    print("Benign false-positive rate:", round(
        predictions[:len(benign_test)].mean(), 4
    ))
