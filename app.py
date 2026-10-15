
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NetSentinel",
    page_icon="🛡️",
    layout="wide",
)

MODEL_PATH = Path("models/netsentinel.joblib")
ANOMALY_PATH = Path("models/netsentinel_anomaly_tuned.joblib")
LIVE_LOG = Path("live_flows.jsonl")
ALERT_LOG = Path("alerts.jsonl")

st.title("🛡️ NetSentinel")
st.caption("Network Traffic Intelligence & Intrusion Detection")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

@st.cache_data
def load_sample():
    files = list(Path("data").rglob("*.csv"))
    if not files:
        return pd.DataFrame()

    df = pd.read_csv(
        files[0], encoding="latin1", low_memory=False
    )
    df.columns = df.columns.str.strip()
    return df


def read_jsonl(path):
    """Read JSON Lines safely, ignoring incomplete records."""
    if not path.exists():
        return pd.DataFrame()

    records = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    records.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        return pd.DataFrame()

    return pd.DataFrame(records)


def prepare_features(df, features):
    X = df.reindex(columns=features).copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------
# LOAD SUPERVISED MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python train_model.py")
    st.stop()

saved = joblib.load(MODEL_PATH)
model = saved["pipeline"]
features = saved["features"]

data = load_sample()

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

overview, predict, anomaly, live, history = st.tabs([
    "Overview",
    "Prediction Lab",
    "Anomaly Detection",
    "Live Monitoring",
    "Alert History",
])

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

with overview:
    st.subheader("Dataset Overview")

    if not data.empty and "Label" in data.columns:
        labels = data["Label"].astype(str).str.strip()
        benign_count = int(labels.str.upper().eq("BENIGN").sum())
        attack_types = labels[
            ~labels.str.upper().eq("BENIGN")
        ].nunique()

        c1, c2, c3 = st.columns(3)
        c1.metric("Sample records", f"{len(data):,}")
        c2.metric("Model features", len(features))
        c3.metric("Attack types", attack_types)

        counts = (
            labels.value_counts()
            .rename_axis("Label")
            .reset_index(name="Records")
        )

        st.plotly_chart(
            px.bar(
                counts,
                x="Label",
                y="Records",
                color="Label",
                title="Traffic Label Distribution",
            ),
            use_container_width=True,
        )
    else:
        st.warning("No sample dataset found in data/.")

    live_df = read_jsonl(LIVE_LOG)
    alert_df = read_jsonl(ALERT_LOG)

    c1, c2 = st.columns(2)
    c1.metric("Recorded live flow windows", len(live_df))
    c2.metric("Recorded alert events", len(alert_df))

    st.info(
        "Historical ML predictions, anomaly detection, and live "
        "traffic heuristics are separate analysis paths."
    )

# --------------------------------------------------
# SUPERVISED PREDICTION LAB
# --------------------------------------------------

with predict:
    st.subheader("Analyze Network Flows")

    st.write(
        "Upload a CIC-IDS2017-compatible CSV containing the "
        "numeric features used to train the Random Forest model."
    )

    upload = st.file_uploader(
        "Upload network flow CSV",
        type=["csv"],
        key="supervised_upload",
    )

    if upload is not None:
        try:
            incoming = pd.read_csv(
                upload, encoding="latin1", low_memory=False
            )
            incoming.columns = incoming.columns.str.strip()

            missing = [f for f in features if f not in incoming.columns]

            if missing:
                st.error(
                    f"CSV is missing {len(missing)} required features. "
                    f"Examples: {missing[:8]}"
                )
            else:
                X = prepare_features(incoming, features)
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1]

                results = incoming.copy()
                results["Prediction"] = np.where(
                    predictions == 1, "ATTACK", "BENIGN"
                )
                results["Attack probability"] = probabilities

                attacks = int((predictions == 1).sum())

                c1, c2, c3 = st.columns(3)
                c1.metric("Flows analyzed", f"{len(results):,}")
                c2.metric("Predicted attacks", f"{attacks:,}")
                c3.metric(
                    "Predicted benign",
                    f"{len(results) - attacks:,}",
                )

                st.plotly_chart(
                    px.histogram(
                        results,
                        x="Prediction",
                        color="Prediction",
                        title="Predicted Traffic Classes",
                    ),
                    use_container_width=True,
                )

                st.dataframe(
                    results.head(500),
                    use_container_width=True,
                )

                st.download_button(
                    "Download predictions CSV",
                    results.to_csv(index=False).encode("utf-8"),
                    file_name="netsentinel_predictions.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"Could not analyze this CSV: {exc}")

# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------

with anomaly:
    st.subheader("Isolation Forest Anomaly Detection")

    if not ANOMALY_PATH.exists():
        st.warning(
            "Tuned anomaly model not found. Run tune_anomaly.py first."
        )
    else:
        anomaly_saved = joblib.load(ANOMALY_PATH)
        anomaly_model = anomaly_saved["pipeline"]
        anomaly_features = anomaly_saved["features"]
        threshold = anomaly_saved["threshold"]

        st.caption(
            "This model was trained on benign historical flows. "
            "Lower decision scores indicate greater anomaly."
        )

        st.metric("Configured threshold", f"{threshold:.5f}")

        anomaly_upload = st.file_uploader(
            "Upload compatible flow CSV",
            type=["csv"],
            key="anomaly_upload",
        )

        if anomaly_upload is not None:
            try:
                incoming = pd.read_csv(
                    anomaly_upload,
                    encoding="latin1",
                    low_memory=False,
                )
                incoming.columns = incoming.columns.str.strip()

                missing = [
                    f for f in anomaly_features
                    if f not in incoming.columns
                ]

                if missing:
                    st.error(
                        f"Missing {len(missing)} required features. "
                        f"Examples: {missing[:8]}"
                    )
                else:
                    X = prepare_features(incoming, anomaly_features)
                    scores = anomaly_model.decision_function(X)
                    is_anomaly = scores < threshold

                    results = incoming.copy()
                    results["Anomaly score"] = scores
                    results["Detection"] = np.where(
                        is_anomaly, "ANOMALY", "NORMAL"
                    )

                    c1, c2 = st.columns(2)
                    c1.metric("Flows analyzed", f"{len(results):,}")
                    c2.metric(
                        "Flagged anomalies",
                        f"{int(is_anomaly.sum()):,}",
                    )

                    st.dataframe(
                        results.head(500),
                        use_container_width=True,
                    )

                    st.download_button(
                        "Download anomaly results",
                        results.to_csv(index=False).encode("utf-8"),
                        file_name="netsentinel_anomalies.csv",
                        mime="text/csv",
                    )

            except Exception as exc:
                st.error(f"Anomaly analysis failed: {exc}")

# --------------------------------------------------
# LIVE MONITORING
# --------------------------------------------------

with live:
    st.subheader("Live Network Monitoring")
    st.caption(
        "Reads the local capture engine's JSONL logs. "
        "Start live_engine.py separately to capture traffic."
    )

    st_autorefresh(interval=5000, key="live_auto_refresh")

    live_df = read_jsonl(LIVE_LOG)

    if live_df.empty:
        st.warning(
            "No live flows recorded yet. Start live_engine.py "
            "and allow some traffic to pass."
        )
    else:
        if "timestamp" in live_df.columns:
            live_df["timestamp"] = pd.to_datetime(
                live_df["timestamp"], errors="coerce"
            )

        for column in ["packets", "bytes", "packets_per_sec",
                       "bytes_per_sec"]:
            if column in live_df.columns:
                live_df[column] = pd.to_numeric(
                    live_df[column], errors="coerce"
                ).fillna(0)

        if "alerts" not in live_df.columns:
            live_df["alerts"] = [[] for _ in range(len(live_df))]

        live_df["alerts"] = live_df["alerts"].apply(
            lambda value: value if isinstance(value, list) else []
        )
        live_df["has_alert"] = live_df["alerts"].apply(bool)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flow windows", f"{len(live_df):,}")
        c2.metric(
            "Packets recorded",
            f"{live_df['packets'].sum():,.0f}",
        )
        c3.metric(
            "Bytes recorded",
            f"{live_df['bytes'].sum():,.0f}",
        )
        c4.metric(
            "Flagged flow windows",
            f"{int(live_df['has_alert'].sum()):,}",
        )

        left, right = st.columns(2)

        with left:
            st.plotly_chart(
                px.line(
                    live_df.dropna(subset=["timestamp"]).sort_values(
                        "timestamp"
                    ),
                    x="timestamp",
                    y="packets_per_sec",
                    title="Packets per Second",
                ),
                use_container_width=True,
            )

        with right:
            if "protocol" in live_df.columns:
                protocol_counts = (
                    live_df["protocol"].value_counts()
                    .rename_axis("Protocol")
                    .reset_index(name="Windows")
                )
                st.plotly_chart(
                    px.bar(
                        protocol_counts,
                        x="Protocol",
                        y="Windows",
                        title="Protocol Distribution",
                    ),
                    use_container_width=True,
                )

        flagged = live_df[live_df["has_alert"]].copy()

        st.subheader("Suspicious flow windows")
        if flagged.empty:
            st.success("No heuristic alerts in recorded flow windows.")
        else:
            st.dataframe(
                flagged.sort_values("timestamp", ascending=False),
                use_container_width=True,
            )

        st.subheader("Recent traffic")
        st.dataframe(
            live_df.sort_values("timestamp", ascending=False).head(100),
            use_container_width=True,
        )

        st.download_button(
            "Download live flow log",
            live_df.to_csv(index=False).encode("utf-8"),
            file_name="netsentinel_live_flows.csv",
            mime="text/csv",
        )

    st.caption(
        "Live alerts are rule-based heuristics, not confirmed attacks "
        "or predictions from the trained CIC-IDS2017 models."
    )

# --------------------------------------------------
# ALERT HISTORY
# --------------------------------------------------

with history:
    st.subheader("Persistent Alert History")

    alert_df = read_jsonl(ALERT_LOG)

    if alert_df.empty:
        st.info(
            "No alerts recorded yet. The alert log is created "
            "when the live engine raises an alert."
        )
    else:
        if "timestamp" in alert_df.columns:
            alert_df["timestamp"] = pd.to_datetime(
                alert_df["timestamp"], errors="coerce"
            )

        c1, c2 = st.columns(2)
        c1.metric("Alert events", len(alert_df))

        if "alert" in alert_df.columns:
            c2.metric("Alert types", alert_df["alert"].nunique())

            counts = (
                alert_df["alert"].value_counts()
                .rename_axis("Alert")
                .reset_index(name="Events")
            )

            st.plotly_chart(
                px.bar(
                    counts,
                    x="Alert",
                    y="Events",
                    color="Alert",
                    title="Alert Types",
                ),
                use_container_width=True,
            )

        st.dataframe(
            alert_df.sort_values("timestamp", ascending=False),
            use_container_width=True,
        )

        st.download_button(
            "Download alert history CSV",
            alert_df.to_csv(index=False).encode("utf-8"),
            file_name="netsentinel_alert_history.csv",
            mime="text/csv",
        )

    st.caption(
        "Alert severity and reasons are generated by the current "
        "heuristic engine and have not been independently validated."
    )
