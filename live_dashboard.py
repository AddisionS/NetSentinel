
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

LOG_FILE = Path("live_flows.jsonl")

st.set_page_config(
    page_title="NetSentinel Live",
    page_icon="🛡️",
    layout="wide",
)

st_autorefresh(interval=5000, key="live_refresh")

st.title("NetSentinel | Live Operations")
st.caption("Real-time network telemetry • Heuristic threat monitoring")

if not LOG_FILE.exists():
    st.warning("No flow log found. Start live_engine.py first.")
    st.stop()

records = []
with LOG_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

if not records:
    st.info("Waiting for traffic...")
    st.stop()

df = pd.DataFrame(records)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["alerts"] = df["alerts"].apply(
    lambda x: x if isinstance(x, list) else []
)
df["alert_count"] = df["alerts"].apply(len)

alerts = df[df["alert_count"] > 0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flow windows", len(df))
c2.metric("Packets observed", f"{df['packets'].sum():,.0f}")
c3.metric("Bytes observed", f"{df['bytes'].sum():,.0f}")
c4.metric("Alert windows", len(alerts))

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Packets per flow window")
    st.line_chart(
        df.dropna(subset=["timestamp"])
          .sort_values("timestamp")
          .set_index("timestamp")["packets"]
    )

with right:
    st.subheader("Protocol distribution")
    st.bar_chart(df["protocol"].value_counts())

st.subheader("Alert history")
if alerts.empty:
    st.success("No heuristic alerts recorded.")
else:
    st.dataframe(
        alerts[["timestamp", "src_ip", "dst_ip", "protocol",
                "packets", "bytes", "alerts"]]
        .sort_values("timestamp", ascending=False),
        use_container_width=True,
    )

st.subheader("Recent traffic")
st.dataframe(
    df.sort_values("timestamp", ascending=False).head(100),
    use_container_width=True,
)

st.caption(
    "Alerts are heuristic indicators, not confirmed attacks. "
    "The dashboard displays recorded flow windows, not every packet."
)
