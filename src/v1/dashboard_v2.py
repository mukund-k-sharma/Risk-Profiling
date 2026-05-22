import json
import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from kafka import KafkaConsumer
import shap
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh


# --- Kafka Configuration ---
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
METRICS_TOPIC = "performance-metrics"
ALERTS_TOPIC = "alerts"
EXPLANATIONS_TOPIC = "explanations"

# --- State ---
metrics_history = []
roc_points = []
confusion_matrix_accum = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
shap_values_accum = []


# Keep a rolling history of explanations in Streamlit state
if "expl_history" not in st.session_state:
    st.session_state.expl_history = []  # list of {nameOrig, z_score, shap: dict}

EXPL_HISTORY_MAX = 500


# --- Streamlit Layout ---
# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="dashboard_refresh")
st.set_page_config(layout="wide")
st.title("Real-Time Risk Profiling Dashboard")


# --- Utility functions ---
@st.cache_resource
def consume_kafka(topic, timeout=2):
    """Fetch latest Kafka messages from given topic."""
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        consumer_timeout_ms=timeout * 1000,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    messages = [msg.value for msg in consumer]
    consumer.close()
    return messages


def plot_confusion_matrix(cm_dict):
    cm = np.array(
        [
            [cm_dict["tp"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tn"]],
        ]
    )
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fraud", "Not Fraud"])
    ax.set_yticklabels(["Fraud", "Not Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    st.pyplot(fig)


def plot_shap_summary(shap_vals, feature_names):
    if not shap_vals:
        st.warning("No SHAP values accumulated yet.")
        return
    df = pd.DataFrame(shap_vals)
    mean_abs_vals = df.abs().mean().sort_values(ascending=False).head(5)
    st.bar_chart(mean_abs_vals)


# --- Streamlit Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(
    [" Metrics", " ROC & Confusion Matrix", " Drift", " Explainability"]
)

# --- Tab 1: Metrics (Precision/Recall/F1, Latency, Throughput) ---
with tab1:
    st.subheader("Rolling Metrics")

    new_metrics = consume_kafka(METRICS_TOPIC)
    if new_metrics:
        metrics_history.extend(new_metrics)

    if metrics_history:
        df = pd.DataFrame(metrics_history)

        # Line chart for precision, recall, f1
        chart = (
            alt.Chart(df)
            .transform_fold(["precision", "recall", "f1_score"])
            .mark_line()
            .encode(
                x="timestamp:T",
                y="value:Q",
                color="key:N",
                tooltip=["batch_id", "precision", "recall", "f1_score"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

        # Latency plot (avg + p95)
        latency_df = df[["timestamp", "latency_ms_avg", "latency_ms_p95"]]
        latency_chart = (
            alt.Chart(latency_df)
            .transform_fold(["latency_ms_avg", "latency_ms_p95"])
            .mark_line()
            .encode(x="timestamp:T", y="value:Q", color="key:N")
        )
        st.altair_chart(latency_chart, use_container_width=True)

        # Throughput trend
        st.line_chart(df.set_index("timestamp")["throughput_tps"])

# --- Tab 2: ROC Curve & Confusion Matrix ---
with tab2:
    st.subheader("ROC Curve (Rolling) & Confusion Matrix")

    new_metrics = consume_kafka(METRICS_TOPIC)
    if new_metrics:
        for m in new_metrics:
            if "roc_curve" in m:
                roc_points.append(m["roc_curve"])
            confusion_matrix_accum["tp"] += m.get("true_positives", 0)
            confusion_matrix_accum["fp"] += m.get("false_positives", 0)
            confusion_matrix_accum["tn"] += m.get("true_negatives", 0)
            confusion_matrix_accum["fn"] += m.get("false_negatives", 0)

    if roc_points:
        latest = roc_points[-1]
        roc_df = pd.DataFrame({"fpr": latest["fpr"], "tpr": latest["tpr"]})
        roc_chart = (
            alt.Chart(roc_df)
            .mark_line()
            .encode(x="fpr", y="tpr")
            .properties(
                title=(
                    f"ROC Curve (AUC={latest['auc']:.2f})"
                    if latest.get("auc") is not None
                    else "ROC Curve (AUC=N/A)"
                )
            )
        )
        st.altair_chart(roc_chart, use_container_width=True)

    plot_confusion_matrix(confusion_matrix_accum)

# --- Tab 3: Drift Detection ---
with tab3:
    # st.subheader("Drift Detection Timeline")
    # df = pd.DataFrame(metrics_history)
    # if not df.empty and "drift_detected" in df.columns:
    #     drift_df = df[["timestamp", "drift_detected"]]
    #     drift_chart = (
    #         alt.Chart(drift_df)
    #         .mark_line()
    #         .encode(x="timestamp:T", y="drift_detected:Q")
    #         .properties(title="Drift Events Over Time")
    #     )
    #     st.altair_chart(drift_chart, use_container_width=True)
    st.subheader(" Drift Detection")

    st.markdown("### Raw Kafka Messages for Drift")
    for msg in metrics_history:
        if "drift_detected" in msg or "drift_score" in msg:
            st.json(msg)

    # Prepare chart only if drift info exists
    df = pd.DataFrame(metrics_history)
    if not df.empty and any(
        col in df.columns for col in ["drift_detected", "drift_score"]
    ):
        # Prefer drift_score if exists, fallback to drift_detected
        drift_col = "drift_score" if "drift_score" in df.columns else "drift_detected"
        drift_df = df[["timestamp", drift_col]].copy()
        drift_df[drift_col] = drift_df[drift_col].fillna(0)

        drift_chart = (
            alt.Chart(drift_df)
            .mark_line()
            .encode(
                x="timestamp:T",
                y=f"{drift_col}:Q",
                tooltip=["timestamp", drift_col],
            )
            .properties(title="Drift Events Over Time")
        )
        st.altair_chart(drift_chart, use_container_width=True)
    else:
        st.info(" No drift messages received yet.")


# --- Tab 4: Explainability ---
with tab4:
    st.subheader("SHAP Explanations")

    # Pull new explanation messages from Kafka and append to rolling history
    new_expls = consume_kafka(EXPLANATIONS_TOPIC)
    if new_expls:
        for e in new_expls:
            try:
                shap_dict = json.loads(e.get("shap_explanation", "{}"))
            except Exception:
                continue
            st.session_state.expl_history.append(
                {
                    "nameOrig": e.get("nameOrig", "unknown"),
                    "z_score": float(e.get("z_score", 0.0) or 0.0),
                    "shap": shap_dict,
                }
            )
        # Cap size
        if len(st.session_state.expl_history) > EXPL_HISTORY_MAX:
            st.session_state.expl_history = st.session_state.expl_history[
                -EXPL_HISTORY_MAX:
            ]

    hist = st.session_state.expl_history

    if not hist:
        st.info("No SHAP explanations received yet.")
    else:
        # Build a small meta table for the selector
        meta_df = pd.DataFrame(
            [
                {"idx": i, "nameOrig": r["nameOrig"], "z_score": r["z_score"]}
                for i, r in enumerate(hist)
            ]
        )
        # Selector defaults to the most recent alert
        default_index = len(meta_df) - 1
        selected_idx = st.selectbox(
            "Select alert to inspect",
            options=meta_df["idx"].tolist(),
            index=default_index,
            format_func=lambda i: f'{meta_df.loc[meta_df.idx==i, "nameOrig"].values[0]} (z={meta_df.loc[meta_df.idx==i, "z_score"].values[0]:.2f})',
        )

        rec = hist[selected_idx]
        shap_dict = rec["shap"]
        if not shap_dict:
            st.warning("Selected alert has empty SHAP explanation.")
        else:
            feature_names = list(shap_dict.keys())
            shap_vals = np.array([shap_dict[f] for f in feature_names], dtype=float)

            # Top contributors table (|SHAP| sorted)
            st.markdown("**Top contributing features (by |SHAP|)**")
            topk = (
                pd.Series(shap_dict)
                .abs()
                .sort_values(ascending=False)
                .head(10)
                .rename("abs_shap")
            )
            st.dataframe(topk.to_frame())

            # Per-alert waterfall plot (matplotlib)
            st.markdown("**Per-alert SHAP Waterfall**")
            try:
                expl = shap.Explanation(
                    values=shap_vals,
                    base_values=0.0,
                    data=np.zeros_like(shap_vals),
                    feature_names=feature_names,
                )
                fig_w = plt.figure()
                shap.plots.waterfall(expl, max_display=12, show=False)
                st.pyplot(fig_w)
            except Exception as ex:
                st.warning(f"Could not render waterfall plot: {ex}")

            st.markdown("**Beeswarm summary (last 100 alerts)**")
            N = min(100, len(hist))
            last = hist[-N:]

            matrix = np.array(
                [[float(r["shap"].get(f, 0.0)) for f in feature_names] for r in last],
                dtype=float,
            )

            try:
                expl_multi = shap.Explanation(
                    values=matrix,
                    base_values=np.zeros(matrix.shape[0]),
                    data=np.zeros_like(matrix),
                    feature_names=feature_names,
                )
                fig_b = plt.figure()
                shap.plots.beeswarm(expl_multi, max_display=12, show=False)
                st.pyplot(fig_b)
            except Exception as ex:
                st.warning(f"Could not render beeswarm plot: {ex}")

            # Aggregate bar (mean |SHAP| over last N)
            st.markdown("**Aggregate top features (mean |SHAP| over last 100 alerts)**")
            mean_abs = (
                pd.DataFrame(matrix, columns=feature_names)
                .abs()
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            st.bar_chart(mean_abs)
