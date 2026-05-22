import logging
from collections import deque
import os
import json
import pickle
from typing import Iterator, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from river.drift import ADWIN
from sklearn.metrics import roc_curve, auc, roc_auc_score

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_json,
    pandas_udf,
    to_json,
    struct,
    when,
    current_timestamp,
    max as spark_max,
    min as spark_min,
)
from pyspark.sql.streaming.state import GroupState
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    ArrayType,
    LongType,
    TimestampType,
    BinaryType,
)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Application Configuration ---
MODEL_PATH = "/home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/autoencoder_model.pth"
PREPROCESSOR_PATH = "/home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/preprocessor.joblib"
GLOBAL_STATS_PATH = (
    "/home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/global_stats.json"
)
TRAIN_SAMPLE_PATH = (
    "/home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/X_train_sample.csv"
)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = "transactions"
ALERTS_TOPIC = "alerts"
EXPLANATIONS_TOPIC = "explanations"
METRICS_TOPIC = "performance-metrics"
CHECKPOINT_LOCATION_BASE = "/tmp/spark_checkpoints_v5"
MIN_HISTORY_FOR_Z_SCORE = 5
ALERT_THRESHOLD = 2.5

# --- Feature columns for the PaySim dataset ---
FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@pandas_udf(DoubleType())
def get_reconstruction_error_udf(*cols: pd.Series) -> pd.Series:
    model = Autoencoder(input_dim=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X = pd.concat(cols, axis=1)
    X.columns = FEATURE_COLUMNS
    X_scaled = preprocessor.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        reconstructions = model(X_tensor)
        mse = torch.mean((X_tensor - reconstructions) ** 2, axis=1)
    return pd.Series(mse.numpy())


def build_dynamic_risk_scorer(global_stats_bcast):
    """
    Factory function for the stateful UDF, now with ADWIN drift detection.
    """
    output_schema = StructType(
        [
            StructField("nameOrig", StringType()),
            StructField("timestamp", TimestampType()),
            StructField("reconstruction_error", DoubleType()),
            StructField("z_score", DoubleType()),
            StructField("isFraud", IntegerType()),
            StructField("is_alert", IntegerType()),
            StructField("drift_detected", IntegerType()),  # NEW: Flag for drift
            StructField("producer_timestamp_ms", LongType()),
        ]
    )

    state_schema = StructType(
        [
            StructField("history", ArrayType(DoubleType())),
            StructField("adwin_detector", BinaryType()),
        ]
    )

    def calculate_dynamic_risk_score(
        key: Tuple[str], pdf_iterator: Iterator[pd.DataFrame], state: GroupState
    ) -> Iterator[pd.DataFrame]:
        global_stats = global_stats_bcast.value

        if state.exists:
            history_list, adwin_pickle = state.get
            history = deque(history_list, maxlen=100)
            adwin = pickle.loads(adwin_pickle)
        else:
            history = deque(maxlen=100)
            adwin = ADWIN()

        output_rows = []
        for pdf in pdf_iterator:
            for _, row in pdf.iterrows():
                error = row.reconstruction_error
                drift_detected = 0

                adwin.update(error)
                if adwin.drift_detected:
                    drift_detected = 1
                    logger.info(
                        f"Concept drift detected for user {row.nameOrig}. Resetting state."
                    )
                    # Reset the state by clearing the history
                    history.clear()
                    # re-initialize ADWIN to clear its internal state
                    adwin = ADWIN()

                if len(history) < MIN_HISTORY_FOR_Z_SCORE:
                    mean_error, std_dev_error = (
                        global_stats["mean"],
                        global_stats["std"],
                    )
                else:
                    mean_error, std_dev_error = np.mean(history), np.std(history)

                z_score = (
                    (error - mean_error) / std_dev_error if std_dev_error > 0 else 0.0
                )
                is_alert = 1 if z_score > ALERT_THRESHOLD else 0
                history.append(error)

                output_rows.append(
                    {
                        "nameOrig": row.nameOrig,
                        "timestamp": row.timestamp,
                        "reconstruction_error": error,
                        "z_score": z_score,
                        "isFraud": row.isFraud,
                        "is_alert": is_alert,
                        "drift_detected": drift_detected,
                        "producer_timestamp_ms": row.producer_timestamp_ms,
                    }
                )

        state.update((list(history), pickle.dumps(adwin)))

        if output_rows:
            yield pd.DataFrame(output_rows)

    return calculate_dynamic_risk_score, output_schema, state_schema


def build_shap_explainer_udf():
    @pandas_udf(StringType())
    def generate_explanation_udf(*cols: pd.Series) -> pd.Series:
        model = Autoencoder(input_dim=len(FEATURE_COLUMNS))
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        try:
            background_data = pd.read_csv(TRAIN_SAMPLE_PATH).head(50)
            background_scaled = preprocessor.transform(background_data)
        except FileNotFoundError:
            logger.warning(
                f"{TRAIN_SAMPLE_PATH} not found. Using zero-background for SHAP."
            )
            background_scaled = np.zeros((1, len(FEATURE_COLUMNS)))

        def predict_error(data_np):
            data_tensor = torch.FloatTensor(data_np)
            with torch.no_grad():
                reconstructions = model(data_tensor)
                return torch.mean((data_tensor - reconstructions) ** 2, axis=1).numpy()

        explainer = shap.KernelExplainer(predict_error, background_scaled)
        instance_to_explain = pd.concat(cols, axis=1)
        instance_to_explain.columns = FEATURE_COLUMNS
        instance_scaled = preprocessor.transform(instance_to_explain)
        shap_values = explainer.shap_values(instance_scaled)
        explanations = [
            json.dumps(
                {feat: shap_values[i, j] for j, feat in enumerate(FEATURE_COLUMNS)}
            )
            for i in range(len(instance_to_explain))
        ]
        return pd.Series(explanations)

    return generate_explanation_udf


def update_performance_metrics(batch_df, batch_id):
    if batch_df.count() == 0:
        return
    spark = batch_df.sparkSession
    stats = batch_df.agg(
        spark_max("processing_timestamp_ms").alias("max_proc_ts"),
        spark_min("producer_timestamp_ms").alias("min_prod_ts"),
    ).first()
    if stats.max_proc_ts is None or stats.min_prod_ts is None:
        logger.warning(
            f"Skipping metrics for batch {batch_id} due to missing timestamps."
        )
        return

    processing_time_sec = max((stats.max_proc_ts - stats.min_prod_ts) / 1000.0, 0.001)
    throughput = batch_df.count() / processing_time_sec
    latencies = (
        batch_df.withColumn(
            "latency", col("processing_timestamp_ms") - col("producer_timestamp_ms")
        )
        .select("latency")
        .toPandas()["latency"]
    )

    avg_latency = float(latencies.mean() if not latencies.empty else 0.0)
    p95_latency = float(latencies.quantile(0.95) if not latencies.empty else 0.0)

    tp = batch_df.filter("isFraud = 1 AND is_alert = 1").count()
    fp = batch_df.filter("isFraud = 0 AND is_alert = 1").count()
    tn = batch_df.filter("isFraud = 0 AND is_alert = 0").count()
    fn = batch_df.filter("isFraud = 1 AND is_alert = 0").count()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 - score
    f1_score = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    # ROC Curve and AUC
    y_true = batch_df.select("isFraud").toPandas()["isFraud"]
    y_pred = batch_df.select("is_alert").toPandas()["is_alert"]

    if len(set(y_true)) > 1:  # Only compute if both classes present
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = float(roc_auc_score(y_true, y_pred))
    else:
        fpr, tpr, auc = [0.0, 1.0], [0.0, 1.0], None

    metrics_data = [
        {
            "batch_id": batch_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "latency_ms_avg": avg_latency,
            "latency_ms_p95": p95_latency,
            "throughput_tps": throughput,
            "roc_curve": {
                "fpr": [float(x) for x in fpr],
                "tpr": [float(y) for y in tpr],
                "auc": auc,
            },
        }
    ]
    metrics_df = spark.createDataFrame(metrics_data)
    metrics_df.select(to_json(struct("*")).alias("value")).write.format("kafka").option(
        "kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS
    ).option("topic", METRICS_TOPIC).save()
    logger.info(f"Published metrics for batch {batch_id}")


def main():
    spark = (
        SparkSession.builder.appName("RealTimeRiskProfilingV4")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        with open(GLOBAL_STATS_PATH, "r") as f:
            global_stats = json.load(f)
        global_stats_bcast = spark.sparkContext.broadcast(global_stats)
        logger.info(f"Loaded and broadcasted global stats: {global_stats}")
    except FileNotFoundError:
        logger.error(
            f"{GLOBAL_STATS_PATH} not found! Please ensure it exists and the path is correct."
        )
        return
    kafka_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", TRANSACTIONS_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )
    schema = StructType(
        [
            StructField("step", IntegerType()),
            StructField("type", StringType()),
            StructField("amount", DoubleType()),
            StructField("nameOrig", StringType()),
            StructField("oldbalanceOrg", DoubleType()),
            StructField("newbalanceOrig", DoubleType()),
            StructField("nameDest", StringType()),
            StructField("oldbalanceDest", DoubleType()),
            StructField("newbalanceDest", DoubleType()),
            StructField("isFraud", IntegerType()),
            StructField("isFlaggedFraud", IntegerType()),
            StructField("producer_timestamp_ms", LongType()),
        ]
    )

    parsed_df = kafka_df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")

    type_dummies = [
        when(col("type") == t, 1).otherwise(0).alias(f"type_{t}")
        for t in ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    ]
    processed_df = parsed_df.withColumn(
        "timestamp", col("step").cast(TimestampType())
    ).select("*", *type_dummies)

    errors_df = processed_df.withColumn(
        "reconstruction_error", get_reconstruction_error_udf(*FEATURE_COLUMNS)
    )

    scorer_udf, scorer_output_schema, scorer_state_schema = build_dynamic_risk_scorer(
        global_stats_bcast
    )

    scores_df = (
        errors_df.withWatermark("timestamp", "10 minutes")
        .groupBy("nameOrig")
        .applyInPandasWithState(
            func=scorer_udf,
            outputStructType=scorer_output_schema,
            stateStructType=scorer_state_schema,
            timeoutConf="ProcessingTimeTimeout",
            outputMode="append",
        )
    )

    scores_with_ts_df = scores_df.withColumn(
        "processing_timestamp_ms", (current_timestamp().cast("long") * 1000)
    )

    alerts_df = scores_with_ts_df.filter(col("is_alert") == 1)
    alerts_df.select(to_json(struct("*")).alias("value")).writeStream.format(
        "kafka"
    ).option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS).option(
        "topic", ALERTS_TOPIC
    ).option(
        "checkpointLocation", os.path.join(CHECKPOINT_LOCATION_BASE, "alerts")
    ).start()

    alerts_with_features = alerts_df.join(
        processed_df, ["nameOrig", "timestamp"], "inner"
    )

    explanations_df = alerts_with_features.withColumn(
        "shap_explanation", build_shap_explainer_udf()(*FEATURE_COLUMNS)
    )

    # --- TEMPORARY MODIFIED CODE ---
    # To get low-z-score explanations, we can filter the scores_with_ts_df
    # explanations_for_low_score = scores_with_ts_df.filter("is_alert == 0").limit(5)

    # alerts_with_features = explanations_for_low_score.join(
    #     processed_df, ["nameOrig", "timestamp"], "inner"
    # )

    # explanations_df = alerts_with_features.withColumn(
    #     "shap_explanation", build_shap_explainer_udf()(*FEATURE_COLUMNS)
    # )
    # ----------------------------------------------

    explanations_df.select("nameOrig", "z_score", "shap_explanation").select(
        to_json(struct("*")).alias("value")
    ).writeStream.format("kafka").option(
        "kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS
    ).option(
        "topic", EXPLANATIONS_TOPIC
    ).option(
        "checkpointLocation", os.path.join(CHECKPOINT_LOCATION_BASE, "explanations")
    ).start()

    scores_with_ts_df.writeStream.foreachBatch(update_performance_metrics).outputMode(
        "append"
    ).option(
        "checkpointLocation", os.path.join(CHECKPOINT_LOCATION_BASE, "metrics")
    ).start()

    logger.info("All streaming queries started.")

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
