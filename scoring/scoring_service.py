import json
import logging
import os
import time
from collections import deque
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from river.drift import ADWIN
from sklearn.metrics import roc_curve, auc, roc_auc_score
from kafka import KafkaConsumer, KafkaProducer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
CONSUME_TOPIC = os.getenv("ERRORS_TOPIC", "reconstruction-errors")
ALERTS_TOPIC = os.getenv("ALERTS_TOPIC", "alerts")
EXPLANATIONS_TOPIC = os.getenv("EXPLANATIONS_TOPIC", "explanations")
METRICS_TOPIC = os.getenv("METRICS_TOPIC", "performance-metrics")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/autoencoder_model.pth")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "/app/preprocessor.joblib")
GLOBAL_STATS_PATH = os.getenv("GLOBAL_STATS_PATH", "/app/global_stats.json")
TRAIN_SAMPLE_PATH = os.getenv("TRAIN_SAMPLE_PATH", "/app/X_train_sample.csv")

MIN_HISTORY_FOR_Z_SCORE = 5
ALERT_THRESHOLD = 2.5
METRICS_PUBLISH_INTERVAL_EVENTS = 20 # Publish metrics every 20 transactions
MAX_STATE_HISTORY = 100

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

# --- PyTorch Autoencoder definition ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim=11):
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

# --- Global objects & state ---
user_states = {} # user_id -> {"history": deque, "adwin": ADWIN}
global_stats = {"mean": 0.03, "std": 0.015} # Fallback defaults
model = None
preprocessor = None
background_scaled = None
explainer = None

# Metrics tracking state
processed_records = []
batch_counter = 0
last_publish_time = time.time()

def load_resources():
    global global_stats, model, preprocessor, background_scaled, explainer
    # Load Global Stats
    try:
        with open(GLOBAL_STATS_PATH, "r") as f:
            global_stats = json.load(f)
        logger.info(f"Loaded global stats baseline: {global_stats}")
    except Exception as e:
        logger.warning(f"Could not load global stats from {GLOBAL_STATS_PATH}: {e}. Using default fallback.")

    # Load PyTorch model for SHAP
    try:
        model = Autoencoder(input_dim=11)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        logger.info("Loaded Autoencoder weights for SHAP successfully.")
    except Exception as e:
        logger.error(f"Failed to load Autoencoder weights for SHAP: {e}")

    # Load Preprocessor for SHAP
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info("Loaded StandardScaler preprocessor for SHAP successfully.")
    except Exception as e:
        logger.error(f"Failed to load StandardScaler preprocessor for SHAP: {e}")

    # Load Background CSV for SHAP KernelExplainer
    try:
        bg_data = pd.read_csv(TRAIN_SAMPLE_PATH).head(50)
        background_scaled = preprocessor.transform(bg_data)
        logger.info(f"Loaded {len(bg_data)} background samples from {TRAIN_SAMPLE_PATH} for SHAP.")
        
        # Build SHAP KernelExplainer
        def predict_error(data_np):
            data_tensor = torch.FloatTensor(data_np)
            with torch.no_grad():
                reconstructions = model(data_tensor)
                return torch.mean((data_tensor - reconstructions) ** 2, axis=1).numpy()
                
        explainer = shap.KernelExplainer(predict_error, background_scaled)
        logger.info("Initialized SHAP KernelExplainer successfully.")
    except Exception as e:
        logger.warning(f"Failed to build SHAP KernelExplainer: {e}. SHAP explanations will fall back to empty values.")

def create_consumer_producer():
    try:
        consumer = KafkaConsumer(
            CONSUME_TOPIC,
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            auto_offset_reset="latest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="scoring-service-group",
        )
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        logger.info("Kafka consumer and producer initialized successfully.")
        return consumer, producer
    except Exception as e:
        logger.error(f"Failed to initialize Kafka consumer/producer: {e}")
        raise e

def generate_shap_explanation(scaled_features):
    """Generate local SHAP explanations for an alert."""
    if explainer is None or preprocessor is None:
        return "{}"
    try:
        # Explain scaled features directly
        instance_scaled = np.array([scaled_features])
        shap_values = explainer.shap_values(instance_scaled)
        # Map values to feature names
        expl_dict = {feat: float(shap_values[0, j]) for j, feat in enumerate(FEATURE_COLUMNS)}
        return json.dumps(expl_dict)
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {e}")
        return "{}"

def update_and_publish_metrics(producer):
    global processed_records, batch_counter, last_publish_time
    if not processed_records:
        return

    current_time = time.time()
    time_elapsed = max(current_time - last_publish_time, 0.001)
    throughput = len(processed_records) / time_elapsed
    
    # Calculate Latencies
    latencies = [r["latency_ms"] for r in processed_records]
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    # Evaluate rolling classification metrics
    tp, fp, tn, fn = 0, 0, 0, 0
    y_true, y_pred = [], []
    for r in processed_records:
        is_fraud = int(r["isFraud"])
        is_alert = int(r["is_alert"])
        y_true.append(is_fraud)
        y_pred.append(is_alert)
        
        if is_fraud == 1 and is_alert == 1:
            tp += 1
        elif is_fraud == 0 and is_alert == 1:
            fp += 1
        elif is_fraud == 0 and is_alert == 0:
            tn += 1
        elif is_fraud == 1 and is_alert == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Calculate ROC and AUC
    if len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_val = float(roc_auc_score(y_true, y_pred))
    else:
        fpr, tpr, auc_val = [0.0, 1.0], [0.0, 1.0], None

    # Track concept drift flag in this batch
    drift_detected = 1 if any(r.get("drift_detected", 0) == 1 for r in processed_records) else 0

    batch_counter += 1
    metrics_payload = {
        "batch_id": batch_counter,
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
        "drift_detected": drift_detected,
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(y) for y in tpr],
            "auc": auc_val,
        },
    }

    try:
        producer.send(METRICS_TOPIC, value=metrics_payload)
        producer.flush()
        logger.info(f"Published Performance Metrics | Batch: {batch_counter} | F1: {f1_score:.4f} | Latency: {p95_latency:.1f}ms (p95)")
    except Exception as e:
        logger.error(f"Failed to publish metrics to Kafka: {e}")

    # Reset metrics tracking state
    processed_records = []
    last_publish_time = current_time

def process_stream():
    global user_states, processed_records
    load_resources()
    consumer, producer = create_consumer_producer()

    logger.info(f"Risk Scoring & Alerting microservice started. Listening on '{CONSUME_TOPIC}'...")

    for message in consumer:
        payload = message.value
        try:
            user_id = payload["nameOrig"]
            error = payload["reconstruction_error"]
            prod_ts = payload.get("producer_timestamp_ms")
            
            # Create user state if not exists
            if user_id not in user_states:
                user_states[user_id] = {
                    "history": deque(maxlen=MAX_STATE_HISTORY),
                    "adwin": ADWIN()
                }
                
            state = user_states[user_id]
            history = state["history"]
            adwin = state["adwin"]
            
            drift_detected = 0
            
            # 1. Update ADWIN drift detector
            adwin.update(error)
            if adwin.drift_detected:
                drift_detected = 1
                logger.warning(f"Concept drift detected dynamically for user: {user_id}. Resetting behavioral baseline.")
                history.clear()
                state["adwin"] = ADWIN() # Re-initialize ADWIN
            
            # 2. Dynamic rolling Z-score calculation
            if len(history) < MIN_HISTORY_FOR_Z_SCORE:
                mean_error = global_stats["mean"]
                std_error = global_stats["std"]
            else:
                mean_error = np.mean(history)
                std_error = np.std(history)
                
            z_score = (error - mean_error) / std_error if std_error > 0 else 0.0
            is_alert = 1 if z_score > ALERT_THRESHOLD else 0
            
            # Update user reconstruction error history
            history.append(error)
            
            # Append alerts data to current payload
            payload["z_score"] = z_score
            payload["is_alert"] = is_alert
            payload["drift_detected"] = drift_detected
            payload["processing_timestamp_ms"] = int(time.time() * 1000)
            
            # 3. Publish alerts if triggered
            if is_alert == 1:
                producer.send(ALERTS_TOPIC, value=payload)
                logger.warning(f"🚨 ALERT TRIGGERED: High behavioral anomaly for user {user_id}! (Z-Score: {z_score:.2f})")
                
                # 4. Generate SHAP explainability
                scaled_features = payload["scaled_features"]
                shap_explanation = generate_shap_explanation(scaled_features)
                
                explanation_payload = {
                    "nameOrig": user_id,
                    "z_score": z_score,
                    "shap_explanation": shap_explanation
                }
                producer.send(EXPLANATIONS_TOPIC, value=explanation_payload)
                logger.info(f"Published SHAP local explainability details for alert on: {user_id}")
            
            # Track operational metrics
            latency_ms = payload["processing_timestamp_ms"] - prod_ts if prod_ts else 0
            processed_records.append({
                "isFraud": payload["isFraud"],
                "is_alert": is_alert,
                "latency_ms": latency_ms,
                "drift_detected": drift_detected
            })
            
            if len(processed_records) >= METRICS_PUBLISH_INTERVAL_EVENTS:
                update_and_publish_metrics(producer)
                
        except Exception as e:
            logger.error(f"Error processing transaction in risk scoring: {e}")

if __name__ == "__main__":
    process_stream()
