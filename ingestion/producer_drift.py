import json
import logging
import os
import random
import uuid
from time import sleep, time
import pandas as pd
from kafka import KafkaProducer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC_NAME = os.getenv("TRANSACTIONS_TOPIC", "transactions")
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "/home/n00b/workspace/Risk-Profiling/data/PS_20174392719_1491204439457_log.csv")
TARGET_USER = os.getenv("TARGET_USER", "C351297720")

NORMAL_TRANSACTIONS_COUNT = 55
DRIFT_TRANSACTIONS_COUNT = 55
DRIFT_MAGNITUDE_MIN = 5.0
DRIFT_MAGNITUDE_MAX = 35.0
STREAM_SPEED_SEC = float(os.getenv("STREAM_SPEED_SEC", "1.0"))

def create_producer():
    """Create a Kafka producer."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,
            request_timeout_ms=30000,
        )
        logger.info(f"Kafka producer created successfully. Bootstrapping with: {KAFKA_BOOTSTRAP_SERVERS}")
        return producer
    except Exception as e:
        logger.error(f"Failed to create Kafka producer: {e}")
        raise e

def generate_normal_transaction(base_transaction, step):
    """Generates a 'normal' transaction by slightly varying the base transaction."""
    message = base_transaction.copy()
    variation_factor = random.uniform(0.9, 1.1)
    new_amount = max(0.0, round(message["amount"] * variation_factor, 2))

    balance_diff = new_amount - message["amount"]
    message["amount"] = new_amount
    message["newbalanceOrig"] = max(0.0, round(message["newbalanceOrig"] - balance_diff, 2))
    message["newbalanceDest"] = max(0.0, round(message["newbalanceDest"] + balance_diff, 2))
    message["step"] = step
    message["producer_timestamp_ms"] = int(time() * 1000)
    message["transactionId"] = str(uuid.uuid4())

    return message

def generate_drift_transaction(base_transaction, step):
    """Generates a 'drift' transaction by significantly increasing the amount."""
    message = base_transaction.copy()
    original_amount = message["amount"]
    random_magnitude = random.uniform(DRIFT_MAGNITUDE_MIN, DRIFT_MAGNITUDE_MAX)
    drifted_amount = max(0.0, round(original_amount * random_magnitude, 2))

    balance_diff = drifted_amount - original_amount
    message["amount"] = drifted_amount
    message["newbalanceOrig"] = max(0.0, round(message["newbalanceOrig"] - balance_diff, 2))
    message["newbalanceDest"] = max(0.0, round(message["newbalanceDest"] + balance_diff, 2))
    message["step"] = step
    message["producer_timestamp_ms"] = int(time() * 1000)
    message["transactionId"] = str(uuid.uuid4())

    return message, original_amount

def run_simulation():
    """Runs the normal and drifted transaction simulation."""
    producer = None
    try:
        logger.info(f"Loading dataset from: {DATA_FILE_PATH}")
        df = pd.read_csv(DATA_FILE_PATH)

        user_transactions = df[df["nameOrig"] == TARGET_USER]
        if user_transactions.empty:
            logger.error(f"Could not find any transactions for user '{TARGET_USER}'. Aborting simulation.")
            return

        base_transaction = user_transactions.iloc[0].to_dict()
        # Ensure NaNs are cleaned
        for k, v in base_transaction.items():
            if pd.isna(v):
                base_transaction[k] = None

        logger.info(f"Found a base transaction for user '{TARGET_USER}' to use as a template.")
        producer = create_producer()
        current_step = int(base_transaction.get("step", 1))

        # --- Phase 1: Send Normal Transactions ---
        logger.info(f"--- Phase 1: Sending {NORMAL_TRANSACTIONS_COUNT} NORMAL transactions for user {TARGET_USER} ---")
        for i in range(NORMAL_TRANSACTIONS_COUNT):
            message = generate_normal_transaction(base_transaction, current_step)
            producer.send(TOPIC_NAME, value=message)
            logger.info(f"Sent NORMAL transaction {i+1}/{NORMAL_TRANSACTIONS_COUNT} | Type: {message['type']}, Amount: {message['amount']:.2f}")
            current_step += 1
            if STREAM_SPEED_SEC > 0:
                sleep(STREAM_SPEED_SEC)
        producer.flush()

        # --- Phase 2: Introduce and Send Drifted Transactions ---
        logger.info(f"--- Phase 2: DRIFT! Sending {DRIFT_TRANSACTIONS_COUNT} drifted transactions for user {TARGET_USER} ---")
        for i in range(DRIFT_TRANSACTIONS_COUNT):
            message, original_amount = generate_drift_transaction(base_transaction, current_step)
            producer.send(TOPIC_NAME, value=message)
            logger.info(f"Sent DRIFTED transaction {i+1}/{DRIFT_TRANSACTIONS_COUNT} | Type: {message['type']}, Amount: {message['amount']:.2f} (Original: {original_amount:.2f})")
            current_step += 1
            if STREAM_SPEED_SEC > 0:
                sleep(STREAM_SPEED_SEC)
        producer.flush()

        logger.info("Drift simulation completed successfully.")

    except FileNotFoundError:
        logger.error(f"FATAL: The data file was not found at '{DATA_FILE_PATH}'. Please check the path.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the simulation: {e}")
    finally:
        if producer:
            producer.close()
            logger.info("Kafka producer closed.")

if __name__ == "__main__":
    run_simulation()
