import json
import logging
from time import sleep, time
import random
import uuid
import pandas as pd
from kafka import KafkaProducer

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Kafka server configuration
KAFKA_BOOTSTRAP_SERVERS = ["kafka:9092"]
# Use ["localhost:9092"] if running outside of Docker
TOPIC_NAME = "transactions"

# --- Data and Drift Simulation Parameters ---
# !!! IMPORTANT: Update this path to your local CSV file !!!
DATA_FILE_PATH = "/home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/PS_20174392719_1491204439457_log.csv"
# Define the specific user for the simulation
TARGET_USER = "C351297720"

NORMAL_TRANSACTIONS_COUNT = 55
DRIFT_TRANSACTIONS_COUNT = 55
# Define the range for the random drift magnitude
DRIFT_MAGNITUDE_MIN = 5.0
DRIFT_MAGNITUDE_MAX = 35.0


def create_producer():
    """Creates and returns a Kafka producer."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,
            request_timeout_ms=30000,
        )
        logger.info("Kafka producer created successfully.")
        return producer
    except Exception as e:
        logger.error("Failed to create Kafka producer: %s", e)
        raise


def generate_normal_transaction(base_transaction, step):
    """Generates a 'normal' transaction by slightly varying the base transaction."""
    message = base_transaction.copy()

    # Introduce a small, random variation to the amount (+/- 10%)
    variation_factor = random.uniform(0.9, 1.1)
    new_amount = round(message["amount"] * variation_factor, 2)

    # Update balances logically based on the new amount
    balance_diff = new_amount - message["amount"]
    message["amount"] = new_amount
    message["newbalanceOrig"] = round(message["newbalanceOrig"] - balance_diff, 2)
    message["newbalanceDest"] = round(message["newbalanceDest"] + balance_diff, 2)

    # Update simulation-specific metadata
    message["step"] = step
    message["producer_timestamp_ms"] = int(time() * 1000)
    message["transactionId"] = str(uuid.uuid4())

    return message


def generate_drift_transaction(base_transaction, step):
    """Generates a 'drift' transaction by significantly increasing the amount."""
    message = base_transaction.copy()

    original_amount = message["amount"]
    # Induce drift by multiplying the amount by a random factor within the defined range
    random_magnitude = random.uniform(DRIFT_MAGNITUDE_MIN, DRIFT_MAGNITUDE_MAX)
    drifted_amount = round(original_amount * random_magnitude, 2)

    # Update balances logically based on the drifted amount
    balance_diff = drifted_amount - original_amount
    message["amount"] = drifted_amount
    message["newbalanceOrig"] = round(message["newbalanceOrig"] - balance_diff, 2)
    message["newbalanceDest"] = round(message["newbalanceDest"] + balance_diff, 2)

    # Update simulation-specific metadata
    message["step"] = step
    message["producer_timestamp_ms"] = int(time() * 1000)
    message["transactionId"] = str(uuid.uuid4())

    return message, original_amount


def run_simulation():
    """
    Reads transaction data for a specific user and simulates normal and
    drifted transactions based on a single transaction template.
    """
    producer = None
    try:
        logger.info("Loading dataset from: %s", DATA_FILE_PATH)
        df = pd.read_csv(DATA_FILE_PATH)

        # Find the first transaction for the target user to use as a template
        user_transactions = df[df["nameOrig"] == TARGET_USER]
        if user_transactions.empty:
            logger.error(
                "Could not find any transactions for user '%s'. Aborting simulation.",
                TARGET_USER,
            )
            return

        base_transaction = user_transactions.iloc[0].to_dict()
        logger.info(
            "Found a base transaction for user '%s' to use as a template.", TARGET_USER
        )

        producer = create_producer()
        current_step = int(base_transaction.get("step", 1))

        # --- Phase 1: Send Normal Transactions ---
        logger.info(
            "--- Phase 1: Sending %d NORMAL transactions for user %s ---",
            NORMAL_TRANSACTIONS_COUNT,
            TARGET_USER,
        )
        for i in range(NORMAL_TRANSACTIONS_COUNT):
            message = generate_normal_transaction(base_transaction, current_step)
            producer.send(TOPIC_NAME, value=message)
            logger.info(
                "Sent NORMAL transaction %d/%d | Type: %s, Amount: %.2f",
                i + 1,
                NORMAL_TRANSACTIONS_COUNT,
                message["type"],
                message["amount"],
            )
            current_step += 1
            # sleep(1)
        producer.flush()

        # --- Phase 2: Introduce and Send Drifted Transactions ---
        logger.info(
            "--- Phase 2: DRIFT! Sending %d drifted transactions for user %s ---",
            DRIFT_TRANSACTIONS_COUNT,
            TARGET_USER,
        )
        for i in range(DRIFT_TRANSACTIONS_COUNT):
            message, original_amount = generate_drift_transaction(
                base_transaction, current_step
            )
            producer.send(TOPIC_NAME, value=message)
            logger.info(
                "Sent DRIFTED transaction %d/%d | Type: %s, Amount: %.2f (Original: %.2f)",
                i + 1,
                DRIFT_TRANSACTIONS_COUNT,
                message["type"],
                message["amount"],
                original_amount,
            )
            current_step += 1
            # sleep(1)
        producer.flush()

        logger.info("Drift simulation completed successfully.")

    except FileNotFoundError:
        logger.error(
            "FATAL: The data file was not found at '%s'. Please check the path.",
            DATA_FILE_PATH,
        )
    except Exception as e:
        logger.error("An unexpected error occurred during the simulation: %s", e)
    finally:
        if producer:
            producer.close()
            logger.info("Kafka producer closed.")


if __name__ == "__main__":
    run_simulation()
