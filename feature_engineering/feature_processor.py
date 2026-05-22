import json
import logging
import os
import joblib
import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from pydantic import ValidationError
from schemas import Transaction

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
CONSUME_TOPIC = os.getenv("TRANSACTIONS_TOPIC", "transactions")
PRODUCE_TOPIC = os.getenv("FEATURED_TOPIC", "featured-transactions")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "/app/preprocessor.joblib")

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

def load_preprocessor():
    """Load StandardScaler from path."""
    try:
        scaler = joblib.load(PREPROCESSOR_PATH)
        logger.info(f"Loaded StandardScaler preprocessor successfully from: {PREPROCESSOR_PATH}")
        return scaler
    except Exception as e:
        logger.error(f"Failed to load StandardScaler preprocessor from {PREPROCESSOR_PATH}: {e}")
        raise e

def create_consumer_producer():
    """Create consumer and producer."""
    try:
        consumer = KafkaConsumer(
            CONSUME_TOPIC,
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            auto_offset_reset="latest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="feature-engineering-group",
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

def process_stream():
    """Process incoming raw transactions."""
    scaler = load_preprocessor()
    consumer, producer = create_consumer_producer()

    logger.info(f"Feature Engineering microservice started. Listening on '{CONSUME_TOPIC}'...")

    for message in consumer:
        raw_data = message.value
        try:
            # 1. Pydantic Validation
            validated_tx = Transaction(**raw_data)
            
            # 2. Extract features & one-hot encode transaction type
            tx_dict = validated_tx.model_dump()
            type_val = tx_dict["type"]
            
            tx_dict["type_CASH_OUT"] = 1.0 if type_val == "CASH_OUT" else 0.0
            tx_dict["type_DEBIT"] = 1.0 if type_val == "DEBIT" else 0.0
            tx_dict["type_PAYMENT"] = 1.0 if type_val == "PAYMENT" else 0.0
            tx_dict["type_TRANSFER"] = 1.0 if type_val == "TRANSFER" else 0.0
            
            # Create features dataframe for standard scaler
            features_df = pd.DataFrame([tx_dict])[FEATURE_COLUMNS]
            
            # 3. Apply Scaling
            scaled_features = scaler.transform(features_df)[0]
            
            # 4. Construct featured transaction payload
            featured_payload = {
                "nameOrig": tx_dict["nameOrig"],
                "step": tx_dict["step"],
                "type": tx_dict["type"],
                "amount": tx_dict["amount"],
                "isFraud": tx_dict["isFraud"],
                "isFlaggedFraud": tx_dict["isFlaggedFraud"],
                "producer_timestamp_ms": tx_dict["producer_timestamp_ms"],
                "scaled_features": scaled_features.tolist(),
                "type_CASH_OUT": tx_dict["type_CASH_OUT"],
                "type_DEBIT": tx_dict["type_DEBIT"],
                "type_PAYMENT": tx_dict["type_PAYMENT"],
                "type_TRANSFER": tx_dict["type_TRANSFER"],
                "oldbalanceOrg": tx_dict["oldbalanceOrg"],
                "newbalanceOrig": tx_dict["newbalanceOrig"],
                "oldbalanceDest": tx_dict["oldbalanceDest"],
                "newbalanceDest": tx_dict["newbalanceDest"],
            }
            
            # 5. Publish to featured-transactions
            producer.send(PRODUCE_TOPIC, value=featured_payload)
            logger.info(f"Processed features for user: {tx_dict['nameOrig']} | Ingest TS: {tx_dict['producer_timestamp_ms']}")
            
        except ValidationError as val_err:
            logger.warning(f"Validation error for record {raw_data}: {val_err.json()}")
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")

if __name__ == "__main__":
    process_stream()
