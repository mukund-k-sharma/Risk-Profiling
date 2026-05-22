import json
import logging
import os
from time import sleep, time
import pandas as pd
from kafka import KafkaProducer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC_NAME = os.getenv("TRANSACTIONS_TOPIC", "transactions")
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "/home/n00b/workspace/Risk-Profiling/data/PS_20174392719_1491204439457_log.csv")
STREAM_SPEED_SEC = float(os.getenv("STREAM_SPEED_SEC", "0.01"))

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

def stream_transactions(producer, file_path, topic):
    """Stream transactions from a CSV file to a Kafka topic."""
    if not producer:
        logger.error("Producer is not initialized.")
        return

    logger.info(f"Streaming transactions to topic: '{topic}' from file: '{file_path}'")

    try:
        # Read the CSV file in chunks
        for chunk in pd.read_csv(file_path, chunksize=10_000):
            for _, row in chunk.iterrows():
                message = row.to_dict()
                # Clean up float values that are NaN
                for k, v in message.items():
                    if pd.isna(v):
                        message[k] = None
                # Add producer timestamp for latency calculation
                message["producer_timestamp_ms"] = int(time() * 1000)
                
                producer.send(topic, value=message)
                logger.info(f"Sent transaction for user: {message['nameOrig']} | Amount: {message['amount']}")
                
                if STREAM_SPEED_SEC > 0:
                    sleep(STREAM_SPEED_SEC)
            producer.flush()

        logger.info("All transactions streamed successfully.")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error while streaming transactions: {e}")
    finally:
        if producer:
            producer.close()
            logger.info("Kafka producer closed.")

if __name__ == "__main__":
    producer_client = create_producer()
    if not producer_client:
        logger.error("Failed to create Kafka producer. Exiting.")
        exit(1)

    stream_transactions(producer_client, DATA_FILE_PATH, TOPIC_NAME)
