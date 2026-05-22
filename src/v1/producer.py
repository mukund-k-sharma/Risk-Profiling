import json
import logging
from time import sleep, time

import pandas as pd
from kafka import KafkaProducer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_producer():
    """Create a Kafka producer."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=["kafka:9092"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        logger.info("Kafka producer created successfully.")
        return producer
    except Exception as e:
        logger.error("Failed to create Kafka producer: {e}")
        raise e


def stream_transaction(producer, file_path, topic="transactions"):
    """Stream transactions from a CSV file to a Kafka topic."""
    if not producer:
        logger.error("Producer is not initialized.")
        return

    logger.info(f"Streaming transactions to topic: {topic} from file: {file_path}")

    try:
        # Read the CSV file in chunks
        for chunk in pd.read_csv(file_path, chunksize=10_000):
            for _, row in chunk.iterrows():
                message = row.to_dict()
                # Add a timestamp to the message
                message["producer_timestamp_ms"] = int(time() * 1000)
                producer.send(topic, value=message)
                logger.info(f"Sent transaction for: {message['nameOrig']}")
                # sleep(0.05)  # Sleep to simulate real-time streaming
            producer.flush()

        logger.info("All transactions streamed successfully.")
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        logger.error(str(e))
    except Exception as e:
        logger.error(f"Error while streaming transactions: {e}")
        # raise e
    finally:
        if producer:
            producer.close()
            logger.info("Kafka producer closed.")


if __name__ == "__main__":
    # Kafka producer
    producer_client = create_producer()
    if not producer_client:
        logger.error("Failed to create Kafka producer. Exiting.")
        exit(1)

    # PaySim data
    data_file_path = "/home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/PS_20174392719_1491204439457_log.csv"

    stream_transaction(producer_client, data_file_path)
    logger.info("Producer script completed.")
