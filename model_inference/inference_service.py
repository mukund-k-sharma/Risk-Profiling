import json
import logging
import os
import torch
from kafka import KafkaConsumer, KafkaProducer
from model import Autoencoder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
CONSUME_TOPIC = os.getenv("FEATURED_TOPIC", "featured-transactions")
PRODUCE_TOPIC = os.getenv("ERRORS_TOPIC", "reconstruction-errors")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/autoencoder_model.pth")

def load_model():
    """Load PyTorch Autoencoder model."""
    try:
        model = Autoencoder(input_dim=11)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        logger.info(f"Loaded Autoencoder model successfully from: {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Failed to load Autoencoder model from {MODEL_PATH}: {e}")
        raise e

def create_consumer_producer():
    """Create consumer and producer."""
    try:
        consumer = KafkaConsumer(
            CONSUME_TOPIC,
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            auto_offset_reset="latest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id="model-inference-group",
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
    """Process incoming featured transactions."""
    model = load_model()
    consumer, producer = create_consumer_producer()

    logger.info(f"Model Inference microservice started. Listening on '{CONSUME_TOPIC}'...")

    for message in consumer:
        payload = message.value
        try:
            scaled_features = payload["scaled_features"]
            
            # Forward pass
            X_tensor = torch.FloatTensor([scaled_features])
            with torch.no_grad():
                reconstruction = model(X_tensor)
                mse = torch.mean((X_tensor - reconstruction) ** 2).item()
            
            # Append reconstruction error to record
            payload["reconstruction_error"] = mse
            
            # Publish to reconstruction-errors topic
            producer.send(PRODUCE_TOPIC, value=payload)
            logger.info(f"Inference complete for: {payload['nameOrig']} | Error: {mse:.6f}")
            
        except Exception as e:
            logger.error(f"Error processing transaction in model inference: {e}")

if __name__ == "__main__":
    process_stream()
