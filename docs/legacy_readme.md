## Start Docker

docker compose up -d --remove-orphans

docker compose ps

## Start Data Stream (Kafkak Producer)

docker exec -it spark-master bash

cd /home/n00b/....

pip install kafka-python pandas

python producer.py

## Real-Time processing (spark)

-- Spark worker node
----------------------

docker exec -it spark-worker bash

cd /home/n00b....
<!-- pip install joblib torch torchvision torchaudio pyarrow pandas scikit-learn matplotlib seaborn -->
pip install -r requirements.txt --no-cache-dir

-- Spark master node
----------------------

docker exec -it spark-master bash

cd /home/n00b....
<!-- pip install joblib torch torchvision torchaudio pyarrow pandas scikit-learn matplotlib seaborn -->
pip install -r requirements.txt --no-cache-dir

spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 streaming_app.py

spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 streaming_app_v1.py

spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 streaming_app_v2.py

docker compose exec spark-master spark-submit \
    --master spark://spark-master:7077 \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
    /home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/streaming_app_v2.py

spark-submit streaming_app.py

## Observe and verify the result

docker exec -it kafka bash

kafka-console-consumer --bootstrap-server kafka:9092 --topic alerts --from-beginning

    -----------------------

pip install torch==2.3.1 --extra-index-url <https://download.pytorch.org/whl/cpu>
pip install -r requirements.txt --no-deps

# Spark Job

docker compose exec spark-master spark-submit \
    --master spark://spark-master:7077 \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
    --driver-memory 4g \
    --executor-memory 2g \
    /home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/streaming_app_v5.py

# Kafka Producer

docker compose exec spark-master python /home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/producer.py

# Transactions

docker compose exec kafka kafka-console-consumer \
    --bootstrap-server kafka:9092 \
    --topic transactions \
    --from-beginning

# Alerts

docker compose exec kafka kafka-console-consumer \
    --bootstrap-server kafka:9092 \
    --topic alerts \
    --from-beginning

# Performance metrics

docker compose exec kafka kafka-console-consumer \
    --bootstrap-server kafka:9092 \
    --topic performance-metrics \
    --from-beginning

# Explanation

docker compose exec kafka kafka-console-consumer \
    --bootstrap-server kafka:9092 \
    --topic explanations \
    --from-beginning

# Dashboard
<!-- pip install streamlit
streamlit run /home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/dashboard_v2.py -->

KAFKA_BOOTSTRAP_SERVERS=localhost:29092 streamlit run /home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/dashboard_v2.py

<!-- docker compose exec spark-master streamlit run /home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/dashboard.py -->

<http://localhost:8501>

# Drift simulation (see the alert topic, 15 normal, then 15 high valued)

docker compose exec spark-master python /home/n00b/workspace/ArtificialIntelligence/Dissertation/src/v1/producer_drift.py

# lag

docker compose exec kafka kafka-consumer-groups \
    --bootstrap-server kafka:9092 \
    --list

Look for one that starts with spark-kafka-source-

# Replace YOUR_SPARK_GROUP_ID with the ID you found in Step 1

docker compose exec kafka kafka-consumer-groups \
    --bootstrap-server kafka:9092 \
    --describe \
    --group YOUR_SPARK_GROUP_ID
