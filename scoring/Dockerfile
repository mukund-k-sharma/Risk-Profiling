FROM python:3.11-slim

WORKDIR /app

# Install build dependencies if needed for packages like river/shap
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY autoencoder_model.pth .
COPY preprocessor.joblib .
COPY global_stats.json .
COPY X_train_sample.csv .
COPY scoring_service.py .

CMD ["python", "scoring_service.py"]
