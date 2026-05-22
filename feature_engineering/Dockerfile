FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocessor.joblib .
COPY schemas.py .
COPY feature_processor.py .

CMD ["python", "feature_processor.py"]
