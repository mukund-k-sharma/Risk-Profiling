FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Install packages, with special PyTorch pip registry
RUN pip install --no-cache-dir -r requirements.txt

COPY autoencoder_model.pth .
COPY model.py .
COPY inference_service.py .

CMD ["python", "inference_service.py"]
