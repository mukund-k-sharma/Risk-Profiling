FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run tail -f /dev/null by default so the container stays alive,
# allowing users to run standard producer or drift producer on-demand
CMD ["tail", "-f", "/dev/null"]
