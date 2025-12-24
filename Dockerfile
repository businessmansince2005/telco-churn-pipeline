# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (g++ is needed for xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY app/ ./app/
COPY scripts/train.py ./scripts/train.py
# Ensure data directory exists and copy dataset from scripts/data
RUN mkdir -p scripts/data
COPY scripts/data/telco_churn.csv.csv ./scripts/data/telco_churn.csv.csv
# Copy MLflow artifacts (models)
COPY mlruns/ ./mlruns/

# Expose port 8000
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=file:./mlruns
ENV PORT=8000

# Run the application (use PORT env var for Render compatibility)
CMD uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-8000}
