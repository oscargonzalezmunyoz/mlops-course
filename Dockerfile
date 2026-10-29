FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached independently of code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and the MLflow registry
COPY src/ ./src/
COPY mlruns/ ./mlruns
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]