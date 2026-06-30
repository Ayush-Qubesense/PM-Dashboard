FROM python:3.11-slim

WORKDIR /app

# Build tools needed by lightgbm / scikit-learn C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install production Python deps first (layer-cached unless file changes)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code and data
COPY . .

# Render injects $PORT at runtime; default to 8000 for local docker run
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]