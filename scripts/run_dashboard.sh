#!/bin/bash
set -euo pipefail

IMAGE_NAME="summarizer-app"

echo "Building Docker image ($IMAGE_NAME)..."
docker build -t "$IMAGE_NAME" .

echo "Starting unified dashboard (Summarizer / Causal What-If / Knowledge Graph Explorer / Integrated View) on http://localhost:7860 ..."
docker run --rm -p 7860:7860 \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/cache:/app/cache" \
  -v "$(pwd)/cache_ui:/app/cache_ui" \
  -v "$(pwd)/cache_causal:/app/cache_causal" \
  -v "$(pwd)/cache_kg:/app/cache_kg" \
  -v "$(pwd)/cache_kg_embeddings:/app/cache_kg_embeddings" \
  -v "$(pwd)/output:/app/output" \
  --env-file .env \
  "$IMAGE_NAME" \
  python app/dashboard.py
