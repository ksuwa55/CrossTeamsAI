#!/bin/bash
set -euo pipefail

IMAGE_NAME="summarizer-app"

echo "Building Docker image ($IMAGE_NAME)..."
docker build -t "$IMAGE_NAME" .

echo "Starting Knowledge Graph Explorer on http://localhost:7862 ..."
docker run --rm -p 7862:7862 \
  -v "$(pwd)/cache_kg:/app/cache_kg" \
  -v "$(pwd)/cache_kg_embeddings:/app/cache_kg_embeddings" \
  --env-file .env \
  "$IMAGE_NAME" \
  python app/kg_dashboard.py
