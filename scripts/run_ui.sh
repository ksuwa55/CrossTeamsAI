#!/bin/bash
set -euo pipefail

IMAGE_NAME="summarizer-app"

echo "Building Docker image ($IMAGE_NAME)..."
docker build -t "$IMAGE_NAME" .

echo "Starting UI on http://localhost:7860 ..."
docker run --rm -p 7860:7860 \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/cache_ui:/app/cache_ui" \
  --env-file .env \
  "$IMAGE_NAME" \
  python app/ui_runner.py
