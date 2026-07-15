#!/bin/bash
set -e
set -o pipefail

IMAGE_NAME="summarizer-app"
OUT_DIR="$(pwd)/output"
PRED_FILE="output/qmsum_test_preds_alldata.jsonl"

# Build (ensure image is up to date)
echo "Building Docker image ($IMAGE_NAME)..."
docker build -t "$IMAGE_NAME" .

# Evaluate ROUGE over the whole test set
echo "Evaluating ROUGE on QMSum predictions..."
docker run --rm \
  -v "$PWD/output:/app/output" \
  --entrypoint python \
  summarizer-app \
  eval/evaluate_qmsum.py \
    --preds_jsonl /app/output/qmsum_test_preds_alldata.jsonl
