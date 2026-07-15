#!/bin/bash
set -e
set -o pipefail

# ===== Settings =====
IMAGE_NAME="summarizer-app"
QMSUM_TEST_DIR="/Users/kazukisuwabe/Documents/dev/CrossTeamsAI/data/QMSum/data/ALL/test"
OUT_DIR="/Users/kazukisuwabe/Documents/dev/CrossTeamsAI/output"
LOG_DIR="/Users/kazukisuwabe/Documents/dev/CrossTeamsAI/logs"
ENV_FILE=".env"

# ===== Checks =====
if [ ! -d "$QMSUM_TEST_DIR" ]; then
  echo "[Error] QMSUM test dir not found: $QMSUM_TEST_DIR"
  echo "Usage: $0 /abs/path/to/QMSum/data/ALL/test"
  exit 1
fi
mkdir -p "$OUT_DIR" "$LOG_DIR"

# ===== Build =====
echo "Building Docker image ($IMAGE_NAME)..."
docker build -t "$IMAGE_NAME" .

# ===== Run predictions on limited QMSum test files =====
echo "Running cli_runner..."
docker run --rm \
  -v "$QMSUM_TEST_DIR:/data/qmsum/test:ro" \
  -v "$OUT_DIR:/app/output" \
  -v "$LOG_DIR:/app/logs" \
  --env-file "$ENV_FILE" \
  "$IMAGE_NAME" \
  python app/cli_runner.py \
    --qmsum_split_dir /data/qmsum/test \
    --model gpt-3.5-turbo \
    --out_jsonl output/qmsum_test_preds.jsonl \
    --sample_ratio 1.0 \
    --max_meetings 1000000 \
    --max_queries_per_meeting 1000000 \
    --chunk_chars 7000 \
    --overlap_chars 600

echo "Done. Predictions written to output/qmsum_test_preds.jsonl"
