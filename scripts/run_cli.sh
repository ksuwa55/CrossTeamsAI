#!/bin/bash
set -e
set -o pipefail

# ===== Settings =====
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_NAME="summarizer-app"
# docker -v needs absolute paths, so resolve everything from the repo root.
QMSUM_TEST_DIR="${1:-$REPO_ROOT/data/QMSum/data/ALL/test}"
OUT_DIR="$REPO_ROOT/output"
LOG_DIR="$REPO_ROOT/logs"
ENV_FILE="$REPO_ROOT/.env"

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
echo "Running run_qmsum_benchmark..."
docker run --rm \
  -v "$QMSUM_TEST_DIR:/data/qmsum/test:ro" \
  -v "$OUT_DIR:/app/output" \
  -v "$LOG_DIR:/app/logs" \
  --env-file "$ENV_FILE" \
  "$IMAGE_NAME" \
  python app/run_qmsum_benchmark.py \
    --qmsum_split_dir /data/qmsum/test \
    --model gpt-3.5-turbo \
    --out_jsonl output/qmsum_test_preds.jsonl \
    --sample_ratio 1.0 \
    --max_meetings 1000000 \
    --max_queries_per_meeting 1000000 \
    --chunk_chars 7000 \
    --overlap_chars 600

echo "Done. Predictions written to output/qmsum_test_preds.jsonl"
