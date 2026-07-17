#!/bin/bash
# ============================================================
# Compare multiple OpenAI models on QMSum test set.
#
# Models compared:
#   gpt-3.5-turbo   (cheap baseline)
#   gpt-4-turbo     (high-capability)
#   gpt-4o          (latest, fast)
#
# Usage:
#   bash scripts/run_model_comparison.sh
#
# To run only specific models, set MODEL_LIST:
#   MODEL_LIST="gpt-3.5-turbo gpt-4o" bash scripts/run_model_comparison.sh
#
# Prerequisites:
#   pip install -r requirements.txt
#   OPENAI_API_KEY must be set (or present in .env)
# ============================================================

set -e
set -o pipefail

QMSUM_TEST_DIR="data/QMSum/data/ALL/test"
OUT_DIR="output/model_comparison"
EVAL_DIR="eval/results/model_comparison"
mkdir -p "$OUT_DIR" "$EVAL_DIR"

# Load .env if present
if [ -f .env ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

# ── Verify dataset exists ──
if [ ! -d "$QMSUM_TEST_DIR" ]; then
  echo "[Error] QMSum test dir not found: $QMSUM_TEST_DIR"
  exit 1
fi

# Default model list (override via env)
MODEL_LIST="${MODEL_LIST:-gpt-3.5-turbo gpt-4-turbo gpt-4o}"

# ── Helper: run one model ──
run_model() {
  local MODEL="$1"
  local TAG
  TAG="model_$(echo "$MODEL" | tr -d '-.')"
  local PRED_FILE="$OUT_DIR/${TAG}.jsonl"

  echo ""
  echo "============================================"
  echo "  Model: $MODEL  (tag: $TAG)"
  echo "============================================"

  python app/cli_runner.py \
    --qmsum_split_dir "$QMSUM_TEST_DIR" \
    --model "$MODEL" \
    --out_jsonl "$PRED_FILE" \
    --max_meetings 1000000 \
    --max_queries_per_meeting 1000000 \
    --sample_ratio 1.0 \
    --prefilter on \
    --preserve_ngrams on \
    --few_shot off \
    --revise off \
    --chunk_chars 12000 \
    --overlap_chars 1000 \
    --temperature 0.0 \
    --max_tokens 180 \
    --max_sentences 4

  # Evaluate with ROUGE only (fast); add BERTScore below for final model
  python eval/evaluate_qmsum.py \
    --preds_jsonl "$PRED_FILE" \
    --out_dir "$EVAL_DIR" \
    --tag "$TAG" \
    --skip_bertscore

  echo "  Done: $MODEL"
}

# ── Run all models ──
for MODEL in $MODEL_LIST; do
  run_model "$MODEL"
done

# ── Full evaluation (ROUGE + BERTScore) on gpt-4o ──
GPT4O_PRED="$OUT_DIR/model_gpt4o.jsonl"
if [ -f "$GPT4O_PRED" ]; then
  echo ""
  echo "Running BERTScore on gpt-4o predictions..."
  python eval/evaluate_qmsum.py \
    --preds_jsonl "$GPT4O_PRED" \
    --out_dir "$EVAL_DIR" \
    --tag "model_gpt4o_bertscore"
fi

echo ""
echo "============================================"
echo "  Model comparison complete!"
echo "  Predictions: $OUT_DIR/"
echo "  Results:     $EVAL_DIR/"
echo "============================================"
