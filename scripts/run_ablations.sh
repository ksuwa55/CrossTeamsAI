#!/bin/bash
# ============================================================
# Run ablation experiments for QMSum query-focused summarization.
#
# Each experiment produces a JSONL prediction file and is then
# evaluated by eval/evaluate_qmsum.py.
#
# Usage:
#   bash scripts/run_ablations.sh
#
# Prerequisites:
#   pip install -r requirements.txt
#   pip install bert-score        # for BERTScore (optional)
# ============================================================

set -e
set -o pipefail

QMSUM_TEST_DIR="data/QMSum/data/ALL/test"
OUT_DIR="output/ablations"
EVAL_DIR="eval/results/ablations"
mkdir -p "$OUT_DIR" "$EVAL_DIR"

# ── Verify dataset exists ──
if [ ! -d "$QMSUM_TEST_DIR" ]; then
  echo "[Error] QMSum test dir not found: $QMSUM_TEST_DIR"
  echo "Please ensure the QMSum dataset is at data/QMSum/"
  exit 1
fi

# ── Helper: run one experiment ──
run_experiment() {
  local TAG="$1"
  shift
  local PRED_FILE="$OUT_DIR/${TAG}.jsonl"

  echo ""
  echo "============================================"
  echo "  Experiment: $TAG"
  echo "============================================"

  # Generate predictions
  python app/cli_runner.py \
    --qmsum_split_dir "$QMSUM_TEST_DIR" \
    --out_jsonl "$PRED_FILE" \
    --max_meetings 1000000 \
    --max_queries_per_meeting 1000000 \
    --sample_ratio 1.0 \
    "$@"

  # Evaluate (skip BERTScore for speed during ablations)
  python eval/evaluate_qmsum.py \
    --preds_jsonl "$PRED_FILE" \
    --out_dir "$EVAL_DIR" \
    --tag "$TAG" \
    --skip_bertscore
}

# ── Helper: run baselines ──
run_baseline() {
  local TAG="$1"
  local METHOD="$2"
  local N_SENTS="${3:-5}"
  local PRED_FILE="$OUT_DIR/${TAG}.jsonl"

  echo ""
  echo "============================================"
  echo "  Baseline: $TAG"
  echo "============================================"

  python eval/baselines.py \
    --qmsum_split_dir "$QMSUM_TEST_DIR" \
    --method "$METHOD" \
    --n_sents "$N_SENTS" \
    --out_jsonl "$PRED_FILE"

  python eval/evaluate_qmsum.py \
    --preds_jsonl "$PRED_FILE" \
    --out_dir "$EVAL_DIR" \
    --tag "$TAG" \
    --skip_bertscore
}

# ============================================================
# BASELINES
# ============================================================
run_baseline "baseline_lead5"   "lead-n"  5
run_baseline "baseline_lead10"  "lead-n"  10
run_baseline "baseline_tfidf5"  "tfidf"   5
run_baseline "baseline_tfidf10" "tfidf"   10
run_baseline "baseline_random5" "random"  5

# ============================================================
# FULL PIPELINE (default config — this is the "proposed method")
# ============================================================
run_experiment "full_default" \
  --prefilter on --preserve_ngrams on --few_shot off --revise off \
  --chunk_chars 12000 --overlap_chars 1000 --max_tokens 180 --max_sentences 4

# ============================================================
# ABLATION: prefilter off
# ============================================================
run_experiment "ablation_no_prefilter" \
  --prefilter off --preserve_ngrams on --few_shot off --revise off \
  --chunk_chars 12000 --overlap_chars 1000 --max_tokens 180 --max_sentences 4

# ============================================================
# ABLATION: n-gram preservation off
# ============================================================
run_experiment "ablation_no_ngram" \
  --prefilter on --preserve_ngrams off --few_shot off --revise off \
  --chunk_chars 12000 --overlap_chars 1000 --max_tokens 180 --max_sentences 4

# ============================================================
# ABLATION: with revision pass
# ============================================================
run_experiment "ablation_with_revise" \
  --prefilter on --preserve_ngrams on --few_shot off --revise on \
  --chunk_chars 12000 --overlap_chars 1000 --max_tokens 180 --max_sentences 4

# ============================================================
# ABLATION: with few-shot
# ============================================================
run_experiment "ablation_with_fewshot" \
  --prefilter on --preserve_ngrams on --few_shot on --revise off \
  --chunk_chars 12000 --overlap_chars 1000 --max_tokens 180 --max_sentences 4

# ============================================================
# ABLATION: chunk size = 8000
# ============================================================
run_experiment "ablation_chunk8k" \
  --prefilter on --preserve_ngrams on --few_shot off --revise off \
  --chunk_chars 8000 --overlap_chars 800 --max_tokens 180 --max_sentences 4

# ============================================================
# ABLATION: chunk size = 16000
# ============================================================
run_experiment "ablation_chunk16k" \
  --prefilter on --preserve_ngrams on --few_shot off --revise off \
  --chunk_chars 16000 --overlap_chars 1200 --max_tokens 180 --max_sentences 4

# ============================================================
# ABLATION: max_sentences = 2 (shorter output)
# ============================================================
run_experiment "ablation_2sent" \
  --prefilter on --preserve_ngrams on --few_shot off --revise off \
  --chunk_chars 12000 --overlap_chars 1000 --max_tokens 120 --max_sentences 2

# ============================================================
# ABLATION: max_sentences = 6 (longer output)
# ============================================================
run_experiment "ablation_6sent" \
  --prefilter on --preserve_ngrams on --few_shot off --revise off \
  --chunk_chars 12000 --overlap_chars 1000 --max_tokens 250 --max_sentences 6

# ============================================================
# ABLATION: map-reduce vs single-pass (chunk_chars 0 = no chunking)
# ============================================================
run_experiment "ablation_single_pass" \
  --prefilter on --preserve_ngrams on --few_shot off --revise off \
  --chunk_chars 0 --overlap_chars 0 --max_tokens 180 --max_sentences 4

# ============================================================
# ABLATION: temperature = 0.2
# ============================================================
run_experiment "ablation_temp02" \
  --prefilter on --preserve_ngrams on --few_shot off --revise off \
  --chunk_chars 12000 --overlap_chars 1000 --max_tokens 180 --max_sentences 4 \
  --temperature 0.2

# ============================================================
# ABLATION: temperature = 0.5
# ============================================================
run_experiment "ablation_temp05" \
  --prefilter on --preserve_ngrams on --few_shot off --revise off \
  --chunk_chars 12000 --overlap_chars 1000 --max_tokens 180 --max_sentences 4 \
  --temperature 0.5

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  Predictions: $OUT_DIR/"
echo "  Results:     $EVAL_DIR/"
echo "============================================"
echo ""
echo "Running BERTScore on full_default (best config)..."
python eval/evaluate_qmsum.py \
  --preds_jsonl "$OUT_DIR/full_default.jsonl" \
  --out_dir "$EVAL_DIR" \
  --tag "full_default_bertscore"
echo "BERTScore complete. Final results in $EVAL_DIR/"
