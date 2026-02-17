# Phase 1: Meeting Summarization — Evaluation Strategy

## Objective

Measure how well the summarization pipeline produces faithful, relevant, and concise query-focused meeting summaries against the QMSum benchmark.

## Benchmark Dataset

**QMSum** (Query-based Multi-domain Meeting Summarization)

- Domain: academic meetings (AMI, ICSI), product meetings, committee meetings
- 32 test meetings with general and specific queries
- Each query has a human-written reference summary

### Dataset Statistics (test split)

| Category | Count |
|---|---|
| Meetings | 32 |
| General queries | ~32 (1 per meeting) |
| Specific queries | ~100+ (varies per meeting) |

## Metrics

### Primary Metrics

| Metric | What it measures | Library |
|---|---|---|
| ROUGE-1 | Unigram overlap (recall of key terms) | `evaluate` (HuggingFace) |
| ROUGE-2 | Bigram overlap (phrase-level similarity) | `evaluate` |
| ROUGE-L | Longest common subsequence (fluency) | `evaluate` |
| BERTScore F1 | Semantic similarity via contextual embeddings | `evaluate` (deberta-xlarge-mnli) |

### Supporting Analysis

| Analysis | Purpose |
|---|---|
| Bootstrap 95% CI | Statistical significance of metric differences |
| General vs specific breakdown | Understand performance by query type |
| Length ratio (pred/ref) | Detect over/under-generation |
| Per-query CSV | Identify failure cases for error analysis |

## Baselines

Three extractive baselines for context:

| Baseline | Method | Purpose |
|---|---|---|
| lead-n | First N transcript sentences | Upper-bound on trivial extraction |
| tfidf | TF-IDF cosine to query | Query-aware extractive baseline |
| random | Random N sentences | Lower-bound sanity check |

## Evaluation Pipeline

### Step 1: Generate Predictions

```bash
# LLM-based predictions
python app/cli_runner.py \
  --qmsum_split_dir data/QMSum/data/ALL/test \
  --model gpt-3.5-turbo \
  --out_jsonl output/qmsum_test_preds.jsonl

# Baseline predictions
python eval/baselines.py \
  --qmsum_split_dir data/QMSum/data/ALL/test \
  --method tfidf \
  --out_jsonl output/qmsum_test_tfidf.jsonl
```

### Step 2: Evaluate

```bash
# Full evaluation (ROUGE + BERTScore)
python eval/evaluate_qmsum.py \
  --preds_jsonl output/qmsum_test_preds.jsonl \
  --tag llm_gpt35

# ROUGE only (faster)
python eval/evaluate_qmsum.py \
  --preds_jsonl output/qmsum_test_preds.jsonl \
  --skip_bertscore \
  --tag llm_gpt35_rouge_only
```

### Step 3: Compare

Compare results across `eval/results/*.json` files.

## Tunable Parameters

Parameters that affect evaluation results:

| Parameter | Default | Impact |
|---|---|---|
| `--temperature` | 0.0 | Lower = more deterministic, better ROUGE |
| `--max_tokens` | 180 | Controls summary length |
| `--max_sentences` | 4 | Length control in prompt |
| `--prefilter` | on | Keyword-based context narrowing |
| `--preserve_ngrams` | on | Encourages bigram reuse for ROUGE-2 |
| `--few_shot` | off | One-shot example in prompt |
| `--revise` | off | Extra polish pass (cost vs quality) |
| `--chunk_chars` | 12000 | Chunk size for map step |
| `--model` | gpt-3.5-turbo | Model selection (cost vs quality) |

## Experiment Tracking

- Predictions: `output/*.jsonl`
- Results: `eval/results/results_<tag>_<timestamp>.json`
- Per-query: `eval/results/per_query_<tag>_<timestamp>.csv`
- API cache: `cache_qmsum/` (avoids re-running identical calls)

## Known Limitations

1. **QMSum-only**: Evaluation is limited to one benchmark. Cross-dataset generalization is untested.
2. **Reference bias**: ROUGE rewards lexical overlap with a single reference, which may penalize valid but differently-worded summaries.
3. **No human evaluation**: BERTScore provides semantic similarity but does not replace human judgment on faithfulness.
4. **Subset sampling**: Default settings sample ~10 meetings x 2 queries for cost control. Full evaluation requires `--sample_ratio 1.0 --max_meetings 999 --max_queries_per_meeting 999`.
