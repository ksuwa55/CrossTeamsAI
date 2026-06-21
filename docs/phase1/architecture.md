# Phase 1: Meeting Summarization — Architecture

## Overview

Phase 1 provides query-focused meeting summarization using OpenAI LLMs with a map-reduce pipeline. It supports single-transcript summarization, batch evaluation on QMSum benchmarks, and a Gradio-based web UI.

## System Architecture

```
                         ┌─────────────────────────┐
                         │       Entry Points       │
                         ├─────────────────────────┤
                         │  CLI Runner (batch eval) │
                         │  UI Runner  (Gradio)     │
                         └────────────┬────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │   Summarization Engine   │
                         │  (MeetingSummarizer)     │
                         ├─────────────────────────┤
                         │  - Prompt building       │
                         │  - OpenAI API calls      │
                         │  - Disk cache            │
                         │  - Revision pass         │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                  │
           ┌────────▼──────┐  ┌──────▼───────┐  ┌──────▼───────┐
           │  QMSum Loader │  │  Evaluation   │  │   Baselines  │
           │  (data I/O)   │  │  (ROUGE/BERT) │  │  (lead/tfidf)│
           └───────────────┘  └──────────────┘  └──────────────┘
```

## Components

### 1. MeetingSummarizer (`01_summarization/summarizer.py`)

Core engine shared by all entry points.

| Responsibility | Method |
|---|---|
| Load & preprocess transcript | `load_and_preprocess_transcript()` |
| Build mode-specific prompts | `build_prompt(mode=general/decision/blocker/query)` |
| Call OpenAI with caching | `run_summarizer()` |
| Polish output | `revise_summary()` |
| Parse & persist results | `parse_output()`, `save_summary()`, `log_experiment()` |

**Key design decisions:**
- Disk cache keyed by `md5(model + system_prompt + prompt)` to avoid redundant API calls
- Low temperature (0.2) for stable, reproducible outputs
- Modest max_tokens (220) to reduce drift and cost

### 2. QMSum Data Loader (`01_summarization/qmsum_loader.py`)

Converts QMSum JSON files into per-query examples.

- Handles `general_query_list` and `specific_query_list` formats
- Cleans transcripts (strips emoji, filler words)
- Yields `{meeting_id, query_id, query, input_text, reference}`

### 3. CLI Runner (`app/cli_runner.py`)

Batch evaluation pipeline with map-reduce strategy.

```
Input Transcript
       │
       ▼
Keyword Prefilter (optional) ── cheap retrieval to narrow context
       │
       ▼
Chunking (12K chars, 1K overlap)
       │
       ▼
Map: summarize each chunk ── per-chunk LLM call
       │
       ▼
Reduce: merge partial summaries ── single LLM call
       │
       ▼
Revise (optional) ── polish pass
       │
       ▼
Output: JSONL predictions
```

**Configurable knobs:**
- `--chunk_chars`, `--overlap_chars`, `--max_chunks`
- `--few_shot`, `--revise`, `--prefilter`, `--preserve_ngrams`
- `--temperature`, `--max_tokens`, `--max_sentences`
- `--sample_ratio`, `--max_meetings`, `--max_queries_per_meeting`

### 4. UI Runner (`app/ui_runner.py`)

Gradio web interface for interactive summarization.

- Supports 6 task types: general, decisions, blockers, action_items, risks, followups
- Adjustable chunking, temperature, and token parameters
- Runs on `0.0.0.0:7860` (Docker-compatible)

### 5. Evaluation (`eval/evaluate_qmsum.py`)

Comprehensive evaluation with:
- ROUGE-1/2/L/Lsum (lexical overlap)
- BERTScore F1 (semantic similarity)
- Bootstrap 95% confidence intervals
- General vs specific query breakdown
- Per-query CSV + aggregate JSON output

### 6. Baselines (`eval/baselines.py`)

Extractive baselines for comparison:
- **lead-n**: first N sentences
- **tfidf**: TF-IDF cosine extractive (query-aware)
- **random**: random N sentences (lower bound)

## Data Flow

```
QMSum JSON files (data/QMSum/data/ALL/test/*.json)
       │
       ▼  qmsum_loader.iter_qmsum()
Per-query examples
       │
       ▼  cli_runner.py (map-reduce pipeline)
Predictions JSONL (output/qmsum_test_preds.jsonl)
       │
       ▼  evaluate_qmsum.py
Results JSON + CSV (eval/results/)
```

## Infrastructure

| Component | Technology |
|---|---|
| LLM | OpenAI API (gpt-3.5-turbo default) |
| UI | Gradio |
| Containerization | Docker (python:3.10-slim) |
| Evaluation | HuggingFace `evaluate` library |
| Configuration | CLI args + `.env` for API keys |

## Directory Structure

```
01_summarization/
├── summarizer.py        # Core summarization engine
├── qmsum_loader.py      # QMSum dataset loader
app/
├── cli_runner.py         # Batch evaluation entry point
├── ui_runner.py          # Gradio web UI
eval/
├── evaluate_qmsum.py    # Evaluation metrics
├── baselines.py         # Extractive baselines
├── results/             # Evaluation outputs
data/
├── QMSum/               # QMSum benchmark dataset
├── zoom_transcript_sample.json
├── slack_transcript_sample.json
cache/                   # API response cache (single transcript)
cache_qmsum/             # API response cache (QMSum batch)
cache_ui/                # API response cache (UI)
output/                  # Prediction outputs
logs/                    # Experiment logs
```
