# Phase 1: Meeting Summarization — Data Design

## Input Formats

### 1. Single Transcript (Zoom/Slack)

Used by `MeetingSummarizer.load_and_preprocess_transcript()` and the UI runner.

```json
[
  {
    "speaker": "Host",
    "timestamp": "00:00:02",
    "text": "Welcome everyone to the weekly product sync."
  },
  {
    "speaker": "Engineer 1",
    "timestamp": "00:00:07",
    "text": "We made progress on the login bug."
  }
]
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `speaker` / `user` | string | yes | Either key is accepted |
| `timestamp` | string | no | Free-form (e.g. "00:01:30") |
| `text` | string | yes | Raw utterance |

### 2. QMSum Benchmark Format

Source: `data/QMSum/data/ALL/{train,val,test}/*.json`

```json
{
  "meeting_id": "ES2004a",
  "meeting_transcripts": [
    { "speaker": "A", "content": "..." }
  ],
  "general_query_list": [
    {
      "query_id": "gen-0",
      "query": "Summarize the whole meeting.",
      "answer": "The team discussed..."
    }
  ],
  "specific_query_list": [
    {
      "query_id": "spec-0",
      "query": "What did they decide about the remote control design?",
      "answer": "They decided..."
    }
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `meeting_id` | string | Unique identifier per meeting |
| `meeting_transcripts` | array | Speaker turns with `speaker` + `content` |
| `general_query_list` | array | Broad summary queries with reference answers |
| `specific_query_list` | array | Targeted queries with reference answers |

## Intermediate Formats

### Preprocessed Transcript (internal)

After `load_and_preprocess_transcript()` or `_concat_transcript()`:

```
Host [00:00:02]: Welcome everyone to the weekly product sync.
Engineer 1 [00:00:07]: We made progress on the login bug.
```

Preprocessing applied:
- Emoji removal (`[\u2600-\u26FF\u2700-\u27BF]`)
- Filler word removal (`uh`, `um`)
- Speaker name normalization (`user` → `speaker`)

### Per-Query Example (from QMSum loader)

Yielded by `iter_qmsum()`:

```python
{
    "meeting_id": "ES2004a",
    "query_id": "gen-0",
    "query": "Summarize the whole meeting.",
    "input_text": "Summarize the meeting in response to this query: '...'\n\n<transcript>",
    "reference": "The team discussed..."
}
```

## Output Formats

### 1. Predictions JSONL

Written by `cli_runner.py` and `baselines.py`.

```jsonl
{"meeting_id": "ES2004a", "query_id": "gen-0", "query": "...", "prediction": "...", "reference": "..."}
{"meeting_id": "ES2004a", "query_id": "spec-0", "query": "...", "prediction": "...", "reference": "..."}
```

### 2. Single Transcript Summary

Written by `MeetingSummarizer.save_summary()`:

```json
{
  "summary": "The team discussed the login bug fix...",
  "action_items": [
    "QA testing by Thursday",
    "Prepare release notes"
  ],
  "decisions": [],
  "blockers": []
}
```

### 3. Evaluation Results JSON

Written by `evaluate_qmsum.py`:

```json
{
  "dataset": "QMSum",
  "source": "output/qmsum_test_preds.jsonl",
  "instances": 20,
  "timestamp": "2026-01-05T11:58:00",
  "metrics": {
    "rouge1": { "mean": 0.3245, "ci95_lo": 0.2980, "ci95_hi": 0.3510 },
    "rouge2": { "mean": 0.0892, "ci95_lo": 0.0720, "ci95_hi": 0.1064 },
    "rougeL": { "mean": 0.2134, "ci95_lo": 0.1890, "ci95_hi": 0.2378 },
    "bertscore_f1": { "mean": 0.6821, "ci95_lo": 0.6650, "ci95_hi": 0.6992 }
  },
  "length": {
    "pred_words_mean": 52.3,
    "ref_words_mean": 68.1,
    "length_ratio_words": 0.77
  },
  "breakdown": {
    "general": { "rouge1": 0.3400, "rouge2": 0.0950, "rougeL": 0.2200, "count": 10 },
    "specific": { "rouge1": 0.3090, "rouge2": 0.0834, "rougeL": 0.2068, "count": 10 }
  }
}
```

### 4. Per-Query Evaluation CSV

Written by `evaluate_qmsum.py`:

| meeting_id | query_id | query_type | rouge1 | rouge2 | rougeL | pred_words | ref_words | bertscore_f1 |
|---|---|---|---|---|---|---|---|---|
| ES2004a | gen-0 | general | 0.3500 | 0.1000 | 0.2300 | 48 | 65 | 0.6900 |

## Cache Format

Stored in `cache*/` directories as JSON files named by MD5 hash.

```json
{
  "prompt": "<full prompt text>",
  "system_prompt": "<system prompt or null>",
  "response": "<LLM response text>"
}
```

Cache key: `md5(model + system_prompt + prompt)`

## Experiment Log Format

Written by `MeetingSummarizer.log_experiment()` to `logs/log_YYYYMMDD_HHMMSS.json`:

```json
{
  "timestamp": "2026-01-05T12:00:00",
  "inputs": { "file": "...", "mode": "general" },
  "outputs": { "summary": "...", "action_items": [] },
  "metadata": { "model": "gpt-3.5-turbo" }
}
```
