# Phase 2: Causal Modeling of Project Bottlenecks — Data Design

## Input Formats

### 1. Synthetic Meeting Transcript

Source: `data/synthetic_transcripts/meeting_XX_<label>.json`. Same shape as Phase 1's single-transcript format.

```json
[
  {
    "speaker": "Aiko - PM",
    "timestamp": "00:00:20",
    "text": "Thanks Jonas. From my side, the main blocker is still the data-sharing agreement with the Berlin office. Legal hasn't signed off yet."
  }
]
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `speaker` / `user` | string | yes | Either key is accepted |
| `timestamp` | string | no | Free-form (e.g. "00:01:30") |
| `text` | string | yes | Raw utterance |

10 meetings ship with the repo, each built around one labeled failure mode: `signoff_blocker`, `ambiguous_requirement`, `missing_resource`, `api_dependency`, `decision_stagnation`, `timezone_communication`, `scope_change`, `qa_delay_deadline`, `security_ambiguity`, `resource_reprioritization`.

### 2. Gold Labels

Source: `data/synthetic_transcripts/meeting_XX_<label>.labels.json`. Hand-written ground-truth cause/effect pairs for evaluation.

```json
[
  {
    "cause": "legal sign-off on the data-sharing agreement with Berlin is pending",
    "effect": "the customer records module cannot be pushed",
    "timestamp": "00:00:45"
  }
]
```

| Field | Type | Notes |
|---|---|---|
| `cause` | string | Free-text paraphrase of the underlying cause |
| `effect` | string | Free-text paraphrase of the underlying effect |
| `timestamp` | string | Timestamp of the utterance where the effect is visible |

## Intermediate Formats

### Enriched Transcript (internal)

Produced by `enrich_transcript()` in `02_causal_modeling/extract_variables.py` — one entry per utterance, tagged by cheap regex rules:

```json
{
  "speaker": "Liam - Engineer",
  "timestamp": "00:00:45",
  "text": "Right, we're blocked on that...",
  "speaker_role": "engineer",
  "topic": "general",
  "utterance_type": "statement",
  "emotion": "negative",
  "decision": false,
  "blocker": true,
  "next_action": false
}
```

These flags (`blocker`, `decision`, `next_action`) are what `select_candidate_windows()` uses to pick which ±N-utterance windows are worth an LLM call.

### Causal Event (LLM output)

Yielded by `extract_causal_events()`, one array item per cause→effect relationship the LLM finds in a window:

```json
{
  "cause": "Data-sharing agreement not signed off",
  "effect": "Blocking development of customer records module",
  "timestamp": "00:00:45",
  "meeting_id": "meeting_01_signoff_blocker"
}
```

## Graph Formats

### Canonical Node Vocabulary

`canonicalize_node()` in `build_causal_graph.py` maps every raw cause/effect phrase onto one of these categories (first matching rule wins), or `other:<first 40 chars>` if nothing matches:

| Node | Example trigger phrases |
|---|---|
| `decision_stagnation` | "stagnat", "no decision", "no consensus", "stalled" |
| `blocked_dependency` | "waiting for", "blocked on", "pending approval", "sign-off" |
| `ambiguous_requirement` | "ambiguous", "unclear requirement", "unclear scope" |
| `scope_change` | "scope change", "new requirement", "re-scope" |
| `missing_resource` | "understaffed", "no developer available", "short-staffed" |
| `qa_delay` | "qa delay", "testing delay", "qa backlog" |
| `integration_delay` | "integration" |
| `missed_deadline` | "deadline", "missed", "slipped", "push the launch" |

### Causal Graph JSON

Written by `save_graph_json()` to `output/causal_graph.json`:

```json
{
  "nodes": ["blocked_dependency", "decision_stagnation", "qa_delay"],
  "edges": [
    {
      "source": "blocked_dependency",
      "target": "decision_stagnation",
      "weight": 3,
      "evidence": [
        {"cause": "...", "effect": "...", "timestamp": "00:00:45", "meeting_id": "meeting_01_signoff_blocker"}
      ]
    }
  ]
}
```

`weight` is the number of raw causal events canonicalized onto that edge; `evidence` retains the original (pre-canonicalization) cause/effect text and meeting_id so the dashboard can show real quotes.

### Occurrence Table (for DoWhy)

Built by `build_occurrence_table()` / `build_occurrence_data()`: one row per meeting, one binary (later stringified) column per DAG node — did that node's cause/effect appear anywhere in the meeting?

| meeting_id | blocked_dependency | decision_stagnation | qa_delay | ... |
|---|---|---|---|---|
| meeting_01_signoff_blocker | 1 | 1 | 0 | ... |

`simulate_intervention.py`'s `build_occurrence_data()` casts every column to string so `dowhy.gcm` treats each node as categorical, and backfills any meeting with zero causal events as an all-zero row so it isn't silently dropped.

## Output Formats

### 1. Predicted Causal Events (per meeting)

Written by `evaluate_causal_extraction.py` to `output/causal_events/<meeting_id>.causal_events.json` — same shape as the LLM output above.

### 2. Causal Extraction Evaluation Report

Written by `evaluate_causal_extraction.py` to `eval/results/causal_extraction/causal_extraction_report.md` (human-readable) and `causal_extraction_raw.json` (machine-readable):

```json
{
  "per_meeting": [
    {
      "meeting_id": "meeting_01_signoff_blocker",
      "num_predicted": 7, "num_gold": 3,
      "tp": 2, "fp": 5, "fn": 1,
      "precision": 0.286, "recall": 0.667, "f1": 0.400,
      "true_positives": [...], "false_positives": [...], "false_negatives": [...]
    }
  ],
  "aggregate": {"tp": 11, "fp": 39, "fn": 8, "precision": 0.220, "recall": 0.579, "f1": 0.319}
}
```

### 3. False Negative Inspection

Written by `inspect_false_negatives.py` to `false_negative_inspection.md` / `.json` — for every missed gold pair, the closest predicted pair and a verdict:

```json
{
  "meeting_id": "meeting_01_signoff_blocker",
  "gold": {"cause": "the customer records module cannot be pushed", "effect": "QA is holding off on writing test cases for that module", "timestamp": "00:01:10"},
  "closest_predicted": {"cause": "Delay in legal approval", "effect": "Stalled progress on writing test cases", "timestamp": "00:01:10"},
  "verdict": "partial_match",
  "reasoning": "..."
}
```

## Cache Format

Same as Phase 1 — reuses `MeetingSummarizer`'s disk cache, keyed by `md5(model + system_prompt + prompt)`:

- `cache_causal/` — causal-event extraction calls (`gpt-3.5-turbo`)
- `cache_causal_eval/` — LLM-judge calls for evaluation and false-negative inspection (`gpt-4o-mini`)
