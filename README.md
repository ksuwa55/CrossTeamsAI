Project has two execution paths:

* Path A (Single transcript summarization): `app/main.py` + `01_summarization/summarizer.py`
* Path B (QMSum benchmark run): `app/cli_runner.py` + `01_summarization/qmsum_loader.py` + `01_summarization/summarizer.py`

### `01_summarization/qmsum_loader.py`

Purpose: Convert QMSum JSON files into per-query examples suitable for evaluation.

What it does:

* Builds a clean, concatenated transcript from `meeting_transcripts` (removes emoji and simple fillers like “uh/um”).
* Extracts queries and reference summaries from `general_query_list` / `specific_query_list` (handles format variations).
* Iterates over all JSON files in a split directory and yields one example per query:

  * `meeting_id`, `query_id`, `query`
  * `input_text` (instruction + full transcript)
  * `reference` (gold summary)

Used by: `app/cli_runner.py`

### `01_summarization/summarizer.py`

Purpose: Minimal OpenAI-based summarization component shared by both paths.

Key responsibilities:

* Load & preprocess a single transcript JSON (speaker/timestamp/text normalization; removes emoji and “uh/um”).
* Build prompts for four modes: `general`, `decision`, `blocker`, `query`.
* Call OpenAI Chat Completions with:

  * optional system prompt
  * low temperature
  * token limit
  * disk cache keyed by `(model + system_prompt + prompt)` to avoid repeated calls.
* Optional polishing pass (`revise_summary`) to edit a draft into a concise, faithful final answer.
* Parse and persist outputs for the single-transcript flow:

  * `parse_output()` splits “Action Items:” into `summary` + `action_items`
  * `save_summary()` writes JSON output
  * `log_experiment()` writes a run log to `logs/`

Used by: `app/main.py`, `app/cli_runner.py`

### `app/cli_runner.py`

Purpose: Run QMSum evaluation and write predictions to a JSONL file.

What it does:

* Loads examples via `iter_qmsum()` and groups them by `meeting_id`.
* Samples meetings and caps queries per meeting (for controllable cost/coverage).
* Runs a map–reduce summarization pipeline per query for long transcripts:

  * optional keyword-based prefiltering (cheap retrieval)
  * chunking with overlap
  * “map” summarization per chunk
  * “reduce” merge of partial answers
  * optional revision pass
* Writes one JSON line per query:

  * `meeting_id`, `query_id`, `query`, `prediction`, `reference`

Output: `output/qmsum_test_preds.jsonl` by default.

### `app/main.py`

Purpose: Simple entry point for summarizing a single transcript file (PoC / demo run).

What it does:

* Loads one transcript JSON (default: `data/zoom_transcript_sample.json`)
* Builds a prompt for a selected mode (`general/decision/blocker/query`)
* Calls the summarizer, parses the result, saves `output/summary.json`, and logs the run.
