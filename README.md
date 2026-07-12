## Phase 1: Meeting Summarization

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

## Phase 2: Causal Modeling of Project Bottlenecks

Builds on Phase 1's summarizer to go one step further than summarizing meetings: it extracts cause→effect events across meetings, merges them into a shared causal graph, and lets a manager simulate mitigating a root cause.

* Path A (Per-meeting causal extraction): `02_causal_modeling/extract_variables.py`
* Path B (Graph build + intervention simulation): `02_causal_modeling/build_causal_graph.py` + `02_causal_modeling/simulate_intervention.py`
* Path C (Manager-facing "what-if" dashboard): `app/intervention_dashboard.py`
* Path D (Extraction evaluation): `eval/evaluate_causal_extraction.py` + `eval/inspect_false_negatives.py`

### `02_causal_modeling/extract_variables.py`

Purpose: Turn a raw transcript into candidate cause→effect events.

What it does:

* `enrich_transcript()` — cheap regex prefilter that tags each utterance with `speaker_role`, `topic`, `utterance_type`, `emotion`, `decision`, `blocker`, `next_action`.
* `select_candidate_windows()` — picks ±N-utterance windows around flagged (blocker/decision/next_action) utterances, so only relevant context gets an LLM call.
* `extract_causal_events()` — sends each window to the LLM (reuses `MeetingSummarizer` from Phase 1, including its disk cache) and asks for `{cause, effect, timestamp}` JSON.
* Parses and de-duplicates the LLM's output by `(cause, effect)`.

Used by: `eval/evaluate_causal_extraction.py`, `app/intervention_dashboard.py` (indirectly, via pre-extracted sample events)

### `02_causal_modeling/build_causal_graph.py`

Purpose: Merge causal events from many meetings into one directed acyclic graph (DAG).

What it does:

* `canonicalize_node()` — maps free-text cause/effect phrases onto a fixed vocabulary (`blocked_dependency`, `decision_stagnation`, `ambiguous_requirement`, `scope_change`, `missing_resource`, `qa_delay`, `integration_delay`, `missed_deadline`, or `other:...`) so the graph doesn't fragment into one node per unique LLM phrasing.
* `build_graph()` — builds a `networkx.DiGraph`; edge weight = number of times that cause→effect was observed, each edge keeps its source transcript evidence.
* `build_occurrence_table()` — one row per meeting, one binary column per DAG node, for feeding `dowhy`.
* Saves graph JSON/GML and a rendered PNG; optionally sanity-checks a treatment/outcome pair with `dowhy.CausalModel`.

Used by: `02_causal_modeling/simulate_intervention.py`, `app/intervention_dashboard.py`

### `02_causal_modeling/simulate_intervention.py`

Purpose: Fit a structural causal model (SCM) over the DAG and answer "what if we mitigated this cause?".

What it does:

* `build_scm()` — root nodes get an empirical (observed-frequency) distribution; downstream nodes get a logistic-regression mechanism over their parents (`dowhy.gcm.ClassifierFCM`), so an intervention on a parent propagates through do-calculus.
* `simulate_intervention()` — draws baseline samples vs. samples with the cause node forced to `"0"`, reports baseline probability, intervened probability, and absolute/relative reduction in the outcome node.

Used by: `app/intervention_dashboard.py`

### `app/intervention_dashboard.py`

Purpose: Gradio "what-if" web UI (port 7861) for managers to explore mitigating a root cause.

What it does:

* Loads the DAG + SCM once at startup from `data/sample_causal_events/*.json`.
* Lets a manager pick a cause and an outcome, toggle mitigation, and run the simulation.
* Renders a bar chart of baseline vs. mitigated probability.
* Shows the shortest causal path and the real transcript quotes behind each hop, so the graph's claim is traceable back to source text rather than an opaque node name.

### `eval/evaluate_causal_extraction.py` and `eval/inspect_false_negatives.py`

Purpose: Score the causal-event extraction pipeline against 10 hand-labeled synthetic meetings (`data/synthetic_transcripts/`).

What it does:

* Runs `enrich_transcript()` + `extract_causal_events()` unmodified, then matches predicted vs. gold cause/effect pairs using an LLM judge (`gpt-4o-mini`) plus one-to-one maximum bipartite matching (embeddings can't reliably tell "same causal claim, different wording" apart from "related but topically similar").
* Computes per-meeting and pooled precision/recall/F1, writing a markdown report and raw JSON to `eval/results/causal_extraction/`.
* `inspect_false_negatives.py` classifies every missed gold pair (`phrasing_mismatch` / `partial_match` / `genuine_miss`) by asking the judge to find the closest prediction.

Results (pooled, `gpt-3.5-turbo` extraction / `gpt-4o-mini` judge): Precision 22.0%, Recall 57.9%, F1 31.9% — see `docs/phase2/pipeline-flow-and-results.md` for the full per-meeting breakdown and interpretation.

## Phase 3: AI Documentation

Generates standardized "model card" documentation for Phase 1 and Phase 2, 
following the four documentation-requirement categories from Königstorfer & 
Thalmann (2022) — Model Design Choices, Data Characteristics, Evaluation 
Metrics, System Modification Log — supplemented by the Model Cards framework 
(Mitchell et al., 2019).

* Path A (Card generation): `03_AI_documentation/generate_model_card.py`

### `03_AI_documentation/generate_model_card.py`

Purpose: Render a model card for any component under `03_AI_documentation/components/<name>/`.

What it does:

* Reads six human-authored Markdown fragments per component (intended use, 
  design choices, data characteristics, eval metrics narrative, modification 
  log, known limitations) — judgment calls that can't be derived from code.
* Reads evaluation numbers live from `eval/results/**`, keyed by a dotted 
  path declared in the component's `config.json`, so the metrics table can't 
  drift from the actual result files.
* Stamps the output with the current git commit/branch so each card is 
  traceable to the exact code state it describes.
* Renders both cards: `docs/phase3/summarization.md`, `docs/phase3/causal_modeling.md`

Usage:

    python 03_AI_documentation/generate_model_card.py --component causal_modeling
    python 03_AI_documentation/generate_model_card.py --all