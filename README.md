# CrossTeamsAI

An LLM-based pipeline that turns meeting transcripts into: query-focused
summaries (Phase 1), a causal graph of project bottlenecks with what-if
simulation (Phase 2), auto-generated model-card documentation (Phase 3), and
a searchable knowledge graph of people/decisions/issues/tasks (Phase 4) — all
bundled into one dashboard and cross-linked (Phase 5).

## Quickstart

```bash
git clone https://github.com/ksuwa55/CrossTeamsAI.git
cd CrossTeamsAI
pip install -r requirements.txt
cp .env.example .env   # then edit .env and set OPENAI_API_KEY
python app/dashboard.py
# open http://localhost:7860
```

Or with Docker:

```bash
bash scripts/run_dashboard.sh   # builds the image and serves the dashboard on :7860
```

The Knowledge Graph Explorer's keyword search and the Causal What-If simulator
work offline (no API key needed) since they run against committed sample data
(`data/sample_causal_events/`, `data/sample_kg_triples/`). Live summarization,
causal/KG extraction on new transcripts, and semantic search require a valid
`OPENAI_API_KEY` with credits.

**Note on reproducing the Integrated View / evaluation numbers:** `output/causal_events/`,
`output/kg_triples/`, and `eval/results/` are *not* committed (they're
regenerable, LLM-derived artifacts, not source). On a fresh clone, the
Integrated View tab and the numbers cited in the Phase 1-4 docs/model cards
won't be populated until you regenerate them with a valid `OPENAI_API_KEY`:

```bash
python eval/evaluate_causal_extraction.py   # regenerates output/causal_events/, eval/results/causal_extraction/
python eval/evaluate_kg_extraction.py       # regenerates output/kg_triples/, eval/results/kg_extraction/
python eval/evaluate_qmsum.py --preds_jsonl <predictions.jsonl>  # regenerates eval/results/metrics_*
```

## License

MIT — see [LICENSE](LICENSE). See [CITATION.cff](CITATION.cff) if you use this
work in academic writing.

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

## Phase 4: Knowledge Graph + Dashboard

Builds a knowledge graph connecting people, decisions, issues/blockers, tasks, and topics from meeting transcripts, then exposes graph search, network visualization, and team-dynamics analytics through a Gradio dashboard. Reuses Phase 1's summarizer/cache and Phase 2's regex prefilter + candidate-window selection unmodified.

* Path A (Per-meeting entity/relation extraction): `04_knowledgegraph_dashboard/extract_entities_relations.py`
* Path B (Graph build + coherence metrics): `04_knowledgegraph_dashboard/build_knowledge_graph.py`
* Path C (Search/exploration): `04_knowledgegraph_dashboard/graph_search.py`
* Path D (Dashboard): `app/kg_dashboard.py`
* Path E (Extraction evaluation): `eval/evaluate_kg_extraction.py`

### `04_knowledgegraph_dashboard/extract_entities_relations.py`

Purpose: Turn a raw transcript into candidate `(subject, relation, object)` knowledge-graph triples.

What it does:

* Reuses `enrich_transcript()` and `select_candidate_windows()` from `02_causal_modeling/extract_variables.py` unmodified — same regex prefilter and windowing around blocker/decision/next_action utterances.
* `extract_kg_triples()` — sends each window to the LLM (reuses `MeetingSummarizer` and its disk cache) and asks for triples over a controlled entity-type vocabulary (`person`/`decision`/`issue`/`task`/`topic`) and relation vocabulary (`makes_decision`/`raises_issue`/`blocks`/`causes`/`assigned_to`/`owns`/`discusses`/`resolves`/`depends_on`/`related_to`).
* `resolve_person_alias()` collapses person mentions onto the transcript's known speaker names (e.g. "Aiko" → "Aiko - PM") for lightweight entity linking.
* `extract_topic_timeline()` — zero-LLM-call per-utterance topic tags, reused for the dashboard's topic-drift view.

Used by: `eval/evaluate_kg_extraction.py`, `app/kg_dashboard.py` (indirectly, via pre-extracted sample triples)

### `04_knowledgegraph_dashboard/build_knowledge_graph.py`

Purpose: Merge triples from many meetings into one entity-resolved knowledge graph.

What it does:

* `node_key()`: normalizes free-text entity mentions (lowercase, strip punctuation/leading article) scoped by entity type, so the graph doesn't fragment into one node per unique phrasing.
* `build_graph()`: builds a `networkx.MultiDiGraph` (multiple relation types can connect the same two entities); edge weight = observation count, each edge keeps its source transcript evidence (quote + timestamp).
* `graph_coherence_metrics()`: density, connected components, modularity/community count, average inter-community conductance — computable without gold labels.
* `plot_graph()`: matplotlib visualization colored by entity type.

Used by: `04_knowledgegraph_dashboard/graph_search.py`, `app/kg_dashboard.py`, `eval/evaluate_kg_extraction.py`

### `04_knowledgegraph_dashboard/graph_search.py`

Purpose: Search and explore the merged knowledge graph.

What it does:

* `keyword_search()` — substring match over node labels + evidence quotes, zero API calls.
* `semantic_search()` — OpenAI embeddings (`text-embedding-3-small`), disk-cached, cosine-ranked; requires live API credits for cache misses.
* `find_path()` / `format_path_evidence()` — shortest path between two entities with the real transcript quotes behind each hop.
* `ego_subgraph()` — local neighborhood around an entity, for dashboard exploration.

### `app/kg_dashboard.py`

Purpose: Gradio "Knowledge Graph Explorer" web UI (port 7862).

What it does:

* Loads the graph once at startup from `data/sample_kg_triples/*.json` and computes topic timelines + a discussion (speaker co-occurrence) network from `data/synthetic_transcripts/*.json`.
* **Explore tab**: keyword/semantic search → matching entities + an ego-subgraph plot of the top match.
* **Network tab**: full graph plot colored by entity type, coherence metrics, and the discussion network (team dynamics).
* **Topic Drift tab**: per-meeting topic sequence, or topic mix across all meetings ordered by meeting id.

### `eval/evaluate_kg_extraction.py`

Purpose: Score the entity/relation extraction pipeline against 10 hand-labeled synthetic meetings (`data/synthetic_transcripts/*.kg_labels.json`).

What it does:

* Runs `extract_kg_triples()` unmodified, computes entity-linking precision/recall (set overlap on node keys) and relation/fact extraction precision/recall/F1 (LLM judge `gpt-4o-mini` + one-to-one maximum bipartite matching — reuses `parse_judge_response()`/`max_bipartite_matching()` from `eval/evaluate_causal_extraction.py` unmodified).
* Also computes graph coherence metrics on the pooled predicted-triple graph.
* Writes a markdown report and raw JSON to `eval/results/kg_extraction/`.

See `docs/phase4/architecture.md`, `data-design.md`, `evaluation-strategy.md`, and `pipeline-flow-and-results.md` for the full design, evaluation results, and known limitations.

## Phase 5: Integration + Open-Sourcing

Bundles Phases 1-4 into a single architecture and fixes packaging so the repo
is actually reproducible as open source.

* Path A (Cross-linking): `05_integration/cross_link.py`
* Path B (Orchestration): `05_integration/pipeline.py`
* Path C (Unified dashboard): `app/dashboard.py`

### `05_integration/cross_link.py`

Purpose: Connect Phase 2's causal events to Phase 4's KG triples — both are extracted from the same 10 synthetic transcripts (`data/synthetic_transcripts/`), keyed by the same `meeting_id`, but neither extraction pipeline knows about the other's output.

What it does:

* `link_causal_and_kg()` — scores every `(causal_event, kg_triple)` pair from the same meeting by word-token Jaccard overlap between `{cause, effect}` and `{subject_text, object_text, quote}`, with a bonus for an exact `timestamp` match. No embeddings, no extra API calls — fully offline and deterministic.
* Verified against real data: 69 cross-links found across all 10 meetings (e.g. meeting_01's causal event about the data-sharing agreement blocking the customer records module links to the matching KG triple, sharing timestamp `00:00:45`).

### `05_integration/pipeline.py`

Purpose: CLI orchestrator that runs Phase 1 (summary) + Phase 2 (causal events) + Phase 4 (KG triples) over a transcript directory and cross-links the results into one JSON per meeting (`output/integrated/<meeting_id>.integrated.json`).

* `--skip-llm` (default): reuses existing `output/causal_events/`/`output/kg_triples/`, zero API calls.
* `--with-llm`: extracts fresh + generates a live summary.

### `app/dashboard.py`

Purpose: Single entry point bundling all three existing dashboards (Phase 1 Summarizer, Phase 2 Causal What-If, Phase 4 Knowledge Graph Explorer) as tabs, plus a new **Integrated View** tab showing the Phase 2/4 cross-links per meeting (with an optional live-summary button). Single port: **7860**. Reuses the existing dashboards' Gradio `Blocks` objects unmodified.

See `docs/phase5/architecture.md` for the full design.

### Open-sourcing / reproducibility decisions

* **License**: MIT (`LICENSE`), citation metadata in `CITATION.cff`.
* **Fixed a `.gitignore` bug**: `Dockerfile`, `requirements.txt`, and `scripts/*.sh` were previously excluded from git entirely (never committed to any branch). They're now tracked.
* **Pinned dependencies**: `requirements.txt` now pins exact versions instead of unpinned package names.
* **`eval/results/`, `output/causal_events/`, and `output/kg_triples/` stay gitignored** — these are LLM-derived, regenerable artifacts, not source, and are kept out of the repo. See "Note on reproducing the Integrated View / evaluation numbers" above for the exact commands to regenerate them from a fresh clone.
* **`.env.example`** added as a template; `.env` itself stays gitignored.