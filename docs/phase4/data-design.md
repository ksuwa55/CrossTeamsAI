# Phase 4: Knowledge Graph + Dashboard — Data Design

## Entity/Relation Schema

**Entity types** (controlled vocabulary, one per node):

| Type | Meaning | Example |
|---|---|---|
| `person` | A meeting participant | "Aiko - PM" |
| `decision` | A decision made in a meeting | "get written confirmation from client before coding" |
| `issue` | A blocker, risk, or open problem | "QA is understaffed with one person covering all testing" |
| `task` | A piece of work someone owns or is assigned | "customer records module" |
| `topic` | A discussion subject (used for topic-drift only, not extracted as triples) | "testing schedule" |

**Relations** (controlled vocabulary, one per edge):

`makes_decision`, `raises_issue`, `blocks`, `causes`, `assigned_to`, `owns`, `discusses`, `resolves`, `depends_on`, `related_to`

The extractor (`extract_entities_relations.py`) is prompted to only use these two vocabularies; anything else is dropped by `parse_kg_response()`. This keeps the merged graph coherent across meetings instead of fragmenting into one node/edge type per unique LLM phrasing.

## Triple Record

```json
{
  "subject_text": "Aiko - PM",
  "subject_type": "person",
  "relation": "raises_issue",
  "object_text": "legal sign-off pending on data-sharing agreement",
  "object_type": "issue",
  "timestamp": "00:00:20",
  "quote": "Legal hasn't signed off yet.",
  "meeting_id": "meeting_01_signoff_blocker"
}
```

`quote` is the grounding utterance — the same evidence-traceability idea as Phase 2's per-event `timestamp`, but explicit text rather than just a time reference, so the dashboard can show *why* an edge exists.

## Entity Resolution / Node Identity

`node_key(entity_type, text) = f"{entity_type}:{normalize_entity(text)}"`, where `normalize_entity()` lowercases, strips punctuation, drops a leading article ("the"/"a"/"an"), and collapses whitespace.

For `person` entities specifically, `resolve_person_alias()` runs first: it matches the extracted mention against the transcript's own `speaker` list (case-insensitive substring in either direction) and rewrites the mention to the matched speaker's full name before normalization. This means "Aiko", "aiko", and "Aiko - PM" all resolve to the same node — a lightweight, deterministic stand-in for coreference resolution that works because the source transcripts already carry a speaker label per utterance.

This is a heuristic, not a learned entity linker — it will not resolve a person referred to only by a pronoun ("she said...") or a nickname that isn't a substring of their speaker label. See Known Limitations in `evaluation-strategy.md`.

## Corpus (reused from Phase 2)

**`data/synthetic_transcripts/`** — the same 10 synthetic meetings Phase 2 already uses (signoff blocker, ambiguous requirement, missing resource, API dependency, decision stagnation, timezone communication, scope change, QA delay/deadline, security ambiguity, resource reprioritization). Reusing the corpus means Phase 4's extraction can be directly compared against the same source material Phase 2's causal extraction was evaluated on.

**New for Phase 4**: `data/synthetic_transcripts/meeting_XX_*.kg_labels.json` — hand-authored gold triples per meeting (one annotator, same caveat as Phase 2's `.labels.json`; see `docs/gaps_toward_academic_deliverable.md`). ~4 triples per meeting, covering at least one `person`→`issue`/`decision`/`task` triple and one `issue`→`issue` or `issue`→`task` triple, so both entity-linking and relation-extraction metrics have real gold to score against.

## Dashboard Seed Data

**`data/sample_kg_triples/meeting_0X.json`** — hand-authored, not pipeline output. Same role as Phase 2's `data/sample_causal_events/*.json`: lets `app/kg_dashboard.py` start up and be demoed without needing a live OpenAI call or API credits. One file per synthetic meeting, matching the extraction output schema (including `quote`/`meeting_id`) so the dashboard code path is identical to what real extraction output would look like.

## Topic-Drift Data

No separate stored file — `app/kg_dashboard.py` computes topic timelines at startup by calling `enrich_transcript()` (Phase 2's regex prefilter, already tags each utterance with a `topic`) and `extract_topic_timeline()` directly over `data/synthetic_transcripts/*.json`. Zero LLM calls, so this view is always available regardless of API credit state.

## Output Artifacts (pipeline runs)

- `output/kg_triples/<meeting_id>.kg_triples.json` — real extraction output per meeting
- `output/kg_graph.json`, `output/kg_graph.png` — merged graph + visualization
- `eval/results/kg_extraction/kg_extraction_report.md`, `kg_extraction_raw.json` — evaluation report
