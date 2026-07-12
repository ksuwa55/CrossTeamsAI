- **Evaluation dataset:** 10 synthetic meeting transcripts
  (`data/synthetic_transcripts/`), each authored to instantiate one labeled
  failure mode: signoff blocker, ambiguous requirement, missing resource, API
  dependency, decision stagnation, timezone/cross-border communication, scope
  change, QA delay/deadline, security ambiguity, resource reprioritization —
  chosen to cover the 8-category canonical node vocabulary above, plus one
  cross-border-specific scenario reflecting the original research proposal's
  focus on cross-border Agile teams.
- **Gold labels:** 19 hand-written cause/effect pairs across the 10 meetings
  (1–3 per meeting), in `*.labels.json`.
- **Known bias:** all gold labels were written by a single annotator (a
  non-native English speaker, self-reported). No second annotator or
  inter-annotator agreement score (e.g. Cohen's kappa) exists. No written
  labeling rubric was used to standardize what counts as "the cause" vs. an
  intermediate effect in a causal chain — see the granularity-mismatch
  finding under "Evaluation Metrics."
- **Dashboard seed data:** `data/sample_causal_events/` — pre-extracted
  events used only to populate the Gradio dashboard at startup; distinct from
  the labeled evaluation set and not itself evaluated against gold labels.
- **Real transcripts:** none used. All development and evaluation data is
  synthetic.
