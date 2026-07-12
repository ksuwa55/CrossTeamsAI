No automated change-tracking platform is used; this section is maintained
manually as of this iteration.

| Change | Notes |
|---|---|
| Regex-only prefilter (`extract_variables.py`) | Initial state on `main` |
| LLM-based extraction, DAG builder, SCM/DoWhy simulator, Gradio dashboard | Developed on a feature branch |
| Feature branch merged into `main` | Brings `02_causal_modeling/build_causal_graph.py`, `simulate_intervention.py`, `app/intervention_dashboard.py` into the tracked repo |
| Evaluation harness added | `eval/evaluate_causal_extraction.py`, `eval/inspect_false_negatives.py`, plus the 10 synthetic transcripts + gold labels |
| `docs/phase2/` four-document set added | architecture, data-design, evaluation-strategy, pipeline-flow-and-results |
| README updated | Phase 2 section added, mirroring Phase 1's documentation style |
| `gaps_toward_academic_deliverable.md` added | Cross-phase record of what remains for academic-quality rigor |
| Migrated to `03_AI_documentation/generate_model_card.py` | This card is now rendered from `components/causal_modeling/` instead of hand-maintained |
