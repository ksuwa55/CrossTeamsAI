# Gaps Toward an Academic-Quality Deliverable

This project (Phase 1: Meeting Summarization, Phase 2: Causal Modeling of Project
Bottlenecks) is implementation-complete and documented to an engineering standard
(architecture, data design, evaluation strategy, and results for both phases). The
items below are what would still be required to bring this to the standard of a
publishable academic deliverable. They are intentionally left out of scope for this
iteration and are recorded here as a reference for future work.

## 1. Literature Review / Theoretical Framing

- No related-work section exists. The current docs are engineering documentation
  (what was built, how it works) rather than academic framing (how it relates to
  and differs from prior work).
- Needs an explicit positioning against the literature cited in the original
  research proposal, e.g.:
  - Recursive/hierarchical summarization (Zhong et al. 2021 QMSum; Krishna et al.
    2021 abstractive summarization with LLMs) — how does the map-reduce pipeline
    in `01_summarization` relate to and differ from these baselines?
  - Causal representation learning and LLM-based causal reasoning (Schölkopf et
    al. 2021; Kiciman et al. 2025) — what does this pipeline's LLM-as-causal-
    extractor approach confirm, contradict, or add relative to that work?
  - Graphical causal inference (Sharma et al. 2022 DoWhy; Blöbaum et al. 2024
    DoWhy-GCM) — how does the SCM/do-calculus design in
    `02_causal_modeling/simulate_intervention.py` fit into that toolchain, and
    what limitations are specific to this application vs. inherited from DoWhy
    itself?
  - Knowledge graph embedding (Wang et al. 2017) — relevant if/when Phase 4
    (knowledge graph builder) is implemented.
- No explicit statement of novelty: what, if anything, is new here relative to
  existing engineering-intelligence / root-cause-analysis tools (e.g. LinearB,
  Jellyfish, Dynatrace Davis AI) versus academic causal-NLP work.

## 2. Evaluation Scale and Statistical Rigor

- Phase 2's evaluation set is 10 synthetic meetings / 19 gold cause-effect pairs —
  enough to characterize failure modes directionally, but too small for
  confidence intervals or claims of generalization.
- Phase 1's QMSum evaluation is on a subset of meetings (sampled for cost
  control), not the full test split.
- No evaluation on real (non-synthetic) meeting transcripts for either phase.
- No cross-dataset evaluation (e.g. testing Phase 2's extractor on AMI/ICSI
  Corpus data, as the original proposal specified as a Phase 1 benchmark).

## 3. Missing Baselines (Phase 2)

- Phase 1 has baseline comparisons (`eval/baselines.py`: lead-n, TF-IDF,
  random) that contextualize how much the LLM summarizer adds over simple
  extractive methods.
- Phase 2 has no equivalent. There is nothing to compare the LLM-based
  cause/effect extractor against — e.g. a keyword co-occurrence baseline, a
  rule-based baseline, or a simpler prompting strategy — so the pipeline's
  Precision/Recall/F1 numbers cannot currently support a claim that using an
  LLM extractor is better than a simpler method.

## 4. Annotation Reliability

- All gold labels in `data/synthetic_transcripts/*.labels.json` were written by
  a single annotator (non-native English speaker, self-reported).
- No inter-annotator agreement (e.g. Cohen's kappa) has been measured. A
  second annotator's independent labels on the same transcripts, compared for
  agreement, would be needed to treat the gold set as a reliable ground truth
  rather than one person's judgment calls.
- Related: no explicit, written labeling guideline/rubric exists (e.g. what
  counts as "the cause" vs. "an intermediate effect" in a causal chain — see
  the granularity-mismatch finding in
  `docs/phase2/pipeline-flow-and-results.md`). Without a rubric, a second
  annotator's agreement score would be hard to interpret.

## 5. Discussion Section (Connecting Results to Research Questions)

- The original research proposal's Phase 1 RQ is: "What is the capability of
  recursive summarization with LLMs in generating causal arguments?" Phase 2's
  RQ is: "How to extract and use knowledge effectively? What is the capability
  of LLMs in generating causal arguments?"
- Current docs report results (ROUGE/BERTScore for Phase 1; Precision/Recall/F1
  and false-negative classification for Phase 2) but do not explicitly connect
  those results back to these research questions — e.g. what the 22%
  precision / 57.9% recall / granularity-mismatch finding actually implies
  about LLMs' capability to generate causal arguments, stated as a claim
  rather than left as a raw number.

## 6. Deferred From Phase 2 Results (Cross-Reference)

The following were already identified as known limitations in
`docs/phase2/evaluation-strategy.md` and `pipeline-flow-and-results.md`, and are
listed here again only for completeness of this gap list:

- No refutation/robustness testing of the DoWhy causal estimates
  (`refute_estimate`).
- No comparison against Bayesian networks, despite the original proposal
  specifying "Bayesian networks + DoWhy/CausalML" rather than DoWhy alone.
- No expert review (Agile coaches / PMs) of the causal graph's face validity.
- No integration between Phase 1 (summarization) and Phase 2 (causal
  querying of past discussions), as specified in the original proposal.
- Two meetings (`meeting_03_missing_resource`, `meeting_05_decision_stagnation`)
  scored 0% precision/recall; left untuned by scope decision.

## 7. Human Evaluation (Phase 1)

- `eval/human_eval/google_form_blueprint.md` is designed but not administered.
  The original plan assumed in-person recruitment of undergraduates after
  relocating to the UK; that plan is no longer applicable, and self-sourced
  recruitment was out of scope for this timeboxed round.