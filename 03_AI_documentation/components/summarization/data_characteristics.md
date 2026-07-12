- **Evaluation dataset: QMSum** (Query-based Multi-domain Meeting Summarization),
  `data/QMSum/data/ALL/test`.
  - 32 test meetings (academic meetings from AMI/ICSI, product meetings,
    committee meetings), each with 1 general query and multiple specific
    queries, all with human-written reference summaries (~32 general + 100+
    specific queries across the full test split).
  - Default evaluation runs sample a subset (`--sample_ratio`,
    `--max_meetings 10`, `--max_queries_per_meeting 2`) for API cost control —
    the instance counts in Section 3 reflect the sampled subset, not the full
    32-meeting split, unless a row says otherwise.
- **No training or fine-tuning data.** The summarizer is a frozen third-party
  API (OpenAI Chat Completions) called through prompts; QMSum data is used
  only for (a) the reference summaries evaluation is scored against, and (b)
  an optional one-shot example inserted into the prompt (`--few_shot on`).
- **No real (non-benchmark) meeting transcripts** have been used for
  development or evaluation — `data/zoom_transcript_sample.json` /
  `data/slack_transcript_sample.json` are demo inputs for the
  single-transcript path (`app/main.py`), not scored against references.
- **Known bias:** QMSum reference summaries were written by the dataset's
  original annotators; no additional review of reference quality was
  performed by this project.
