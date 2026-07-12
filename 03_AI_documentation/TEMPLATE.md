# AI Documentation Template

This is the shared model-card format for every component in this project
(`docs/phase3/*.md`). It follows the four documentation-requirement categories
from Königstorfer & Thalmann (2022), *AI Documentation: A Path to
Accountability*, Journal of Responsible Technology, 11:100043 — **Model Design
Choices, Data Characteristics, Evaluation Metrics, System Modification Log** —
bookended by an Intended Use section and a Known Limitations & Risks section
from the Model Cards framework (Mitchell et al., 2019), plus the
fairness/accountability/transparency framing from *"Software documentation is
not enough! Requirements for the documentation of AI"*.

`docs/phase3/causal_modeling.md` is the reference instance of this format,
written by hand first; `generate_model_card.py` renders every card
(including that one) from this template plus a per-component config so the
format stays identical across phases and the parts that can drift from the
code (metrics, provenance) can't.

## Placeholder tokens

Rendered by `generate_model_card.py` from `components/<name>/`:

| Token | Source | Why here |
|---|---|---|
| `{{TITLE}}` | `config.json: title` | Card heading |
| `{{CITATION_BLOCK}}` | Auto: fixed boilerplate + `config.json: files_documented` | Keeps the citation/framing wording identical across every card |
| `{{INTENDED_USE}}` | `intended_use.md` (human-authored) | What counts as valid use is a judgment call, not derivable from code |
| `{{DESIGN_CHOICES}}` | `design_choices.md` (human-authored) | Design rationale ("why X over Y") only exists in the author's head |
| `{{DATA_CHARACTERISTICS}}` | `data_characteristics.md` (human-authored) | Dataset provenance/bias notes aren't inferable from files alone |
| `{{EVAL_METRICS_TABLE}}` | Auto: `config.json: metrics_files` read from `eval/results/**` | The exact numbers a hand-maintained doc is most likely to go stale on |
| `{{EVAL_METRICS_NARRATIVE}}` | `eval_metrics_narrative.md` (human-authored) | Interpreting *why* a number looks the way it does is analysis, not extraction |
| `{{MODIFICATION_LOG}}` | `modification_log.md` (human-authored) | No automated change-tracking platform is wired up (see below) |
| `{{KNOWN_LIMITATIONS}}` | `known_limitations.md` (human-authored) | Known-unknowns are a judgment call the codebase can't self-report |
| `{{FOOTER}}` | Auto: generation timestamp + git commit | Traceability: which commit this exact card describes |

## Document skeleton

```markdown
# Model Card: {{TITLE}}

{{CITATION_BLOCK}}

## Intended Use

{{INTENDED_USE}}

## 1. Model Design Choices

{{DESIGN_CHOICES}}

## 2. Data Characteristics

{{DATA_CHARACTERISTICS}}

## 3. Evaluation Metrics

{{EVAL_METRICS_TABLE}}

{{EVAL_METRICS_NARRATIVE}}

## 4. System Modification Log

{{MODIFICATION_LOG}}

## Known Limitations & Risks

{{KNOWN_LIMITATIONS}}

---
{{FOOTER}}
```

## Adding a new component

1. `mkdir 03_AI_documentation/components/<name>`
2. Add `config.json` (`title`, `files_documented`, `output_path`, `metrics_files`)
3. Write the five fragment files (`intended_use.md`, `design_choices.md`,
   `data_characteristics.md`, `eval_metrics_narrative.md`,
   `modification_log.md`, `known_limitations.md`) — plain Markdown body text,
   no top-level heading (the template supplies the heading).
4. `python 03_AI_documentation/generate_model_card.py --component <name>`

`metrics_files` entries look like:

```json
{
  "label": "Row label shown in the table",
  "path": "eval/results/<file>.json",
  "fields": {"Column header": "dotted.path.into.the.json", "...": "..."}
}
```

Each entry becomes one table row; `fields` are read via dotted-key lookup
(e.g. `"metrics.rouge1.mean"` or `"aggregate.precision"`) so the same table
renderer works for any results schema already in `eval/results/`.
