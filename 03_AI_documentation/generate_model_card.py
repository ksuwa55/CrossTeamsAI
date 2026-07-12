"""
Renders an AI documentation card (model card) for any component under
`03_AI_documentation/components/<name>/`, following the shared format in
`03_AI_documentation/TEMPLATE.md`.

Why this split: design rationale, intended use, and known limitations are
judgment calls that only a human can write -- they live in per-component
Markdown fragments. Evaluation numbers are the part a hand-maintained doc
goes stale on the moment someone reruns an experiment -- those are read
straight from `eval/results/**` on every render, keyed by a dotted path
declared in the component's `config.json`, so they can't drift from the
actual result files.

Usage:
    python 03_AI_documentation/generate_model_card.py --component causal_modeling
    python 03_AI_documentation/generate_model_card.py --component summarization
    python 03_AI_documentation/generate_model_card.py --all
"""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
COMPONENTS_DIR = os.path.join(HERE, "components")
TEMPLATE_SKELETON = """# Model Card: {TITLE}

{CITATION_BLOCK}

## Intended Use

{INTENDED_USE}

## 1. Model Design Choices

{DESIGN_CHOICES}

## 2. Data Characteristics

{DATA_CHARACTERISTICS}

## 3. Evaluation Metrics

{EVAL_METRICS_TABLE}

{EVAL_METRICS_NARRATIVE}

## 4. System Modification Log

{MODIFICATION_LOG}

## Known Limitations & Risks

{KNOWN_LIMITATIONS}

---
{FOOTER}
"""

FRAGMENT_FILES = [
    "intended_use.md",
    "design_choices.md",
    "data_characteristics.md",
    "eval_metrics_narrative.md",
    "modification_log.md",
    "known_limitations.md",
]

CITATION_BOILERPLATE = (
    "This card documents {files_list} following the four documentation-requirement "
    "categories identified in Königstorfer & Thalmann (2022), *AI Documentation: A "
    "Path to Accountability*, Journal of Responsible Technology, 11:100043 — "
    "supplemented by the Model Cards framework (Mitchell et al., 2019)."
)


def read_fragment(component_dir: str, filename: str) -> str:
    path = os.path.join(component_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required fragment: {path}\n"
            f"Every component needs all of: {', '.join(FRAGMENT_FILES)}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def dotted_get(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v * 100:.1f}%" if abs(v) <= 1 else f"{v:.2f}"
    if v is None:
        return "—"
    return str(v)


def build_metrics_table(metrics_files: list) -> str:
    if not metrics_files:
        return "_No evaluation results configured for this component._"

    columns = []
    for entry in metrics_files:
        for col in entry.get("fields", {}):
            if col not in columns:
                columns.append(col)

    header = "| Run | " + " | ".join(columns) + " |\n"
    header += "|---|" + "|".join(["---"] * len(columns)) + "|\n"

    rows = []
    for entry in metrics_files:
        result_path = os.path.join(REPO_ROOT, entry["path"])
        if not os.path.exists(result_path):
            values = ["_missing: " + entry["path"] + "_"] + ["—"] * (len(columns) - 1)
            rows.append(f"| {entry['label']} | " + " | ".join(values) + " |")
            continue
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cells = []
        for col in columns:
            key_path = entry.get("fields", {}).get(col)
            cells.append(fmt_value(dotted_get(data, key_path)) if key_path else "—")
        rows.append(f"| {entry['label']} | " + " | ".join(cells) + " |")

    return header + "\n".join(rows)


def get_git_info() -> dict:
    def run(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", REPO_ROOT, *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    return {
        "commit": run("rev-parse", "--short", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def render_component(name: str) -> str:
    component_dir = os.path.join(COMPONENTS_DIR, name)
    config_path = os.path.join(component_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No component '{name}' found (expected {config_path})")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    fragments = {
        "INTENDED_USE": read_fragment(component_dir, "intended_use.md"),
        "DESIGN_CHOICES": read_fragment(component_dir, "design_choices.md"),
        "DATA_CHARACTERISTICS": read_fragment(component_dir, "data_characteristics.md"),
        "EVAL_METRICS_NARRATIVE": read_fragment(component_dir, "eval_metrics_narrative.md"),
        "MODIFICATION_LOG": read_fragment(component_dir, "modification_log.md"),
        "KNOWN_LIMITATIONS": read_fragment(component_dir, "known_limitations.md"),
    }

    files_list = ", ".join(f"`{p}`" for p in config["files_documented"])
    citation_block = CITATION_BOILERPLATE.format(files_list=files_list)
    metrics_table = build_metrics_table(config.get("metrics_files", []))

    git = get_git_info()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    footer = (
        f"*Generated {generated_at} by `03_AI_documentation/generate_model_card.py` "
        f"from commit `{git['commit']}`{' (uncommitted changes present)' if git['dirty'] else ''} "
        f"on `{git['branch']}`. Do not hand-edit — edit `03_AI_documentation/components/{name}/` "
        f"and re-run the generator instead.*"
    )

    return TEMPLATE_SKELETON.format(
        TITLE=config["title"],
        CITATION_BLOCK=citation_block,
        EVAL_METRICS_TABLE=metrics_table,
        FOOTER=footer,
        **fragments,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--component", help="Name of the component under 03_AI_documentation/components/")
    ap.add_argument("--all", action="store_true", help="Render every component")
    args = ap.parse_args()

    if not args.component and not args.all:
        ap.error("pass --component <name> or --all")

    names = (
        sorted(
            d for d in os.listdir(COMPONENTS_DIR)
            if os.path.isdir(os.path.join(COMPONENTS_DIR, d))
        )
        if args.all
        else [args.component]
    )

    for name in names:
        component_dir = os.path.join(COMPONENTS_DIR, name)
        with open(os.path.join(component_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        doc = render_component(name)
        out_path = os.path.join(REPO_ROOT, config["output_path"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc)
        print(f"[{name}] wrote {out_path}")


if __name__ == "__main__":
    main()
