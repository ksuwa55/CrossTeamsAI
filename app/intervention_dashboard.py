import os
import sys

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
CAUSAL_PKG = os.path.join(REPO_ROOT, "02_causal_modeling")
sys.path.insert(0, CAUSAL_PKG)

from build_causal_graph import load_events, build_graph
from simulate_intervention import all_meeting_ids, build_occurrence_data, build_scm, simulate_intervention

DEFAULT_EVENTS_GLOB = os.path.join(REPO_ROOT, "data", "sample_causal_events", "*.json")


def _load_model(events_glob: str):
    events = load_events([events_glob])
    graph = build_graph(events)
    data = build_occurrence_data(events, graph, meeting_ids=all_meeting_ids([events_glob]))
    scm = build_scm(graph, data)
    return graph, scm


_graph, _scm = _load_model(DEFAULT_EVENTS_GLOB)
_cause_options = [n for n in _graph.nodes() if _graph.out_degree(n) > 0]
_outcome_options = [n for n in _graph.nodes() if _graph.in_degree(n) > 0]


def _format_path_evidence(graph: nx.DiGraph, path, max_examples_per_edge: int = 2) -> str:
    """Real transcript excerpts behind each hop of cause -> outcome, so a
    non-technical PM can see *why* the DAG claims this link without reading
    code or JSON."""
    sections = []
    for u, v in zip(path[:-1], path[1:]):
        edge = graph[u][v]
        examples = sorted(edge["evidence"], key=lambda e: e.get("timestamp", ""))[:max_examples_per_edge]
        quotes = "\n".join(
            f'- _{ex.get("meeting_id", "unknown meeting")} ({ex.get("timestamp", "n/a")})_: '
            f'"{ex["cause"]}" → "{ex["effect"]}"'
            for ex in examples
        )
        sections.append(f"**{u} → {v}** (seen {edge['weight']}x)\n{quotes}")
    return "\n\n".join(sections)


def run_simulation(cause: str, outcome: str, mitigate: bool, num_samples: int):
    if cause == outcome:
        return "Cause and outcome must be different nodes.", None, ""

    if not nx.has_path(_graph, cause, outcome):
        return f"No causal path from '{cause}' to '{outcome}' in the current DAG.", None, ""

    path = nx.shortest_path(_graph, cause, outcome)
    evidence_md = _format_path_evidence(_graph, path)

    result = simulate_intervention(_scm, cause_node=cause, outcome_node=outcome, num_samples=int(num_samples))
    current_prob = result["intervened_probability"] if mitigate else result["baseline_probability"]

    fig, ax = plt.subplots(figsize=(4, 3))
    labels = ["Baseline", f"If {cause}\nis mitigated"]
    values = [result["baseline_probability"], result["intervened_probability"]]
    colors = ["#8f8f8f", "#2f9e44"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"P({outcome} = 1)")
    for bar in bars:
        ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()

    scenario_label = "with mitigation" if mitigate else "no mitigation (current state)"
    summary = (
        f"Scenario: {scenario_label}\n"
        f"P({outcome}) right now: {current_prob:.2f}\n\n"
        f"Baseline P({outcome}):                    {result['baseline_probability']:.2f}\n"
        f"P({outcome}) if we mitigate {cause}:  {result['intervened_probability']:.2f}\n"
        f"Predicted reduction:                      {result['absolute_reduction']:.2f} "
        f"({result['relative_reduction_pct']:.0f}% relative)"
    )
    return summary, fig, evidence_md


with gr.Blocks(title="Causal Intervention Simulator") as demo:
    gr.Markdown("## What-if: Simulate Mitigating a Root Cause")
    gr.Markdown(
        "Pick a cause and an outcome from the causal DAG built by `build_causal_graph.py`, "
        "toggle mitigation, and see the predicted shift in the outcome's probability "
        "(computed with do-calculus via `dowhy.gcm`)."
    )
    with gr.Row():
        cause = gr.Dropdown(_cause_options, value=_cause_options[0] if _cause_options else None, label="Cause to mitigate")
        outcome = gr.Dropdown(_outcome_options, value=_outcome_options[-1] if _outcome_options else None, label="Outcome to track")
    mitigate = gr.Checkbox(value=False, label="Apply mitigation (assume the cause is resolved)")
    num_samples = gr.Slider(500, 5000, value=3000, step=500, label="Simulation samples")
    run_btn = gr.Button("Simulate", variant="primary")

    out_text = gr.Textbox(label="Result", lines=6)
    out_plot = gr.Plot(label="Probability shift")
    out_evidence = gr.Markdown(label="Why (evidence from transcripts)")

    run_btn.click(run_simulation, inputs=[cause, outcome, mitigate, num_samples], outputs=[out_text, out_plot, out_evidence])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
