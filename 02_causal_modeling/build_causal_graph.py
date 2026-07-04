import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


# Ordered rules: canonicalize each raw cause/effect phrase into a small, fixed
# vocabulary of node categories so the DAG doesn't fragment into one node per
# unique LLM phrasing. First matching rule wins, so put more specific ones first.
CANONICAL_RULES: List[Tuple[str, List[str]]] = [
    ("decision_stagnation", ["decision stagnation", "stagnat", "no decision", "pending decision", "undecided", "no consensus", "couldn't decide", "stalled"]),
    ("blocked_dependency", ["waiting for", "waiting on", "blocked on", "pending approval", "sign-off", "approval", "blocker", "blocked", "can't", "dependency issue"]),
    ("ambiguous_requirement", ["ambiguous", "unclear requirement", "unclear scope", "spec unclear"]),
    ("scope_change", ["scope change", "new requirement", "changed requirement", "re-scope"]),
    ("missing_resource", ["understaffed", "no developer available", "resource constraint", "short-staffed"]),
    ("qa_delay", ["qa delay", "testing delay", "qa backlog", "quality assurance"]),
    ("integration_delay", ["integration"]),
    ("missed_deadline", ["deadline", "missed", "slipped", "push the launch", "delayed launch", "late delivery"]),
]

def canonicalize_node(phrase: str) -> str:
    text = phrase.lower()
    for canonical, keywords in CANONICAL_RULES:
        if any(kw in text for kw in keywords):
            return canonical
    return "other:" + text.strip()[:40]

def load_events(input_patterns: List[str]) -> List[Dict]:
    paths = []
    for pattern in input_patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])

    events = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        default_meeting_id = os.path.splitext(os.path.basename(path))[0]
        for e in data:
            events.append({
                "cause": e["cause"],
                "effect": e["effect"],
                "timestamp": e.get("timestamp", ""),
                "meeting_id": e.get("meeting_id", default_meeting_id),
            })
    return events

def build_graph(events: List[Dict]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for e in events:
        cause_node = canonicalize_node(e["cause"])
        effect_node = canonicalize_node(e["effect"])
        if cause_node == effect_node:
            continue
        graph.add_node(cause_node)
        graph.add_node(effect_node)
        if graph.has_edge(cause_node, effect_node):
            graph[cause_node][effect_node]["weight"] += 1
            graph[cause_node][effect_node]["evidence"].append(e)
        else:
            graph.add_edge(cause_node, effect_node, weight=1, evidence=[e])
    return graph

def build_occurrence_table(events: List[Dict], graph: nx.DiGraph) -> pd.DataFrame:
    """One row per meeting, one binary column per DAG node: did that node's
    cause/effect show up in that meeting? Feeds dowhy.CausalModel(data=...)."""
    nodes = list(graph.nodes())
    per_meeting = defaultdict(lambda: {n: 0 for n in nodes})
    for e in events:
        cause_node = canonicalize_node(e["cause"])
        effect_node = canonicalize_node(e["effect"])
        mid = e["meeting_id"]
        per_meeting[mid][cause_node] = 1
        per_meeting[mid][effect_node] = 1
    df = pd.DataFrame.from_dict(per_meeting, orient="index")
    df.index.name = "meeting_id"
    return df.reset_index()

def save_graph_json(graph: nx.DiGraph, path: str):
    data = {
        "nodes": list(graph.nodes()),
        "edges": [
            {"source": u, "target": v, "weight": d["weight"], "evidence": d["evidence"]}
            for u, v, d in graph.edges(data=True)
        ],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved graph JSON to {path}")

def save_graph_gml(graph: nx.DiGraph, path: str):
    export = nx.DiGraph()
    export.add_nodes_from(graph.nodes())
    export.add_edges_from(graph.edges())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nx.write_gml(export, path)
    print(f"Saved graph GML to {path}")

def plot_graph(graph: nx.DiGraph, path: str):
    plt.figure(figsize=(10, 7))
    pos = nx.shell_layout(graph)
    weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    node_size = 2600
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color="#f2b134")
    nx.draw_networkx_labels(graph, pos, font_size=8)
    nx.draw_networkx_edges(
        graph, pos, node_size=node_size, width=[1 + w for w in weights],
        arrowstyle="-|>", arrowsize=18, edge_color="#333333",
    )
    edge_labels = {(u, v): d["weight"] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved DAG visualization to {path}")

def run_dowhy_identification(df: pd.DataFrame, graph: nx.DiGraph, treatment: str, outcome: str):
    from dowhy import CausalModel

    model = CausalModel(data=df, treatment=treatment, outcome=outcome, graph=graph)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)

    try:
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
        )
        print(f"\nEstimated effect of '{treatment}' on '{outcome}': {estimate.value:.3f}")
    except Exception as exc:
        print(f"\n[skip] Could not estimate effect with this sample size/shape: {exc}")

    return model, identified_estimand

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="One or more causal_events.json files or glob patterns")
    parser.add_argument("--graph-json", default="output/causal_graph.json")
    parser.add_argument("--graph-gml", default="output/causal_graph.gml")
    parser.add_argument("--plot", default="output/causal_dag.png")
    parser.add_argument("--treatment", default=None, help="Node name to sanity-check with DoWhy's identify/estimate (e.g. blocked_dependency)")
    parser.add_argument("--outcome", default=None, help="Node name to sanity-check with DoWhy's identify/estimate (e.g. decision_stagnation)")
    args = parser.parse_args()

    events = load_events(args.input)
    if not events:
        raise SystemExit("No causal events found in the given --input path(s)")

    graph = build_graph(events)
    save_graph_json(graph, args.graph_json)
    save_graph_gml(graph, args.graph_gml)
    plot_graph(graph, args.plot)

    print(f"\n{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    for u, v, d in graph.edges(data=True):
        print(f"  {u} -> {v}  (seen {d['weight']}x)")

    if args.treatment and args.outcome:
        df = build_occurrence_table(events, graph)
        run_dowhy_identification(df, graph, args.treatment, args.outcome)
