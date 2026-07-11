import argparse
import glob
import os
import sys
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd
from dowhy import gcm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_causal_graph import load_events, build_graph, build_occurrence_table


def all_meeting_ids(input_patterns: List[str]) -> List[str]:
    paths = []
    for pattern in input_patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def build_occurrence_data(events: List[Dict], graph: nx.DiGraph, meeting_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """Binary occurrence table, one row per meeting, string-typed so dowhy.gcm
    treats every node as categorical rather than continuous."""
    df = build_occurrence_table(events, graph)
    if meeting_ids:
        missing = set(meeting_ids) - set(df["meeting_id"])
        if missing:
            zero_rows = [{**{c: 0 for c in df.columns if c != "meeting_id"}, "meeting_id": mid} for mid in missing]
            df = pd.concat([df, pd.DataFrame(zero_rows)], ignore_index=True)
    return df.drop(columns=["meeting_id"]).astype(str)


def build_scm(graph: nx.DiGraph, data: pd.DataFrame) -> gcm.StructuralCausalModel:
    """Root nodes get an empirical (observed-frequency) distribution; every
    downstream node gets a logistic-regression mechanism over its parents, so
    a hard intervention on a parent actually propagates through do-calculus."""
    scm = gcm.StructuralCausalModel(graph)
    for node in graph.nodes():
        if list(graph.predecessors(node)):
            scm.set_causal_mechanism(node, gcm.ClassifierFCM(gcm.ml.create_logistic_regression_classifier()))
        else:
            scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
    gcm.fit(scm, data)
    return scm


def simulate_intervention(
    scm: gcm.StructuralCausalModel,
    cause_node: str,
    outcome_node: str,
    mitigated_value: str = "0",
    num_samples: int = 3000,
) -> Dict[str, float]:
    baseline_samples = gcm.interventional_samples(scm, {}, num_samples_to_draw=num_samples)
    intervened_samples = gcm.interventional_samples(
        scm, {cause_node: lambda _: mitigated_value}, num_samples_to_draw=num_samples
    )

    baseline_prob = float((baseline_samples[outcome_node] == "1").mean())
    intervened_prob = float((intervened_samples[outcome_node] == "1").mean())
    reduction = baseline_prob - intervened_prob

    return {
        "cause": cause_node,
        "outcome": outcome_node,
        "baseline_probability": baseline_prob,
        "intervened_probability": intervened_prob,
        "absolute_reduction": reduction,
        "relative_reduction_pct": (reduction / baseline_prob * 100) if baseline_prob > 0 else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="One or more causal_events.json files or glob patterns")
    parser.add_argument("--cause", required=True, help="Node to mitigate, e.g. blocked_dependency")
    parser.add_argument("--outcome", required=True, help="Node to track, e.g. decision_stagnation")
    parser.add_argument("--num-samples", type=int, default=3000)
    args = parser.parse_args()

    events = load_events(args.input)
    graph = build_graph(events)
    data = build_occurrence_data(events, graph, meeting_ids=all_meeting_ids(args.input))
    scm = build_scm(graph, data)

    result = simulate_intervention(scm, args.cause, args.outcome, num_samples=args.num_samples)
    print(
        f"\nP({result['outcome']}) baseline:    {result['baseline_probability']:.3f}\n"
        f"P({result['outcome']}) if we mitigate {result['cause']}: {result['intervened_probability']:.3f}\n"
        f"Reduction: {result['absolute_reduction']:.3f} "
        f"({result['relative_reduction_pct']:.0f}% relative)"
    )
