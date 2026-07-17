import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.cuts import conductance

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_entities_relations import node_key, normalize_entity  # noqa: E402
from graph_search import cosine_similarity, embed_texts  # noqa: E402


SEMANTIC_MERGE_TYPES: Tuple[str, ...] = ("issue", "decision", "task")
DEFAULT_SIMILARITY_THRESHOLD = 0.75


def cluster_by_similarity(embeddings: Sequence[List[float]], threshold: float) -> List[int]:
    """Single-linkage clustering (union-find) of items by cosine similarity of
    their `embeddings`: any pair at or above `threshold` joins the same
    cluster. Pure function, no I/O, so it's testable without an API key.
    Returns the cluster-root index (an id, not a canonical value) for each
    input position -- the shared low-level primitive behind both
    `cluster_mentions()` (predicted-vs-predicted merging, for the graph) and
    `eval.evaluate_kg_extraction`'s predicted-vs-gold semantic entity-linking
    matching."""
    n = len(embeddings)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if cosine_similarity(embeddings[i], embeddings[j]) >= threshold:
                union(i, j)

    return [find(i) for i in range(n)]


def cluster_mentions(texts: Sequence[str], embeddings: Sequence[List[float]], threshold: float) -> Dict[str, str]:
    """Clusters `texts` by cosine similarity of their `embeddings`
    (`cluster_by_similarity()`) and returns a mapping from each original text
    to its cluster's canonical text (the most frequent text in `texts`, ties
    broken by first occurrence, mirroring `resolve_person_alias()`'s
    "collapse onto one representative label" behavior for person mentions)."""
    cluster_ids = cluster_by_similarity(embeddings, threshold)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for i, cluster_id in enumerate(cluster_ids):
        clusters[cluster_id].append(i)

    counts = Counter(texts)
    canonical_for: Dict[str, str] = {}
    for members in clusters.values():
        canonical_idx = min(members, key=lambda idx: (-counts[texts[idx]], idx))
        canonical = texts[canonical_idx]
        for idx in members:
            canonical_for[texts[idx]] = canonical
    return canonical_for


def merge_similar_entities(
    triples: List[Dict],
    entity_types: Tuple[str, ...] = SEMANTIC_MERGE_TYPES,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    model: str = "text-embedding-3-small",
    cache_dir: str = "cache_kg_embeddings",
) -> List[Dict]:
    """Embedding-based semantic merge for free-text entity types that have no
    deterministic anchor the way `person` mentions get from
    `resolve_person_alias()`. For each type in `entity_types`, separately:
    embeds every unique mention text of that type (same model + disk-cache
    pattern as `graph_search.semantic_search()`), clusters them by cosine
    similarity >= `threshold` (`cluster_mentions()`), and rewrites every
    mention in a cluster onto that cluster's canonical text.

    Kept as a distinct pass over the triples that runs before `build_graph()`
    rather than folded into `node_key()`'s string normalization, so the
    string-based and embedding-based merging steps stay separable and can be
    unit-tested independently (`cluster_mentions()` needs no API key at all).
    """
    merged = [dict(t) for t in triples]

    for entity_type in entity_types:
        texts = []
        seen = set()
        for t in merged:
            for role in ("subject", "object"):
                if t[f"{role}_type"] == entity_type:
                    text = t[f"{role}_text"]
                    if text not in seen:
                        seen.add(text)
                        texts.append(text)
        if len(texts) < 2:
            continue

        embeddings = embed_texts(texts, model=model, cache_dir=cache_dir)
        canonical_for = cluster_mentions(texts, embeddings, threshold)

        for t in merged:
            for role in ("subject", "object"):
                if t[f"{role}_type"] == entity_type:
                    t[f"{role}_text"] = canonical_for[t[f"{role}_text"]]

    return merged


ENTITY_COLORS = {
    "person": "#4c6ef5",
    "decision": "#2f9e44",
    "issue": "#e03131",
    "task": "#f08c00",
    "topic": "#868e96",
}


def load_triples(input_patterns: List[str]) -> List[Dict]:
    paths = []
    for pattern in input_patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])

    triples = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        default_meeting_id = os.path.splitext(os.path.basename(path))[0]
        for t in data:
            triples.append({
                "subject_text": t["subject_text"],
                "subject_type": t["subject_type"],
                "relation": t["relation"],
                "object_text": t["object_text"],
                "object_type": t["object_type"],
                "timestamp": t.get("timestamp", ""),
                "quote": t.get("quote", ""),
                "meeting_id": t.get("meeting_id", default_meeting_id),
            })
    return triples


def build_graph(triples: List[Dict]) -> nx.MultiDiGraph:
    """Builds a MultiDiGraph (not DiGraph): unlike Phase 2's causal DAG where a
    cause/effect pair implies a single relationship, two entities here can be
    connected by more than one relation type (e.g. a person both `discusses`
    and is `assigned_to` the same task), so each relation gets its own edge."""
    graph = nx.MultiDiGraph()
    for t in triples:
        s_key = node_key(t["subject_type"], t["subject_text"])
        o_key = node_key(t["object_type"], t["object_text"])
        if s_key == o_key:
            continue

        for key, entity_type, label in ((s_key, t["subject_type"], t["subject_text"]),
                                          (o_key, t["object_type"], t["object_text"])):
            if key not in graph:
                graph.add_node(key, label=label, entity_type=entity_type, mentions=[])
            graph.nodes[key]["mentions"].append({
                "text": label, "meeting_id": t["meeting_id"], "timestamp": t["timestamp"],
            })

        edge_data = graph.get_edge_data(s_key, o_key)
        existing_key = None
        if edge_data:
            for ekey, edata in edge_data.items():
                if edata.get("relation") == t["relation"]:
                    existing_key = ekey
                    break
        if existing_key is not None:
            graph[s_key][o_key][existing_key]["weight"] += 1
            graph[s_key][o_key][existing_key]["evidence"].append(t)
        else:
            graph.add_edge(s_key, o_key, relation=t["relation"], weight=1, evidence=[t])
    return graph


def graph_coherence_metrics(graph: nx.MultiDiGraph) -> Dict:
    """Unsupervised structural metrics (no gold labels needed): density,
    modularity/community structure, average inter-community conductance, and
    connected-component count -- the "graph coherence and clustering quality"
    metric called for in the Phase 4 spec."""
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        undirected.add_edge(u, v)

    metrics = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "num_connected_components": nx.number_connected_components(undirected),
    }

    if undirected.number_of_edges() == 0 or undirected.number_of_nodes() < 2:
        metrics.update({"num_communities": 0, "modularity": None, "avg_conductance": None})
        return metrics

    communities = list(greedy_modularity_communities(undirected))
    metrics["num_communities"] = len(communities)
    metrics["modularity"] = nx.algorithms.community.modularity(undirected, communities)

    conductances = []
    for community in communities:
        if 0 < len(community) < undirected.number_of_nodes():
            conductances.append(conductance(undirected, community))
    metrics["avg_conductance"] = sum(conductances) / len(conductances) if conductances else None

    return metrics


def build_entity_type_counts(graph: nx.MultiDiGraph) -> Dict[str, int]:
    counts = defaultdict(int)
    for _, data in graph.nodes(data=True):
        counts[data.get("entity_type", "unknown")] += 1
    return dict(counts)


def save_graph_json(graph: nx.MultiDiGraph, path: str):
    data = {
        "nodes": [
            {"key": n, "label": d["label"], "entity_type": d["entity_type"], "mentions": d["mentions"]}
            for n, d in graph.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, "relation": d["relation"], "weight": d["weight"], "evidence": d["evidence"]}
            for u, v, d in graph.edges(data=True)
        ],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved graph JSON to {path}")


def plot_graph(graph: nx.MultiDiGraph, path: str):
    plt.figure(figsize=(11, 8))
    simple = nx.DiGraph()
    for n, d in graph.nodes(data=True):
        simple.add_node(n, **d)
    for u, v in graph.edges():
        simple.add_edge(u, v)

    pos = nx.spring_layout(simple, seed=42, k=0.6)
    node_colors = [ENTITY_COLORS.get(simple.nodes[n]["entity_type"], "#adb5bd") for n in simple.nodes()]
    labels = {n: simple.nodes[n]["label"] for n in simple.nodes()}

    nx.draw_networkx_nodes(simple, pos, node_size=1400, node_color=node_colors)
    nx.draw_networkx_labels(simple, pos, labels=labels, font_size=7)
    nx.draw_networkx_edges(simple, pos, node_size=1400, arrowstyle="-|>", arrowsize=14, edge_color="#555555")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=t)
               for t, c in ENTITY_COLORS.items()]
    plt.legend(handles=handles, loc="upper left", fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved knowledge graph visualization to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="One or more triples.json files or glob patterns")
    parser.add_argument("--graph-json", default="output/kg_graph.json")
    parser.add_argument("--plot", default="output/kg_graph.png")
    parser.add_argument("--metrics-output", default=None, help="Optional path to save graph coherence metrics JSON")
    parser.add_argument("--no-semantic-merge", action="store_true",
                         help="Skip the embedding-based merge of issue/decision/task mentions (no OpenAI calls)")
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                         help="Cosine similarity threshold for merge_similar_entities()")
    parser.add_argument("--embedding-cache-dir", default="cache_kg_embeddings")
    args = parser.parse_args()

    triples = load_triples(args.input)
    if not triples:
        raise SystemExit("No triples found in the given --input path(s)")

    if not args.no_semantic_merge:
        triples = merge_similar_entities(
            triples, threshold=args.similarity_threshold, cache_dir=args.embedding_cache_dir,
        )

    graph = build_graph(triples)
    save_graph_json(graph, args.graph_json)
    plot_graph(graph, args.plot)

    metrics = graph_coherence_metrics(graph)
    print(f"\n{metrics['num_nodes']} nodes, {metrics['num_edges']} edges")
    print(f"Entity type counts: {build_entity_type_counts(graph)}")
    print(f"Coherence metrics: {metrics}")

    if args.metrics_output:
        os.makedirs(os.path.dirname(args.metrics_output) or ".", exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Saved coherence metrics to {args.metrics_output}")
