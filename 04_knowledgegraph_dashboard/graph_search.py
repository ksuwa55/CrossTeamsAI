import hashlib
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import networkx as nx
from dotenv import load_dotenv


# ----------------------------
# Keyword search (no API calls)
# ----------------------------
def _node_corpus(graph: nx.MultiDiGraph, node: str, max_quotes: int = 5) -> str:
    data = graph.nodes[node]
    quotes = []
    for _, _, edata in list(graph.out_edges(node, data=True)) + list(graph.in_edges(node, data=True)):
        for ev in edata.get("evidence", []):
            if ev.get("quote"):
                quotes.append(ev["quote"])
    return " ".join([data.get("label", "")] + quotes[:max_quotes])


def keyword_search(graph: nx.MultiDiGraph, query: str, top_k: int = 10) -> List[Tuple[str, int]]:
    """Case-insensitive substring match over node labels + connected evidence
    quotes, scored by how many query tokens appear. Zero API calls, always
    available as a fallback to semantic search."""
    tokens = [t for t in query.lower().split() if t]
    if not tokens:
        return []

    scored = []
    for node in graph.nodes():
        corpus = _node_corpus(graph, node).lower()
        score = sum(corpus.count(t) for t in tokens)
        if score > 0:
            scored.append((node, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ----------------------------
# Semantic search (OpenAI embeddings, disk-cached)
# ----------------------------
def _embedding_cache_path(cache_dir: str, model: str, text: str) -> str:
    key = hashlib.md5((model + text).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.json")


def embed_texts(texts: List[str], model: str = "text-embedding-3-small", cache_dir: str = "cache_kg_embeddings") -> List[List[float]]:
    """Embeds a list of texts with disk caching (same pattern as
    MeetingSummarizer's response cache), so repeated node text and repeated
    queries don't re-spend API calls."""
    load_dotenv()
    os.makedirs(cache_dir, exist_ok=True)

    results: List[Optional[List[float]]] = [None] * len(texts)
    to_fetch_idx = []
    to_fetch_text = []
    for i, text in enumerate(texts):
        cache_path = _embedding_cache_path(cache_dir, model, text)
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                results[i] = json.load(f)["embedding"]
        else:
            to_fetch_idx.append(i)
            to_fetch_text.append(text)

    if to_fetch_text:
        from openai import OpenAI
        client = OpenAI()
        response = client.embeddings.create(model=model, input=to_fetch_text)
        for idx, text, item in zip(to_fetch_idx, to_fetch_text, response.data):
            embedding = item.embedding
            results[idx] = embedding
            with open(_embedding_cache_path(cache_dir, model, text), "w", encoding="utf-8") as f:
                json.dump({"text": text, "embedding": embedding}, f)

    return results  # type: ignore[return-value]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_search(
    graph: nx.MultiDiGraph,
    query: str,
    top_k: int = 10,
    model: str = "text-embedding-3-small",
    cache_dir: str = "cache_kg_embeddings",
) -> List[Tuple[str, float]]:
    """Embedding-based ranking over node label + evidence text. Requires a
    live OpenAI API key/credits (unless every node + the query are already
    cached in `cache_dir`) — callers that need a zero-API-cost path should use
    keyword_search() instead."""
    nodes = list(graph.nodes())
    corpora = [_node_corpus(graph, n) for n in nodes]
    query_embedding, *node_embeddings = embed_texts([query] + corpora, model=model, cache_dir=cache_dir)

    scored = [
        (node, cosine_similarity(query_embedding, emb))
        for node, emb in zip(nodes, node_embeddings)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ----------------------------
# Path finding + local exploration
# ----------------------------
def find_path(graph: nx.MultiDiGraph, source: str, target: str) -> Optional[Dict]:
    if source not in graph or target not in graph:
        return None
    simple = nx.DiGraph()
    simple.add_edges_from(graph.edges())
    if not nx.has_path(simple, source, target):
        return None
    path = nx.shortest_path(simple, source, target)
    return {"path": path, "evidence_md": format_path_evidence(graph, path)}


def format_path_evidence(graph: nx.MultiDiGraph, path: List[str], max_examples_per_edge: int = 2) -> str:
    sections = []
    for u, v in zip(path[:-1], path[1:]):
        u_label = graph.nodes[u]["label"]
        v_label = graph.nodes[v]["label"]
        for _, edata in graph.get_edge_data(u, v).items():
            examples = sorted(edata["evidence"], key=lambda e: e.get("timestamp", ""))[:max_examples_per_edge]
            quotes = "\n".join(
                f'- _{ex.get("meeting_id", "unknown meeting")} ({ex.get("timestamp", "n/a")})_: "{ex.get("quote", "")}"'
                for ex in examples
            )
            sections.append(f"**{u_label} --[{edata['relation']}]--> {v_label}** (seen {edata['weight']}x)\n{quotes}")
    return "\n\n".join(sections)


def ego_subgraph(graph: nx.MultiDiGraph, center: str, radius: int = 1) -> nx.MultiDiGraph:
    """Nodes reachable from `center` within `radius` hops, following edges in
    either direction (so both what `center` caused and what caused `center`
    show up), for a dashboard "explore around this node" view."""
    if center not in graph:
        return graph.__class__()
    nodes = {center}
    frontier = {center}
    for _ in range(radius):
        next_frontier = set()
        for n in frontier:
            next_frontier |= set(graph.successors(n)) | set(graph.predecessors(n))
        next_frontier -= nodes
        nodes |= next_frontier
        frontier = next_frontier
    return graph.subgraph(nodes).copy()
