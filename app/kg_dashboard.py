import glob
import os
import sys
from collections import defaultdict

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
KG_PKG = os.path.join(REPO_ROOT, "04_knowledgegraph_dashboard")
CAUSAL_PKG = os.path.join(REPO_ROOT, "02_causal_modeling")
sys.path.insert(0, KG_PKG)
sys.path.insert(0, CAUSAL_PKG)

from build_knowledge_graph import (  # noqa: E402
    load_triples, build_graph, graph_coherence_metrics, merge_similar_entities,
    ENTITY_COLORS, DEFAULT_SIMILARITY_THRESHOLD,
)
from graph_search import keyword_search, semantic_search, ego_subgraph, format_path_evidence  # noqa: E402
from extract_entities_relations import extract_topic_timeline, extract_speakers  # noqa: E402
from extract_variables import enrich_transcript  # noqa: E402

DEFAULT_TRIPLES_GLOB = os.path.join(REPO_ROOT, "data", "sample_kg_triples", "*.json")
DEFAULT_TRANSCRIPTS_DIR = os.path.join(REPO_ROOT, "data", "synthetic_transcripts")


def _load_model(triples_glob: str, transcripts_dir: str, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
    triples = load_triples([triples_glob])
    try:
        triples = merge_similar_entities(triples, threshold=similarity_threshold)
    except Exception as exc:
        print(f"[kg_dashboard] merge_similar_entities() failed ({exc}); building graph from unmerged triples.")
    graph = build_graph(triples)

    transcript_paths = sorted(
        p for p in glob.glob(os.path.join(transcripts_dir, "*.json"))
        if not p.endswith(".labels.json") and not p.endswith(".kg_labels.json")
    )
    topic_timelines = {}
    person_graph = nx.Graph()
    for path in transcript_paths:
        meeting_id = os.path.splitext(os.path.basename(path))[0]
        enriched = enrich_transcript(path)
        topic_timelines[meeting_id] = extract_topic_timeline(enriched, meeting_id=meeting_id)

        speakers = sorted(set(extract_speakers(enriched)))
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                a, b = speakers[i], speakers[j]
                if person_graph.has_edge(a, b):
                    person_graph[a][b]["weight"] += 1
                else:
                    person_graph.add_edge(a, b, weight=1)

    return graph, topic_timelines, person_graph


_graph, _topic_timelines, _person_graph = _load_model(DEFAULT_TRIPLES_GLOB, DEFAULT_TRANSCRIPTS_DIR)
_meeting_ids = sorted(_topic_timelines.keys())


def _plot_subgraph(subgraph: nx.MultiDiGraph, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    if subgraph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        ax.axis("off")
        return fig

    simple = nx.DiGraph()
    for n, d in subgraph.nodes(data=True):
        simple.add_node(n, **d)
    for u, v in subgraph.edges():
        simple.add_edge(u, v)

    pos = nx.spring_layout(simple, seed=42, k=0.8)
    colors = [ENTITY_COLORS.get(simple.nodes[n]["entity_type"], "#adb5bd") for n in simple.nodes()]
    labels = {n: simple.nodes[n]["label"] for n in simple.nodes()}
    nx.draw_networkx_nodes(simple, pos, node_size=1200, node_color=colors, ax=ax)
    nx.draw_networkx_labels(simple, pos, labels=labels, font_size=7, ax=ax)
    nx.draw_networkx_edges(simple, pos, node_size=1200, arrowstyle="-|>", arrowsize=12, edge_color="#555555", ax=ax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ----------------------------
# Explore tab: graph-based search
# ----------------------------
def run_search(query: str, mode: str):
    if not query.strip():
        return "Enter a search query.", None

    if mode == "Semantic (OpenAI embeddings)":
        try:
            hits = semantic_search(_graph, query, top_k=8)
        except Exception as exc:
            return (
                f"Semantic search failed ({exc}). This requires a live OpenAI API key/credits — "
                "try 'Keyword' search instead, which works offline.",
                None,
            )
    else:
        hits = keyword_search(_graph, query, top_k=8)

    if not hits:
        return "No matches found.", None

    lines = ["**Matching entities:**\n"]
    for node, score in hits:
        data = _graph.nodes[node]
        lines.append(f"- `{data['entity_type']}` **{data['label']}** (score {score:.3f})")

    top_node = hits[0][0]
    sub = ego_subgraph(_graph, top_node, radius=1)
    fig = _plot_subgraph(sub, f"Neighborhood of: {_graph.nodes[top_node]['label']}")
    return "\n".join(lines), fig


# ----------------------------
# Network tab: full graph + coherence metrics + discussion network
# ----------------------------
def render_network_overview():
    fig = _plot_subgraph(_graph, "Full knowledge graph")
    metrics = graph_coherence_metrics(_graph)
    summary = (
        f"Nodes: {metrics['num_nodes']}  |  Edges: {metrics['num_edges']}\n"
        f"Density: {metrics['density']:.4f}\n"
        f"Connected components: {metrics['num_connected_components']}\n"
        f"Communities (greedy modularity): {metrics['num_communities']}\n"
        f"Modularity: {metrics['modularity']}\n"
        f"Avg. inter-community conductance: {metrics['avg_conductance']}"
    )
    return fig, summary


def render_discussion_network():
    fig, ax = plt.subplots(figsize=(6, 5))
    if _person_graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No speaker data", ha="center", va="center")
        ax.axis("off")
        return fig
    pos = nx.spring_layout(_person_graph, seed=7, k=0.9)
    weights = [_person_graph[u][v]["weight"] for u, v in _person_graph.edges()]
    nx.draw_networkx_nodes(_person_graph, pos, node_size=900, node_color=ENTITY_COLORS["person"], ax=ax)
    nx.draw_networkx_labels(_person_graph, pos, font_size=7, ax=ax)
    nx.draw_networkx_edges(_person_graph, pos, width=[1 + w * 0.5 for w in weights], edge_color="#888888", ax=ax)
    ax.set_title("Discussion network (co-occurrence in the same meeting)", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ----------------------------
# Topic drift tab
# ----------------------------
def render_topic_drift(meeting_choice: str):
    fig, ax = plt.subplots(figsize=(7, 4))

    if meeting_choice == "All meetings (by meeting)":
        counts_by_meeting = {}
        all_topics = set()
        for mid in _meeting_ids:
            counts = defaultdict(int)
            for entry in _topic_timelines[mid]:
                counts[entry["topic"]] += 1
                all_topics.add(entry["topic"])
            counts_by_meeting[mid] = counts
        all_topics = sorted(all_topics)

        bottoms = [0] * len(_meeting_ids)
        for topic in all_topics:
            values = [counts_by_meeting[mid].get(topic, 0) for mid in _meeting_ids]
            ax.bar(_meeting_ids, values, bottom=bottoms, label=topic)
            bottoms = [b + v for b, v in zip(bottoms, values)]
        ax.set_ylabel("Utterance count")
        ax.set_title("Topic mix per meeting (drift across the project timeline)")
        ax.tick_params(axis="x", rotation=75, labelsize=6)
        ax.legend(fontsize=6, loc="upper right")
    else:
        timeline = _topic_timelines.get(meeting_choice, [])
        topics = sorted(set(e["topic"] for e in timeline))
        y = [topics.index(e["topic"]) for e in timeline]
        x = list(range(len(timeline)))
        ax.step(x, y, where="mid", marker="o", markersize=4)
        ax.set_yticks(range(len(topics)))
        ax.set_yticklabels(topics, fontsize=8)
        ax.set_xlabel("Utterance index")
        ax.set_title(f"Topic drift within {meeting_choice}")

    fig.tight_layout()
    return fig


with gr.Blocks(title="Knowledge Graph Explorer") as demo:
    gr.Markdown("## Team Knowledge Graph — Search, Explore, and Analyze")
    gr.Markdown(
        "Built by `04_knowledgegraph_dashboard/build_knowledge_graph.py` from entity/relation "
        "triples extracted with `extract_entities_relations.py` (reuses Phase 2's transcript "
        f"prefilter), with `merge_similar_entities()` (cosine similarity threshold={DEFAULT_SIMILARITY_THRESHOLD}) "
        "merging near-duplicate `issue`/`decision`/`task` mentions before the graph is built. "
        "Seed data: `data/sample_kg_triples/*.json`."
    )

    with gr.Tab("Explore"):
        with gr.Row():
            query = gr.Textbox(label="Search query", placeholder="e.g. QA backlog, sign-off, Aiko")
            mode = gr.Radio(["Keyword", "Semantic (OpenAI embeddings)"], value="Keyword", label="Search mode")
        search_btn = gr.Button("Search", variant="primary")
        search_results = gr.Markdown(label="Results")
        search_plot = gr.Plot(label="Neighborhood of top match")
        search_btn.click(run_search, inputs=[query, mode], outputs=[search_results, search_plot])

    with gr.Tab("Network"):
        gr.Markdown("Full graph colored by entity type, graph-coherence metrics, and the discussion network.")
        refresh_btn = gr.Button("Render")
        with gr.Row():
            network_plot = gr.Plot(label="Full knowledge graph")
            discussion_plot = gr.Plot(label="Discussion network")
        network_summary = gr.Textbox(label="Coherence metrics", lines=6)
        refresh_btn.click(render_network_overview, outputs=[network_plot, network_summary])
        refresh_btn.click(render_discussion_network, outputs=[discussion_plot])

    with gr.Tab("Topic Drift"):
        meeting_dropdown = gr.Dropdown(
            ["All meetings (by meeting)"] + _meeting_ids,
            value="All meetings (by meeting)",
            label="Meeting",
        )
        topic_plot = gr.Plot(label="Topic drift")
        meeting_dropdown.change(render_topic_drift, inputs=[meeting_dropdown], outputs=[topic_plot])
        demo.load(render_topic_drift, inputs=[meeting_dropdown], outputs=[topic_plot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)
