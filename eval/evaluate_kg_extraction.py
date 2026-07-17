"""
Evaluate the entity/relation extraction pipeline
(04_knowledgegraph_dashboard/extract_entities_relations.py) against hand-labeled
ground-truth (subject, relation, object) triples for the synthetic meeting
transcripts (the same 10 meetings Phase 2's evaluate_causal_extraction.py uses,
labeled with a new schema: meeting_XX_*.kg_labels.json).

Two metrics, both called for in the Phase 4 spec:
  1. Entity-linking precision/recall: does extract_entities_relations.py resolve
     entity mentions onto the same normalized node key gold labels do?
  2. Relation/fact extraction P/R/F1: does it recover the right (subject,
     relation, object) triples? Matching reuses the LLM-judge + one-to-one
     maximum-bipartite-matching machinery from evaluate_causal_extraction.py
     (parse_judge_response, max_bipartite_matching are schema-agnostic and
     imported unmodified; only the judge prompt is triple-specific).

Also reports unsupervised graph-coherence metrics (density/modularity/
conductance, via build_knowledge_graph.graph_coherence_metrics) on the graph
built from all predicted triples -- no gold labels needed for that part.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "04_knowledgegraph_dashboard"))
sys.path.insert(0, os.path.join(REPO_ROOT, "01_summarization"))
sys.path.insert(0, os.path.join(REPO_ROOT, "02_causal_modeling"))

from extract_entities_relations import enrich_transcript, extract_kg_triples, node_key, save_output  # noqa: E402
from build_knowledge_graph import (  # noqa: E402
    build_graph, graph_coherence_metrics, merge_similar_entities, cluster_by_similarity, SEMANTIC_MERGE_TYPES,
)
from graph_search import embed_texts  # noqa: E402
from summarizer import MeetingSummarizer  # noqa: E402
from evaluate_causal_extraction import parse_judge_response, max_bipartite_matching, format_pct  # noqa: E402


# ─── Discovery / I/O ───────────────────────────────────────────────

def discover_meetings(transcripts_dir: str) -> List[Tuple[str, str, str]]:
    meetings = []
    for transcript_path in sorted(glob.glob(os.path.join(transcripts_dir, "*.json"))):
        if transcript_path.endswith(".labels.json") or transcript_path.endswith(".kg_labels.json"):
            continue
        meeting_id = os.path.splitext(os.path.basename(transcript_path))[0]
        labels_path = os.path.join(transcripts_dir, f"{meeting_id}.kg_labels.json")
        if not os.path.exists(labels_path):
            print(f"[skip] no kg_labels file for {meeting_id}")
            continue
        meetings.append((meeting_id, transcript_path, labels_path))
    return meetings


def load_labels(labels_path: str) -> List[Dict]:
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Entity-linking metrics ────────────────────────────────────────

def _entity_texts_by_type(triples: List[Dict]) -> Dict[str, Dict[str, str]]:
    """entity_type -> {node_key: one representative raw mention text}."""
    by_type: Dict[str, Dict[str, str]] = defaultdict(dict)
    for t in triples:
        for role in ("subject", "object"):
            entity_type = t[f"{role}_type"]
            text = t[f"{role}_text"]
            by_type[entity_type].setdefault(node_key(entity_type, text), text)
    return by_type


def _semantic_match_counts(
    pred_key_to_text: Dict[str, str],
    gold_key_to_text: Dict[str, str],
    threshold: float,
    model: str,
    cache_dir: str,
) -> Tuple[int, int]:
    """Embeds predicted + gold representative mention texts *together* and
    clusters them with build_knowledge_graph's shared clustering primitive
    (`cluster_by_similarity()` -- the exact same single-linkage clustering
    `merge_similar_entities()` uses to merge predicted mentions with each
    other, applied here across the predicted/gold boundary too, so a
    predicted mention can match a differently-worded gold mention of the
    same real-world issue/decision/task instead of requiring exact
    `node_key()` string equality).

    Returns (matched_predicted, matched_gold): a predicted entity counts
    toward precision if its cluster contains >=1 gold entity; a gold entity
    counts toward recall if its cluster contains >=1 predicted entity. (These
    can differ from each other -- one gold entity's cluster can contain
    several matching predicted variants, for instance.)"""
    pred_keys = list(pred_key_to_text.keys())
    gold_keys = list(gold_key_to_text.keys())
    if not pred_keys or not gold_keys:
        return 0, 0

    texts = [pred_key_to_text[k] for k in pred_keys] + [gold_key_to_text[k] for k in gold_keys]
    embeddings = embed_texts(texts, model=model, cache_dir=cache_dir)
    cluster_ids = cluster_by_similarity(embeddings, threshold)

    pred_cluster_ids = cluster_ids[:len(pred_keys)]
    gold_cluster_ids = cluster_ids[len(pred_keys):]
    mixed_clusters = set(pred_cluster_ids) & set(gold_cluster_ids)

    matched_predicted = sum(1 for cid in pred_cluster_ids if cid in mixed_clusters)
    matched_gold = sum(1 for cid in gold_cluster_ids if cid in mixed_clusters)
    return matched_predicted, matched_gold


def entity_linking_metrics_by_type(
    pred_triples: List[Dict],
    gold_triples: List[Dict],
    semantic_types: Tuple[str, ...] = (),
    threshold: float = 0.85,
    model: str = "text-embedding-3-small",
    cache_dir: str = "cache_kg_embeddings",
) -> Dict[str, Dict]:
    """Entity-linking precision/recall broken out per entity_type. For types
    in `semantic_types`, a predicted mention matches a gold mention if an
    embedding clusters them together (`_semantic_match_counts()`), so
    paraphrased mentions of the same real-world issue/decision/task can
    match gold without exact string equality. Other types (`person`, `topic`
    by default) keep exact `node_key()` matching -- `person` already has a
    deterministic anchor via `resolve_person_alias()`, and fuzzy-matching
    people risks conflating two different people who are merely discussed
    similarly."""
    pred_by_type = _entity_texts_by_type(pred_triples)
    gold_by_type = _entity_texts_by_type(gold_triples)
    out = {}
    for entity_type in sorted(set(pred_by_type) | set(gold_by_type)):
        pred_key_to_text = pred_by_type.get(entity_type, {})
        gold_key_to_text = gold_by_type.get(entity_type, {})
        num_predicted = len(pred_key_to_text)
        num_gold = len(gold_key_to_text)

        if entity_type in semantic_types:
            matched_predicted, matched_gold = _semantic_match_counts(
                pred_key_to_text, gold_key_to_text, threshold, model, cache_dir,
            )
        else:
            tp = len(set(pred_key_to_text) & set(gold_key_to_text))
            matched_predicted = matched_gold = tp

        out[entity_type] = {
            "num_predicted_entities": num_predicted,
            "num_gold_entities": num_gold,
            "matched_for_precision": matched_predicted,
            "matched_for_recall": matched_gold,
            "precision": matched_predicted / num_predicted if num_predicted else 0.0,
            "recall": matched_gold / num_gold if num_gold else 0.0,
        }
    return out


def _pool_entity_linking(by_type: Dict[str, Dict]) -> Dict:
    num_predicted = sum(m["num_predicted_entities"] for m in by_type.values())
    num_gold = sum(m["num_gold_entities"] for m in by_type.values())
    matched_predicted = sum(m["matched_for_precision"] for m in by_type.values())
    matched_gold = sum(m["matched_for_recall"] for m in by_type.values())
    return {
        "num_predicted_entities": num_predicted,
        "num_gold_entities": num_gold,
        "matched_for_precision": matched_predicted,
        "matched_for_recall": matched_gold,
        "precision": matched_predicted / num_predicted if num_predicted else 0.0,
        "recall": matched_gold / num_gold if num_gold else 0.0,
    }


def entity_linking_metrics(
    pred_triples: List[Dict],
    gold_triples: List[Dict],
    semantic_types: Tuple[str, ...] = (),
    threshold: float = 0.85,
    model: str = "text-embedding-3-small",
    cache_dir: str = "cache_kg_embeddings",
) -> Dict:
    """Pooled (all entity types combined) version of entity_linking_metrics_by_type()."""
    return _pool_entity_linking(
        entity_linking_metrics_by_type(pred_triples, gold_triples, semantic_types, threshold, model, cache_dir)
    )


# ─── LLM-judge triple matching (reuses causal eval's generic matching code) ─

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating whether a predicted knowledge-graph triple (subject, relation, "
    "object) from a meeting transcript matches any triples in a list of ground-truth "
    "triples. Two triples match only if they describe substantially the same fact: the "
    "subject must be the same underlying entity (not just a related topic), the object "
    "must be the same underlying entity, and the relation must express the same kind of "
    "connection, in the same direction (do not count it as a match if subject and object "
    "are reversed). Wording, phrasing, and level of detail may differ, and the relation "
    "label doesn't need to be an exact string match as long as it captures the same kind "
    "of connection. For each ground-truth triple, briefly reason about whether it matches "
    "(one short sentence each), then on the LAST line output ONLY a JSON array of "
    "booleans, in order, no markdown fences."
)


def build_judge_prompt(pred: Dict, gold_triples: List[Dict]) -> str:
    def fmt(t):
        return f"subject: \"{t['subject_text']}\" ({t['subject_type']}) -- {t['relation']} --> object: \"{t['object_text']}\" ({t['object_type']})"

    gold_list = "\n".join(f"{i + 1}. {fmt(g)}" for i, g in enumerate(gold_triples))
    return (
        f"Predicted triple:\n{fmt(pred)}\n\n"
        f"Ground-truth triples:\n{gold_list}\n\n"
        f"For each of the {len(gold_triples)} ground-truth triples above, does the predicted "
        f"triple capture the same fact? Reason briefly per triple, then end with a JSON array "
        f"of exactly {len(gold_triples)} booleans (true/false), in order, on its own line."
    )


def judge_compatibility_matrix(
    summarizer: MeetingSummarizer,
    pred_triples: List[Dict],
    gold_triples: List[Dict],
    judge_model: str,
) -> List[List[bool]]:
    matrix = []
    for pred in pred_triples:
        prompt = build_judge_prompt(pred, gold_triples)
        raw = summarizer.run_summarizer(
            prompt,
            model=judge_model,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=500,
        )
        matrix.append(parse_judge_response(raw, len(gold_triples)))
    return matrix


# ─── Per-meeting evaluation ────────────────────────────────────────

def evaluate_meeting(
    meeting_id: str,
    pred_triples: List[Dict],
    gold_triples: List[Dict],
    summarizer: MeetingSummarizer,
    judge_model: str,
    entity_linking_pred_triples: List[Dict] = None,
    semantic_entity_types: Tuple[str, ...] = (),
    similarity_threshold: float = 0.85,
    embedding_model: str = "text-embedding-3-small",
    embedding_cache_dir: str = "cache_kg_embeddings",
) -> Dict:
    """`entity_linking_pred_triples` lets entity-linking be scored against a
    semantically-merged view of the predictions (merge_similar_entities(),
    pooled across meetings before slicing back per-meeting) while relation
    P/R/F1 below is still judged against the raw, unmerged `pred_triples` --
    the merge only changes which node a mention resolves to, not whether the
    underlying fact was extracted. `semantic_entity_types` additionally makes
    entity-linking itself (both here and in the by-type breakdown) match
    those types' mentions against gold via embedding clustering rather than
    exact `node_key()` equality -- kept meeting-scoped (never pooled across
    meetings) since gold labels are only valid within their own meeting."""
    if entity_linking_pred_triples is None:
        entity_linking_pred_triples = pred_triples
    entity_metrics_by_type = entity_linking_metrics_by_type(
        entity_linking_pred_triples, gold_triples, semantic_entity_types,
        similarity_threshold, embedding_model, embedding_cache_dir,
    )
    entity_metrics = _pool_entity_linking(entity_metrics_by_type)

    if pred_triples and gold_triples:
        matrix = judge_compatibility_matrix(summarizer, pred_triples, gold_triples, judge_model)
        matches = max_bipartite_matching(matrix)
    else:
        matches = []

    matched_pred_idxs = {i for i, _ in matches}
    matched_gold_idxs = {j for _, j in matches}

    true_positives = [{"predicted": pred_triples[i], "ground_truth": gold_triples[j]} for i, j in matches]
    false_positives = [pred_triples[i] for i in range(len(pred_triples)) if i not in matched_pred_idxs]
    false_negatives = [gold_triples[j] for j in range(len(gold_triples)) if j not in matched_gold_idxs]

    tp, fp, fn = len(true_positives), len(false_positives), len(false_negatives)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "meeting_id": meeting_id,
        "entity_linking": entity_metrics,
        "entity_linking_by_type": entity_metrics_by_type,
        "num_predicted": len(pred_triples),
        "num_gold": len(gold_triples),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


# ─── Report generation ─────────────────────────────────────────────

def build_markdown_report(per_meeting: List[Dict], aggregate: Dict, agg_entity: Dict, agg_entity_by_type: Dict,
                           coherence: Dict, judge_model: str, extract_model: str,
                           similarity_threshold: float, semantic_merge_enabled: bool) -> str:
    lines = []
    lines.append("# Knowledge Graph Extraction Evaluation\n")
    lines.append(
        f"Extraction model: `{extract_model}` (unchanged pipeline). "
        f"Triple matching: LLM judge (`{judge_model}`), one-to-one maximum bipartite matching per meeting. "
        + (
            f"`merge_similar_entities()` (cosine similarity threshold={similarity_threshold}) merges "
            f"`issue`/`decision`/`task` mentions pooled across meetings before graph coherence is computed; "
            f"entity-linking below *also* matches those same types against gold labels via embedding "
            f"clustering at the same threshold (`_semantic_match_counts()`), instead of requiring exact "
            f"`node_key()` string equality. Relation/fact P/R/F1 is still judged against the raw, unmerged "
            f"predictions.\n"
            if semantic_merge_enabled else
            "Semantic merge/matching was disabled for this run (`--no-semantic-merge`); entity-linking and "
            "graph coherence use raw `node_key()` string matching only.\n"
        )
    )

    lines.append("## Entity-linking (Top-N precision/recall)\n")
    lines.append("| Meeting | Pred Entities | Gold Entities | Matched (P) | Matched (R) | Precision | Recall |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in per_meeting:
        e = m["entity_linking"]
        lines.append(
            f"| {m['meeting_id']} | {e['num_predicted_entities']} | {e['num_gold_entities']} | "
            f"{e['matched_for_precision']} | {e['matched_for_recall']} | {format_pct(e['precision'])} | {format_pct(e['recall'])} |"
        )
    lines.append(
        f"| **Aggregate (pooled)** | {agg_entity['num_predicted_entities']} | {agg_entity['num_gold_entities']} | "
        f"{agg_entity['matched_for_precision']} | {agg_entity['matched_for_recall']} | "
        f"{format_pct(agg_entity['precision'])} | {format_pct(agg_entity['recall'])} |\n"
    )
    lines.append(
        "*Matched (P)/(R): a predicted (resp. gold) entity counts as matched if it's exact-`node_key()`-equal "
        "to, or (for semantically-matched types) embedding-clustered with, some gold (resp. predicted) entity "
        "-- these can differ, since one gold mention's cluster can contain several matching predicted variants.*\n"
    )

    lines.append("## Entity-linking by type (pooled, micro-averaged)\n")
    lines.append("| Entity type | Pred Entities | Gold Entities | Matched (P) | Matched (R) | Precision | Recall |")
    lines.append("|---|---|---|---|---|---|---|")
    for entity_type in sorted(agg_entity_by_type):
        e = agg_entity_by_type[entity_type]
        lines.append(
            f"| `{entity_type}` | {e['num_predicted_entities']} | {e['num_gold_entities']} | "
            f"{e['matched_for_precision']} | {e['matched_for_recall']} | {format_pct(e['precision'])} | {format_pct(e['recall'])} |"
        )
    lines.append("")

    lines.append("## Relation / fact extraction (Precision / Recall / F1)\n")
    lines.append("| Meeting | Predicted | Gold | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in per_meeting:
        lines.append(
            f"| {m['meeting_id']} | {m['num_predicted']} | {m['num_gold']} | {m['tp']} | "
            f"{m['fp']} | {m['fn']} | {format_pct(m['precision'])} | {format_pct(m['recall'])} | {format_pct(m['f1'])} |"
        )
    lines.append(
        f"| **Aggregate (pooled)** | {aggregate['num_predicted']} | {aggregate['num_gold']} | "
        f"{aggregate['tp']} | {aggregate['fp']} | {aggregate['fn']} | "
        f"{format_pct(aggregate['precision'])} | {format_pct(aggregate['recall'])} | {format_pct(aggregate['f1'])} |\n"
    )
    lines.append(
        "*Aggregate is pooled (micro-averaged): TP/FP/FN (or matched/pred/gold for entity "
        "linking) are summed across all meetings before computing precision/recall/F1.*\n"
    )

    lines.append("## Graph coherence (unsupervised, no gold labels)\n")
    lines.append(
        f"Built from all predicted triples pooled across meetings: {coherence['num_nodes']} nodes, "
        f"{coherence['num_edges']} edges, density={coherence['density']:.4f}, "
        f"connected components={coherence['num_connected_components']}, "
        f"communities={coherence['num_communities']}, modularity={coherence['modularity']}, "
        f"avg. inter-community conductance={coherence['avg_conductance']}.\n"
    )

    lines.append("## False positives (predicted, no matching ground truth)\n")
    any_fp = False
    for m in per_meeting:
        for fp in m["false_positives"]:
            any_fp = True
            lines.append(
                f"- **{m['meeting_id']}**: {fp['subject_text']} --[{fp['relation']}]--> {fp['object_text']}"
            )
    if not any_fp:
        lines.append("- none")
    lines.append("")

    lines.append("## False negatives (ground truth missed by the model)\n")
    any_fn = False
    for m in per_meeting:
        for fn in m["false_negatives"]:
            any_fn = True
            lines.append(
                f"- **{m['meeting_id']}**: {fn['subject_text']} --[{fn['relation']}]--> {fn['object_text']}"
            )
    if not any_fn:
        lines.append("- none")
    lines.append("")

    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcripts-dir", default=os.path.join(REPO_ROOT, "data", "synthetic_transcripts"))
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "output", "kg_triples"),
                         help="Where predicted triples JSON per meeting is saved")
    parser.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "eval", "results", "kg_extraction"),
                         help="Where the markdown report and raw matched/unmatched JSON are saved")
    parser.add_argument("--extract-model", default="gpt-3.5-turbo")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--window-before", type=int, default=2)
    parser.add_argument("--window-after", type=int, default=1)
    parser.add_argument("--extraction-cache-dir", default=os.path.join(REPO_ROOT, "cache_kg"))
    parser.add_argument("--judge-cache-dir", default=os.path.join(REPO_ROOT, "cache_kg_eval"))
    parser.add_argument("--embedding-cache-dir", default=os.path.join(REPO_ROOT, "cache_kg_embeddings"))
    parser.add_argument("--similarity-threshold", type=float, default=0.85,
                         help="Cosine similarity threshold for merge_similar_entities() and semantic "
                              "entity-linking-vs-gold matching (issue/decision/task only)")
    parser.add_argument("--no-semantic-merge", action="store_true",
                         help="Skip merge_similar_entities() and semantic entity-linking matching; "
                              "entity-linking/graph coherence use raw node_key() string matching only")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    meetings = discover_meetings(args.transcripts_dir)
    print(f"Found {len(meetings)} meetings with transcript + kg_labels.")

    judge_summarizer = MeetingSummarizer(cache_dir=args.judge_cache_dir)

    pred_by_meeting = {}
    gold_by_meeting = {}
    all_pred_triples = []
    for meeting_id, transcript_path, labels_path in meetings:
        print(f"\n[{meeting_id}] enriching + extracting triples...")
        enriched = enrich_transcript(transcript_path)
        pred_triples = extract_kg_triples(
            enriched,
            model=args.extract_model,
            window_before=args.window_before,
            window_after=args.window_after,
            cache_dir=args.extraction_cache_dir,
            meeting_id=meeting_id,
        )
        pred_out_path = os.path.join(args.output_dir, f"{meeting_id}.kg_triples.json")
        save_output(pred_triples, pred_out_path)

        pred_by_meeting[meeting_id] = pred_triples
        gold_by_meeting[meeting_id] = load_labels(labels_path)
        all_pred_triples.extend(pred_triples)

    # Semantic merge runs once, pooled across all meetings' predictions (matching
    # how the real dashboard and the pooled graph-coherence graph below merge
    # triples from every meeting into one graph) -- then results are sliced back
    # per meeting for entity-linking. Relation/fact P/R/F1 below is judged
    # against the raw, unmerged predictions per meeting; the merge only affects
    # which node a mention resolves to.
    if args.no_semantic_merge:
        merged_pred_triples = all_pred_triples
    else:
        print(f"\nRunning merge_similar_entities() (threshold={args.similarity_threshold}) over "
              f"{len(all_pred_triples)} pooled predicted triples...")
        merged_pred_triples = merge_similar_entities(
            all_pred_triples, threshold=args.similarity_threshold, cache_dir=args.embedding_cache_dir,
        )
    merged_by_meeting = defaultdict(list)
    for t in merged_pred_triples:
        merged_by_meeting[t["meeting_id"]].append(t)

    semantic_entity_types = () if args.no_semantic_merge else SEMANTIC_MERGE_TYPES

    per_meeting_results = []
    for meeting_id, _, _ in meetings:
        pred_triples = pred_by_meeting[meeting_id]
        gold_triples = gold_by_meeting[meeting_id]
        print(f"\n[{meeting_id}] judging {len(pred_triples)} predicted vs {len(gold_triples)} gold triples...")
        result = evaluate_meeting(
            meeting_id, pred_triples, gold_triples, judge_summarizer, args.judge_model,
            entity_linking_pred_triples=merged_by_meeting[meeting_id],
            semantic_entity_types=semantic_entity_types,
            similarity_threshold=args.similarity_threshold,
            embedding_cache_dir=args.embedding_cache_dir,
        )
        print(
            f"[{meeting_id}] TP={result['tp']} FP={result['fp']} FN={result['fn']} "
            f"P={format_pct(result['precision'])} R={format_pct(result['recall'])} F1={format_pct(result['f1'])}"
        )
        per_meeting_results.append(result)

    total_tp = sum(m["tp"] for m in per_meeting_results)
    total_fp = sum(m["fp"] for m in per_meeting_results)
    total_fn = sum(m["fn"] for m in per_meeting_results)
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    agg_f1 = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (agg_precision + agg_recall) else 0.0
    aggregate = {
        "num_predicted": sum(m["num_predicted"] for m in per_meeting_results),
        "num_gold": sum(m["num_gold"] for m in per_meeting_results),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": agg_precision, "recall": agg_recall, "f1": agg_f1,
    }

    total_matched_precision = sum(m["entity_linking"]["matched_for_precision"] for m in per_meeting_results)
    total_matched_recall = sum(m["entity_linking"]["matched_for_recall"] for m in per_meeting_results)
    total_pred_ent = sum(m["entity_linking"]["num_predicted_entities"] for m in per_meeting_results)
    total_gold_ent = sum(m["entity_linking"]["num_gold_entities"] for m in per_meeting_results)
    agg_entity = {
        "num_predicted_entities": total_pred_ent,
        "num_gold_entities": total_gold_ent,
        "matched_for_precision": total_matched_precision,
        "matched_for_recall": total_matched_recall,
        "precision": total_matched_precision / total_pred_ent if total_pred_ent else 0.0,
        "recall": total_matched_recall / total_gold_ent if total_gold_ent else 0.0,
    }

    agg_entity_by_type_raw = defaultdict(lambda: {
        "num_predicted_entities": 0, "num_gold_entities": 0, "matched_for_precision": 0, "matched_for_recall": 0,
    })
    for m in per_meeting_results:
        for entity_type, e in m["entity_linking_by_type"].items():
            agg_entity_by_type_raw[entity_type]["num_predicted_entities"] += e["num_predicted_entities"]
            agg_entity_by_type_raw[entity_type]["num_gold_entities"] += e["num_gold_entities"]
            agg_entity_by_type_raw[entity_type]["matched_for_precision"] += e["matched_for_precision"]
            agg_entity_by_type_raw[entity_type]["matched_for_recall"] += e["matched_for_recall"]
    agg_entity_by_type = {}
    for entity_type, e in agg_entity_by_type_raw.items():
        agg_entity_by_type[entity_type] = {
            **e,
            "precision": e["matched_for_precision"] / e["num_predicted_entities"] if e["num_predicted_entities"] else 0.0,
            "recall": e["matched_for_recall"] / e["num_gold_entities"] if e["num_gold_entities"] else 0.0,
        }

    graph = build_graph(merged_pred_triples)
    coherence = graph_coherence_metrics(graph)

    report_md = build_markdown_report(
        per_meeting_results, aggregate, agg_entity, agg_entity_by_type, coherence, args.judge_model, args.extract_model,
        args.similarity_threshold, not args.no_semantic_merge,
    )
    report_path = os.path.join(args.results_dir, "kg_extraction_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nSaved markdown report to {report_path}")

    raw_results_path = os.path.join(args.results_dir, "kg_extraction_raw.json")
    with open(raw_results_path, "w", encoding="utf-8") as f:
        json.dump({
            "per_meeting": per_meeting_results,
            "aggregate": aggregate,
            "entity_linking_aggregate": agg_entity,
            "entity_linking_aggregate_by_type": agg_entity_by_type,
            "graph_coherence": coherence,
            "semantic_merge": {
                "enabled": not args.no_semantic_merge,
                "similarity_threshold": args.similarity_threshold,
                "entity_types": list(SEMANTIC_MERGE_TYPES),
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved raw matched/unmatched triples to {raw_results_path}")

    print(
        f"\nEntity linking (pooled): P={format_pct(agg_entity['precision'])} R={format_pct(agg_entity['recall'])}"
    )
    print(
        f"Relation extraction (pooled): P={format_pct(agg_precision)} R={format_pct(agg_recall)} "
        f"F1={format_pct(agg_f1)} (TP={total_tp} FP={total_fp} FN={total_fn})"
    )
    print(f"Graph coherence: {coherence}")


if __name__ == "__main__":
    main()
