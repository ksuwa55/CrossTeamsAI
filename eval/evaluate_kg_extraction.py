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
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "04_knowledgegraph_dashboard"))
sys.path.insert(0, os.path.join(REPO_ROOT, "01_summarization"))
sys.path.insert(0, os.path.join(REPO_ROOT, "02_causal_modeling"))

from extract_entities_relations import enrich_transcript, extract_kg_triples, node_key, save_output  # noqa: E402
from build_knowledge_graph import build_graph, graph_coherence_metrics  # noqa: E402
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

def triple_entity_keys(triples: List[Dict]) -> set:
    keys = set()
    for t in triples:
        keys.add(node_key(t["subject_type"], t["subject_text"]))
        keys.add(node_key(t["object_type"], t["object_text"]))
    return keys


def entity_linking_metrics(pred_triples: List[Dict], gold_triples: List[Dict]) -> Dict:
    pred_entities = triple_entity_keys(pred_triples)
    gold_entities = triple_entity_keys(gold_triples)
    tp = len(pred_entities & gold_entities)
    precision = tp / len(pred_entities) if pred_entities else 0.0
    recall = tp / len(gold_entities) if gold_entities else 0.0
    return {
        "num_predicted_entities": len(pred_entities),
        "num_gold_entities": len(gold_entities),
        "matched_entities": tp,
        "precision": precision,
        "recall": recall,
    }


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
) -> Dict:
    entity_metrics = entity_linking_metrics(pred_triples, gold_triples)

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
        "num_predicted": len(pred_triples),
        "num_gold": len(gold_triples),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


# ─── Report generation ─────────────────────────────────────────────

def build_markdown_report(per_meeting: List[Dict], aggregate: Dict, agg_entity: Dict,
                           coherence: Dict, judge_model: str, extract_model: str) -> str:
    lines = []
    lines.append("# Knowledge Graph Extraction Evaluation\n")
    lines.append(
        f"Extraction model: `{extract_model}` (unchanged pipeline). "
        f"Triple matching: LLM judge (`{judge_model}`), one-to-one maximum bipartite matching per meeting.\n"
    )

    lines.append("## Entity-linking (Top-N precision/recall)\n")
    lines.append("| Meeting | Pred Entities | Gold Entities | Matched | Precision | Recall |")
    lines.append("|---|---|---|---|---|---|")
    for m in per_meeting:
        e = m["entity_linking"]
        lines.append(
            f"| {m['meeting_id']} | {e['num_predicted_entities']} | {e['num_gold_entities']} | "
            f"{e['matched_entities']} | {format_pct(e['precision'])} | {format_pct(e['recall'])} |"
        )
    lines.append(
        f"| **Aggregate (pooled)** | {agg_entity['num_predicted_entities']} | {agg_entity['num_gold_entities']} | "
        f"{agg_entity['matched_entities']} | {format_pct(agg_entity['precision'])} | {format_pct(agg_entity['recall'])} |\n"
    )

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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    meetings = discover_meetings(args.transcripts_dir)
    print(f"Found {len(meetings)} meetings with transcript + kg_labels.")

    judge_summarizer = MeetingSummarizer(cache_dir=args.judge_cache_dir)

    per_meeting_results = []
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
        all_pred_triples.extend(pred_triples)

        gold_triples = load_labels(labels_path)

        print(f"[{meeting_id}] judging {len(pred_triples)} predicted vs {len(gold_triples)} gold triples...")
        result = evaluate_meeting(meeting_id, pred_triples, gold_triples, judge_summarizer, args.judge_model)
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

    total_matched = sum(m["entity_linking"]["matched_entities"] for m in per_meeting_results)
    total_pred_ent = sum(m["entity_linking"]["num_predicted_entities"] for m in per_meeting_results)
    total_gold_ent = sum(m["entity_linking"]["num_gold_entities"] for m in per_meeting_results)
    agg_entity = {
        "num_predicted_entities": total_pred_ent,
        "num_gold_entities": total_gold_ent,
        "matched_entities": total_matched,
        "precision": total_matched / total_pred_ent if total_pred_ent else 0.0,
        "recall": total_matched / total_gold_ent if total_gold_ent else 0.0,
    }

    graph = build_graph(all_pred_triples)
    coherence = graph_coherence_metrics(graph)

    report_md = build_markdown_report(per_meeting_results, aggregate, agg_entity, coherence, args.judge_model, args.extract_model)
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
            "graph_coherence": coherence,
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
