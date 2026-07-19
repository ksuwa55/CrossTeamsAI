"""
Evaluate the causal-event extraction pipeline (02_causal_modeling/extract_variables.py)
against hand-labeled ground-truth cause/effect pairs for the synthetic meeting transcripts.

Pipeline:
  1. For each meeting_XX_*.json transcript, run enrich_transcript() + extract_causal_events()
     from the existing pipeline (unmodified) to get predicted (cause, effect) pairs.
     Predictions are saved to --output-dir as <meeting_id>.causal_events.json.
  2. Load the corresponding meeting_XX_*.labels.json ground-truth pairs.
  3. Match predicted pairs to gold pairs using an LLM judge (see MATCHING METHOD below).
  4. Compute precision / recall / F1 per meeting and pooled (micro) across all meetings,
     and collect the false positives / false negatives for manual review.
  5. Write a markdown report (per-meeting + aggregate tables) and a JSON file with the
     raw matched/unmatched pairs.

MATCHING METHOD: LLM judge, not embedding similarity.
  Gold and predicted phrases are short, paraphrased descriptions of a cause or an effect
  (e.g. "legal sign-off is pending" vs "the data-sharing agreement hasn't been approved").
  Embedding cosine similarity struggles here because it mostly picks up topical overlap
  ("legal", "sign-off", "approval" all share a topic) and can't reliably tell "same
  causal claim, different wording" apart from "related but distinct claim in the same
  topic area" -- exactly the confusable case we need to get right, and the one that
  determines the whole meeting's recall since most meetings only have 1-3 gold pairs.
  An LLM judge that reads both phrases in context handles paraphrase and directionality
  ("A causes B" vs "B causes A") explicitly, which a bag-of-embeddings similarity score
  does not. The dataset is small (10 meetings, a handful of pairs each), so the extra
  judge calls are cheap -- one call per predicted pair, judging it against all gold pairs
  for that meeting at once (batched into a single structured-JSON response), rather than
  one call per (predicted, gold) combination.
  gpt-4o-mini is used for judging (distinct from the gpt-3.5-turbo extraction model) since
  it follows the structured-JSON judging instructions more reliably; responses are cached
  on disk (cache_causal_eval/) exactly like the pipeline's own summarizer cache, so re-runs
  of this script don't re-spend API calls.

Matching is one-to-one: a predicted pair can match at most one gold pair and vice versa.
Given the judge's boolean compatibility matrix per meeting, we take a maximum-cardinality
bipartite matching (via networkx) rather than greedy first-match, so that one meeting's
matching choice doesn't spuriously starve a later pair of its only valid match.

CHAIN-AWARE RECALL (graph-level, alongside the strict pair-level P/R/F1 above):
  False-negative inspection showed most misses aren't wording mismatches but "which link
  in the causal chain" mismatches -- the model traces one hop further upstream, or stops
  one hop short, of the gold label's chosen cause/effect scoping. The strict pair-level
  match credits none of that, since it requires one predicted pair to match the gold
  pair's cause AND effect at once. Chain-aware recall instead builds a small directed
  graph per meeting from the predicted (cause, effect) pairs (raw predicted text as
  nodes, one edge per pair -- not canonicalized, that happens downstream in
  build_causal_graph.py) and checks whether a *path* exists from a node matching gold's
  cause to a node matching gold's effect, crediting multi-hop chains that reach the same
  conclusion via an intermediate node. Node equivalence ("is this the same real-world
  thing as gold's cause/effect") is judged by the same LLM-judge approach as the
  pair-level matching above (single-phrase comparison instead of whole-pair comparison).
  This is reported as a separate metric, not a replacement for pair-level P/R/F1: it has
  no precision/F1 counterpart, since graph reachability in raw uncanonicalized text
  doesn't have a well-defined "false positive edge".

Does not modify extract_causal_events, canonicalize_node, or any other pipeline logic --
this script only imports and calls them.
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import networkx as nx
from networkx.algorithms import bipartite

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "02_causal_modeling"))
sys.path.insert(0, os.path.join(REPO_ROOT, "01_summarization"))

from extract_variables import enrich_transcript, extract_causal_events, save_output  # noqa: E402
from summarizer import MeetingSummarizer  # noqa: E402


# ─── Discovery / I/O ───────────────────────────────────────────────

def discover_meetings(transcripts_dir: str) -> List[Tuple[str, str, str]]:
    """Returns list of (meeting_id, transcript_path, labels_path), sorted by meeting_id."""
    meetings = []
    for transcript_path in sorted(glob.glob(os.path.join(transcripts_dir, "*.json"))):
        if transcript_path.endswith(".labels.json"):
            continue
        meeting_id = os.path.splitext(os.path.basename(transcript_path))[0]
        labels_path = os.path.join(transcripts_dir, f"{meeting_id}.labels.json")
        if not os.path.exists(labels_path):
            print(f"[skip] no labels file for {meeting_id}")
            continue
        meetings.append((meeting_id, transcript_path, labels_path))
    return meetings


def load_labels(labels_path: str) -> List[Dict]:
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── LLM-judge matching ────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating whether a predicted cause-effect pair from a meeting transcript "
    "matches any pairs in a list of ground-truth cause-effect pairs. Two pairs match only "
    "if they describe substantially the same causal relationship: the cause must be the "
    "same underlying cause (not just a related topic), and the effect must be the same "
    "underlying effect, in the same direction (do not count it as a match if the cause "
    "and effect are reversed). Wording, phrasing, and level of detail may differ. "
    "For each ground-truth pair, briefly reason about whether it matches (one short "
    "sentence each), then on the LAST line output ONLY a JSON array of booleans, in order, "
    "no markdown fences."
)


def build_judge_prompt(pred_pair: Dict, gold_pairs: List[Dict]) -> str:
    gold_list = "\n".join(
        f"{i + 1}. cause: \"{g['cause']}\" | effect: \"{g['effect']}\""
        for i, g in enumerate(gold_pairs)
    )
    return (
        f"Predicted pair:\ncause: \"{pred_pair['cause']}\" | effect: \"{pred_pair['effect']}\"\n\n"
        f"Ground-truth pairs:\n{gold_list}\n\n"
        f"For each of the {len(gold_pairs)} ground-truth pairs above, does the predicted pair "
        f"capture the same causal relationship? Reason briefly per pair, then end with a JSON "
        f"array of exactly {len(gold_pairs)} booleans (true/false), in order, on its own line."
    )


def parse_judge_response(raw: str, expected_len: int) -> List[bool]:
    cleaned = raw.strip()
    cleaned = re.sub(r"```(json)?", "", cleaned).strip()
    # The judge is asked to reason first and end with a JSON array; take the last
    # bracketed array in the response rather than requiring the whole response to be JSON.
    array_matches = re.findall(r"\[[^\[\]]*\]", cleaned, flags=re.DOTALL)
    if not array_matches:
        return [False] * expected_len
    try:
        data = json.loads(array_matches[-1])
    except json.JSONDecodeError:
        return [False] * expected_len
    if not isinstance(data, list) or len(data) != expected_len:
        return [False] * expected_len
    return [bool(x) for x in data]


def judge_compatibility_matrix(
    summarizer: MeetingSummarizer,
    pred_events: List[Dict],
    gold_events: List[Dict],
    judge_model: str,
) -> List[List[bool]]:
    """matrix[i][j] = True if predicted pair i matches gold pair j, per the LLM judge."""
    matrix = []
    for pred in pred_events:
        prompt = build_judge_prompt(pred, gold_events)
        raw = summarizer.run_summarizer(
            prompt,
            model=judge_model,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=400,
        )
        matrix.append(parse_judge_response(raw, len(gold_events)))
    return matrix


def max_bipartite_matching(matrix: List[List[bool]]) -> List[Tuple[int, int]]:
    """Given matrix[i][j] compatibility, returns list of (pred_idx, gold_idx) for a
    maximum-cardinality one-to-one matching."""
    if not matrix or not matrix[0]:
        return []
    g = nx.Graph()
    pred_nodes = [f"p{i}" for i in range(len(matrix))]
    gold_nodes = [f"g{j}" for j in range(len(matrix[0]))]
    g.add_nodes_from(pred_nodes, bipartite=0)
    g.add_nodes_from(gold_nodes, bipartite=1)
    for i, row in enumerate(matrix):
        for j, ok in enumerate(row):
            if ok:
                g.add_edge(f"p{i}", f"g{j}")
    if g.number_of_edges() == 0:
        return []
    matching = bipartite.maximum_matching(g, top_nodes=pred_nodes)
    pairs = []
    for node, partner in matching.items():
        if node.startswith("p"):
            pairs.append((int(node[1:]), int(partner[1:])))
    return pairs


# ─── Per-meeting evaluation ────────────────────────────────────────

def evaluate_meeting(
    meeting_id: str,
    pred_events: List[Dict],
    gold_events: List[Dict],
    summarizer: MeetingSummarizer,
    judge_model: str,
) -> Dict:
    if pred_events and gold_events:
        matrix = judge_compatibility_matrix(summarizer, pred_events, gold_events, judge_model)
        matches = max_bipartite_matching(matrix)
    else:
        matches = []

    matched_pred_idxs = {i for i, _ in matches}
    matched_gold_idxs = {j for _, j in matches}

    true_positives = [
        {"predicted": pred_events[i], "ground_truth": gold_events[j]} for i, j in matches
    ]
    false_positives = [
        pred_events[i] for i in range(len(pred_events)) if i not in matched_pred_idxs
    ]
    false_negatives = [
        gold_events[j] for j in range(len(gold_events)) if j not in matched_gold_idxs
    ]

    tp, fp, fn = len(true_positives), len(false_positives), len(false_negatives)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "meeting_id": meeting_id,
        "num_predicted": len(pred_events),
        "num_gold": len(gold_events),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


# ─── Graph-level (chain-aware) evaluation ──────────────────────────

NODE_JUDGE_SYSTEM_PROMPT = (
    "You are evaluating whether any of several candidate node phrases from a meeting "
    "transcript's causal graph describe the same real-world thing as a target phrase (a "
    "cause or an effect taken from a hand-labeled ground-truth pair). A candidate matches "
    "the target if it refers to the same underlying event, state, decision, or condition, "
    "even if worded differently or at a different level of detail. For each candidate, "
    "briefly reason about whether it matches (one short sentence each), then on the LAST "
    "line output ONLY a JSON array of booleans, in order, no markdown fences."
)


def build_node_judge_prompt(target_phrase: str, node_texts: List[str]) -> str:
    node_list = "\n".join(f"{i + 1}. \"{n}\"" for i, n in enumerate(node_texts))
    return (
        f"Target phrase (a cause or effect from a ground-truth pair):\n\"{target_phrase}\"\n\n"
        f"Candidate node phrases from the predicted causal graph:\n{node_list}\n\n"
        f"For each of the {len(node_texts)} candidate phrases above, does it describe the "
        f"same real-world thing as the target phrase? Reason briefly per phrase, then end "
        f"with a JSON array of exactly {len(node_texts)} booleans (true/false), in order, "
        "on its own line."
    )


def judge_node_matches(
    summarizer: MeetingSummarizer,
    target_phrase: str,
    node_texts: List[str],
    judge_model: str,
) -> List[bool]:
    """Returns a bool per node_texts entry: True if the LLM judge considers it the same
    real-world thing as target_phrase. Reuses parse_judge_response unmodified."""
    if not node_texts:
        return []
    prompt = build_node_judge_prompt(target_phrase, node_texts)
    raw = summarizer.run_summarizer(
        prompt,
        model=judge_model,
        system_prompt=NODE_JUDGE_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=400,
    )
    return parse_judge_response(raw, len(node_texts))


def build_predicted_graph(pred_events: List[Dict]) -> "nx.DiGraph":
    """Builds a directed graph from raw predicted (cause, effect) text -- one edge per
    predicted pair. Nodes are NOT canonicalized (that only happens downstream in
    build_causal_graph.py); this is deliberately out of scope here."""
    graph = nx.DiGraph()
    for pair in pred_events:
        graph.add_edge(pair["cause"], pair["effect"])
    return graph


def evaluate_meeting_chain_aware(
    meeting_id: str,
    pred_events: List[Dict],
    gold_events: List[Dict],
    summarizer: MeetingSummarizer,
    judge_model: str,
) -> Dict:
    """For each gold pair, checks whether a path exists in the predicted graph from a node
    matching gold's cause to a node matching gold's effect, rather than requiring a single
    predicted pair to match the whole gold pair at once. Credits multi-hop chains where the
    model traced the same relationship through an intermediate node the gold label skipped
    over (or vice versa)."""
    graph = build_predicted_graph(pred_events)
    node_texts = sorted(set(graph.nodes()))

    hits = []
    misses = []
    for gold in gold_events:
        if not node_texts:
            misses.append(gold)
            continue
        cause_matches = judge_node_matches(summarizer, gold["cause"], node_texts, judge_model)
        effect_matches = judge_node_matches(summarizer, gold["effect"], node_texts, judge_model)
        cause_nodes = [n for n, ok in zip(node_texts, cause_matches) if ok]
        effect_nodes = [n for n, ok in zip(node_texts, effect_matches) if ok]

        found_path = any(nx.has_path(graph, c, e) for c in cause_nodes for e in effect_nodes)
        (hits if found_path else misses).append(gold)

    hit, miss = len(hits), len(misses)
    chain_recall = hit / (hit + miss) if (hit + miss) else 0.0

    return {
        "meeting_id": meeting_id,
        "num_gold": len(gold_events),
        "chain_hits": hit,
        "chain_misses": miss,
        "chain_recall": chain_recall,
        "chain_missed_pairs": misses,
    }


# ─── Report generation ─────────────────────────────────────────────

def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_markdown_report(per_meeting: List[Dict], aggregate: Dict, judge_model: str, extract_model: str) -> str:
    lines = []
    lines.append("# Causal Event Extraction Evaluation\n")
    lines.append(
        f"Extraction model: `{extract_model}` (unchanged pipeline). "
        f"Matching: LLM judge (`{judge_model}`), one-to-one maximum bipartite matching per meeting.\n"
    )
    lines.append("## Per-meeting results\n")
    lines.append("| Meeting | Predicted | Gold | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in per_meeting:
        lines.append(
            f"| {m['meeting_id']} | {m['num_predicted']} | {m['num_gold']} | {m['tp']} | "
            f"{m['fp']} | {m['fn']} | {format_pct(m['precision'])} | {format_pct(m['recall'])} | "
            f"{format_pct(m['f1'])} |"
        )
    lines.append(
        f"| **Aggregate (pooled)** | {aggregate['num_predicted']} | {aggregate['num_gold']} | "
        f"{aggregate['tp']} | {aggregate['fp']} | {aggregate['fn']} | "
        f"{format_pct(aggregate['precision'])} | {format_pct(aggregate['recall'])} | "
        f"{format_pct(aggregate['f1'])} |"
    )
    lines.append(
        "\n*Aggregate is pooled (micro-averaged): TP/FP/FN are summed across all meetings "
        "before computing precision/recall/F1.*\n"
    )

    lines.append("## Chain-aware recall (graph reachability)\n")
    lines.append(
        "For each gold pair, checks whether a path exists in the predicted graph (raw, "
        "uncanonicalized predicted cause/effect text as nodes, one edge per predicted pair) "
        "from a node matching gold's cause to a node matching gold's effect -- node "
        "equivalence is judged by the same LLM-judge approach as the pair-level matching "
        "above, applied to single phrases instead of whole pairs. Unlike the pair-level "
        "metric, this credits multi-hop chains: it does not require one predicted pair to "
        "match the gold pair's exact cause/effect scoping, so it captures cases where the "
        "model traced the same causal chain one hop further upstream or downstream than "
        "the gold label (see docs/phase2/pipeline-flow-and-results.md).\n"
    )
    lines.append("| Meeting | Gold | Chain Hits | Chain Misses | Chain Recall |")
    lines.append("|---|---|---|---|---|")
    for m in per_meeting:
        ca = m["chain_aware"]
        lines.append(
            f"| {m['meeting_id']} | {ca['num_gold']} | {ca['chain_hits']} | "
            f"{ca['chain_misses']} | {format_pct(ca['chain_recall'])} |"
        )
    ca_agg = aggregate["chain_aware"]
    lines.append(
        f"| **Aggregate (pooled)** | {ca_agg['num_gold']} | {ca_agg['chain_hits']} | "
        f"{ca_agg['chain_misses']} | {format_pct(ca_agg['chain_recall'])} |"
    )
    lines.append(
        "\n*Chain-aware recall has no precision/F1 counterpart: graph reachability over "
        "raw uncanonicalized text doesn't have a well-defined notion of a \"false positive\" "
        "edge, so this metric is recall-only (see docs/phase2/evaluation-strategy.md).*\n"
    )

    lines.append("## Chain-aware misses (no path found, even allowing multi-hop)\n")
    any_chain_miss = False
    for m in per_meeting:
        for gold in m["chain_aware"]["chain_missed_pairs"]:
            any_chain_miss = True
            lines.append(
                f"- **{m['meeting_id']}**: cause=\"{gold['cause']}\" -> effect=\"{gold['effect']}\""
            )
    if not any_chain_miss:
        lines.append("- none")
    lines.append("")

    lines.append("## False positives (predicted, no matching ground truth)\n")
    any_fp = False
    for m in per_meeting:
        for fp in m["false_positives"]:
            any_fp = True
            lines.append(f"- **{m['meeting_id']}**: cause=\"{fp['cause']}\" -> effect=\"{fp['effect']}\" (t={fp.get('timestamp', '')})")
    if not any_fp:
        lines.append("- none")
    lines.append("")

    lines.append("## False negatives (ground truth missed by the model)\n")
    any_fn = False
    for m in per_meeting:
        for fn in m["false_negatives"]:
            any_fn = True
            lines.append(f"- **{m['meeting_id']}**: cause=\"{fn['cause']}\" -> effect=\"{fn['effect']}\" (t={fn.get('timestamp', '')})")
    if not any_fn:
        lines.append("- none")
    lines.append("")

    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcripts-dir", default=os.path.join(REPO_ROOT, "data", "synthetic_transcripts"))
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "output", "causal_events"),
                         help="Where predicted causal-events JSON per meeting is saved")
    parser.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "eval", "results", "causal_extraction"),
                         help="Where the markdown report and raw matched/unmatched JSON are saved")
    parser.add_argument("--extract-model", default="gpt-3.5-turbo", help="Model used by extract_causal_events (pipeline default)")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Model used to judge predicted-vs-gold pair equivalence")
    parser.add_argument("--window-before", type=int, default=2)
    parser.add_argument("--window-after", type=int, default=1)
    parser.add_argument("--extraction-cache-dir", default=os.path.join(REPO_ROOT, "cache_causal"))
    parser.add_argument("--judge-cache-dir", default=os.path.join(REPO_ROOT, "cache_causal_eval"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    meetings = discover_meetings(args.transcripts_dir)
    print(f"Found {len(meetings)} meetings with transcript + labels.")

    judge_summarizer = MeetingSummarizer(cache_dir=args.judge_cache_dir)

    per_meeting_results = []
    for meeting_id, transcript_path, labels_path in meetings:
        print(f"\n[{meeting_id}] enriching + extracting causal events...")
        enriched = enrich_transcript(transcript_path)
        pred_events = extract_causal_events(
            enriched,
            model=args.extract_model,
            window_before=args.window_before,
            window_after=args.window_after,
            cache_dir=args.extraction_cache_dir,
            meeting_id=meeting_id,
        )
        pred_out_path = os.path.join(args.output_dir, f"{meeting_id}.causal_events.json")
        save_output(pred_events, pred_out_path)

        gold_events = load_labels(labels_path)

        print(f"[{meeting_id}] judging {len(pred_events)} predicted vs {len(gold_events)} gold pairs...")
        result = evaluate_meeting(meeting_id, pred_events, gold_events, judge_summarizer, args.judge_model)
        print(
            f"[{meeting_id}] TP={result['tp']} FP={result['fp']} FN={result['fn']} "
            f"P={format_pct(result['precision'])} R={format_pct(result['recall'])} F1={format_pct(result['f1'])}"
        )

        chain_result = evaluate_meeting_chain_aware(
            meeting_id, pred_events, gold_events, judge_summarizer, args.judge_model
        )
        print(
            f"[{meeting_id}] chain-aware recall: {chain_result['chain_hits']}/{chain_result['num_gold']} "
            f"({format_pct(chain_result['chain_recall'])})"
        )
        result["chain_aware"] = chain_result
        per_meeting_results.append(result)

    total_tp = sum(m["tp"] for m in per_meeting_results)
    total_fp = sum(m["fp"] for m in per_meeting_results)
    total_fn = sum(m["fn"] for m in per_meeting_results)
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    agg_f1 = (
        2 * agg_precision * agg_recall / (agg_precision + agg_recall)
        if (agg_precision + agg_recall)
        else 0.0
    )
    total_chain_hit = sum(m["chain_aware"]["chain_hits"] for m in per_meeting_results)
    total_chain_miss = sum(m["chain_aware"]["chain_misses"] for m in per_meeting_results)
    agg_chain_recall = (
        total_chain_hit / (total_chain_hit + total_chain_miss)
        if (total_chain_hit + total_chain_miss)
        else 0.0
    )

    aggregate = {
        "num_predicted": sum(m["num_predicted"] for m in per_meeting_results),
        "num_gold": sum(m["num_gold"] for m in per_meeting_results),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": agg_precision,
        "recall": agg_recall,
        "f1": agg_f1,
        "chain_aware": {
            "num_gold": sum(m["chain_aware"]["num_gold"] for m in per_meeting_results),
            "chain_hits": total_chain_hit,
            "chain_misses": total_chain_miss,
            "chain_recall": agg_chain_recall,
        },
    }

    report_md = build_markdown_report(per_meeting_results, aggregate, args.judge_model, args.extract_model)
    report_path = os.path.join(args.results_dir, "causal_extraction_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nSaved markdown report to {report_path}")

    raw_results_path = os.path.join(args.results_dir, "causal_extraction_raw.json")
    with open(raw_results_path, "w", encoding="utf-8") as f:
        json.dump({"per_meeting": per_meeting_results, "aggregate": aggregate}, f, indent=2, ensure_ascii=False)
    print(f"Saved raw matched/unmatched pairs to {raw_results_path}")

    print(
        f"\nAggregate (pooled): P={format_pct(agg_precision)} R={format_pct(agg_recall)} "
        f"F1={format_pct(agg_f1)} (TP={total_tp} FP={total_fp} FN={total_fn})"
    )
    print(
        f"Chain-aware recall (pooled): {format_pct(agg_chain_recall)} "
        f"({total_chain_hit}/{total_chain_hit + total_chain_miss})"
    )


if __name__ == "__main__":
    main()
