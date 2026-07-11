"""
For each false negative from evaluate_causal_extraction.py (a ground-truth pair the
model's predictions didn't match), find the single closest predicted pair from that
meeting -- even though it fell short of a match -- and ask the judge to explain why,
with a verdict:

  - phrasing_mismatch: the predicted pair captures the same relationship; the miss is
    likely an artifact of wording/threshold, not a real extraction failure.
  - partial_match:     the predicted pair captures part of it (cause OR effect, not both,
                        or a related-but-narrower/broader claim).
  - genuine_miss:       no predicted pair meaningfully overlaps; the pipeline didn't
                        surface this relationship at all.

Reads eval/results/causal_extraction/causal_extraction_raw.json (produced by
evaluate_causal_extraction.py) and output/causal_events/<meeting_id>.causal_events.json
for the full predicted-pair list per meeting. Does not re-run extraction or touch
pipeline logic.
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "01_summarization"))

from summarizer import MeetingSummarizer  # noqa: E402


CLOSEST_SYSTEM_PROMPT = (
    "You are helping a human auditor understand why an automated cause-effect extractor "
    "missed a ground-truth causal pair from a meeting transcript. You will be given the "
    "ground-truth pair and the full list of pairs the extractor predicted for that same "
    "meeting. Identify the SINGLE predicted pair that is closest in meaning to the "
    "ground truth, even if it falls short of a full match (pick null only if none are "
    "even topically related). Then explain in 2-3 sentences why it does or doesn't count "
    "as the same causal relationship, and assign a verdict:\n"
    "  \"phrasing_mismatch\" - it IS the same relationship, just worded differently or at "
    "a different level of detail; a human would call this a match.\n"
    "  \"partial_match\" - it captures the cause OR the effect correctly but not both, or "
    "a related-but-distinct claim.\n"
    "  \"genuine_miss\" - no predicted pair meaningfully overlaps with the ground truth.\n"
    "Respond with ONLY a JSON object: "
    '{"closest_index": <1-based int or null>, "verdict": "phrasing_mismatch"|"partial_match"|"genuine_miss", "reasoning": "..."}'
)


def build_closest_prompt(gold_pair: Dict, pred_events: List[Dict]) -> str:
    pred_list = "\n".join(
        f"{i + 1}. cause: \"{p['cause']}\" | effect: \"{p['effect']}\""
        for i, p in enumerate(pred_events)
    )
    return (
        f"Ground-truth pair:\ncause: \"{gold_pair['cause']}\" | effect: \"{gold_pair['effect']}\"\n\n"
        f"Predicted pairs from this meeting:\n{pred_list}\n\n"
        "Which predicted pair is closest to the ground truth, and why doesn't it count as "
        "a full match (or does it)?"
    )


def parse_closest_response(raw: str) -> Dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"```(json)?", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {"closest_index": None, "verdict": "genuine_miss", "reasoning": "(judge response unparseable)"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"closest_index": None, "verdict": "genuine_miss", "reasoning": "(judge response unparseable)"}
    return {
        "closest_index": data.get("closest_index"),
        "verdict": data.get("verdict", "genuine_miss"),
        "reasoning": data.get("reasoning", ""),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results", default=os.path.join(REPO_ROOT, "eval", "results", "causal_extraction", "causal_extraction_raw.json"))
    parser.add_argument("--predictions-dir", default=os.path.join(REPO_ROOT, "output", "causal_events"))
    parser.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "eval", "results", "causal_extraction"))
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--judge-cache-dir", default=os.path.join(REPO_ROOT, "cache_causal_eval"))
    args = parser.parse_args()

    with open(args.raw_results, "r", encoding="utf-8") as f:
        raw_results = json.load(f)

    summarizer = MeetingSummarizer(cache_dir=args.judge_cache_dir)

    inspections = []
    for m in raw_results["per_meeting"]:
        fns = m["false_negatives"]
        if not fns:
            continue
        meeting_id = m["meeting_id"]
        pred_path = os.path.join(args.predictions_dir, f"{meeting_id}.causal_events.json")
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_events = json.load(f)

        for gold in fns:
            closest_pred: Optional[Dict] = None
            if pred_events:
                prompt = build_closest_prompt(gold, pred_events)
                raw = summarizer.run_summarizer(
                    prompt,
                    model=args.judge_model,
                    system_prompt=CLOSEST_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=300,
                )
                parsed = parse_closest_response(raw)
                idx = parsed["closest_index"]
                if isinstance(idx, int) and 1 <= idx <= len(pred_events):
                    closest_pred = pred_events[idx - 1]
            else:
                parsed = {"closest_index": None, "verdict": "genuine_miss", "reasoning": "(no predictions for this meeting)"}

            inspections.append({
                "meeting_id": meeting_id,
                "gold": gold,
                "closest_predicted": closest_pred,
                "verdict": parsed["verdict"],
                "reasoning": parsed["reasoning"],
            })

    # ─── report ───
    lines = ["# False Negative Inspection\n"]
    lines.append(
        "For each ground-truth pair the model missed: your original label, the single "
        "closest predicted pair from that meeting (even though it didn't match), and the "
        "judge's reasoning for why it fell short.\n"
    )
    verdict_label = {
        "phrasing_mismatch": "PHRASING MISMATCH (likely a real match, wording tripped the judge)",
        "partial_match": "PARTIAL MATCH (captures part of the relationship)",
        "genuine_miss": "GENUINE MISS (no meaningful overlap)",
    }
    for item in inspections:
        lines.append(f"## {item['meeting_id']}\n")
        lines.append(f"**Your label:**\n- cause: \"{item['gold']['cause']}\"\n- effect: \"{item['gold']['effect']}\"\n")
        if item["closest_predicted"]:
            lines.append(
                f"**Closest predicted pair:**\n- cause: \"{item['closest_predicted']['cause']}\"\n"
                f"- effect: \"{item['closest_predicted']['effect']}\"\n"
            )
        else:
            lines.append("**Closest predicted pair:** none found\n")
        lines.append(f"**Verdict:** {verdict_label.get(item['verdict'], item['verdict'])}\n")
        lines.append(f"**Judge reasoning:** {item['reasoning']}\n")

    report_path = os.path.join(args.results_dir, "false_negative_inspection.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved report to {report_path}")

    json_path = os.path.join(args.results_dir, "false_negative_inspection.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(inspections, f, indent=2, ensure_ascii=False)
    print(f"Saved raw inspection data to {json_path}")

    counts = {}
    for item in inspections:
        counts[item["verdict"]] = counts.get(item["verdict"], 0) + 1
    print(f"\n{len(inspections)} false negatives inspected: {counts}")


if __name__ == "__main__":
    main()
