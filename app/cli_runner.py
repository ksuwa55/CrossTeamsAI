import os
import sys
import re
import json
import argparse
import random
from math import ceil
from typing import List, Dict, DefaultDict, Optional
from collections import defaultdict, Counter

# --- Make 01_summarization importable (contains summarizer.py and qmsum_loader.py) ---
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PKG_DIR = os.path.join(REPO_ROOT, "01_summarization")
sys.path.insert(0, PKG_DIR)

from summarizer import MeetingSummarizer
from qmsum_loader import iter_qmsum


# ---------------- Utilities ----------------
def split_chunks(text: str, chunk_chars: int = 12000, overlap_chars: int = 1000) -> List[str]:
    if chunk_chars <= 0:
        return [text]
    out = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_chars, n)
        out.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap_chars)
    return out

def keyword_prefilter(transcript: str, query: str, window: int = 2) -> str:
    """Keep lines containing any query token (+/- window neighbors). Falls back if too small."""
    lines = transcript.splitlines()
    toks = [t for t in query.lower().split() if len(t) > 2]
    if not toks or not lines:
        return transcript
    keep = [False] * len(lines)
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(t in low for t in toks):
            for j in range(max(0, i - window), min(len(lines), i + window + 1)):
                keep[j] = True
    kept = [l for l, k in zip(lines, keep) if k]
    if len(kept) < max(60, len(lines) // 12):
        return transcript
    return "\n".join(kept)

def top_bigrams(text: str, k: int = 8) -> List[str]:
    """Extract top bigrams (very cheap) to encourage reuse in phrasing (helps ROUGE-2)."""
    toks = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())
    toks = [t for t in toks if len(t) > 2]
    if len(toks) < 2:
        return []
    bigs = [" ".join(pair) for pair in zip(toks, toks[1:])]
    counts = Counter(bigs)
    stop2 = {
        "you know", "kind of", "sort of", "a lot",
        "at the", "in the", "on the", "for the",
        "to the", "and the", "of the", "with the"
    }
    cand = [b for b, _ in counts.most_common(60) if b not in stop2]
    return cand[:k]


# ---------------- Prompt builders ----------------
FEW_SHOT = (
    "### Example\n"
    "Query: 'What was decided about the release date?'\n"
    "Transcript slice:\n"
    "Alice: We agreed to aim for mid-July.\n"
    "Bob: The beta finishes end of June; July gives QA a buffer.\n"
    "Reference-style summary: The team decided to target a mid-July release, allowing time after the beta for QA.\n"
    "---\n\n"
)

def build_query_prompt(query: str, transcript_slice: str,
                       phrases_to_preserve: Optional[List[str]] = None,
                       use_few_shot: bool = False) -> str:
    prefix = FEW_SHOT if use_few_shot else ""
    keep = ""
    if phrases_to_preserve:
        keep = "Try to include these exact phrases if factually correct: " + "; ".join(phrases_to_preserve) + "\n\n"
    return (
        prefix +
        "Task: answer ONLY the query using facts from the transcript slice. "
        "Be concise but complete; include all key facts. Reuse wording from the slice when possible. No preamble.\n\n"
        f"{keep}"
        f"Query: '{query}'\n\n"
        f"Transcript slice:\n{transcript_slice}"
    )

def build_reduce_prompt(query: str, partial_summaries: List[str],
                        phrases_to_preserve: Optional[List[str]] = None) -> str:
    joined = "\n\n--- PART ---\n\n".join(partial_summaries)
    keep = ""
    if phrases_to_preserve:
        keep = "Try to include these exact phrases if factually correct: " + "; ".join(phrases_to_preserve) + "\n\n"
    return (
        "Task: merge the partial answers into one coherent answer. "
        "Eliminate duplicates; keep all key facts. Reuse wording from the parts when possible. No preamble.\n\n"
        f"{keep}"
        f"Query: '{query}'\n\n"
        f"Partial answers:\n{joined}"
    )

SYSTEM_PROMPT = (
    "You are a precise, query-focused meeting summarizer. "
    "Write concise, factual summaries. "
    "Do not add any information that is not supported by the transcript."
)


# ---------------- Input extractor ----------------
def extract_transcript_from_input_text(input_text: str) -> str:
    # qmsum_loader creates: "... 'query'\n\nTranscript:\n<full transcript>"
    t_prefix = "Transcript:\n"
    idx = input_text.find(t_prefix)
    if idx == -1:
        sep = "\n\n"
        j = input_text.find(sep)
        return input_text if j == -1 else input_text[j + len(sep):]
    return input_text[idx + len(t_prefix):]


# ---------------- Main runner ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmsum_split_dir", required=True, help="Path to QMSum/data/ALL/test (or val/train)")
    ap.add_argument("--model", default="gpt-3.5-turbo")
    ap.add_argument("--out_jsonl", default="output/qmsum_test_preds.jsonl")

    # Coverage defaults tuned to help recall on tiny subsets
    ap.add_argument("--chunk_chars", type=int, default=12000)
    ap.add_argument("--overlap_chars", type=int, default=1000)
    ap.add_argument("--max_chunks", type=int, default=6)

    # Subset defaults ~20 instances: ~10 meetings × 2 queries
    ap.add_argument("--sample_ratio", type=float, default=1.0)
    ap.add_argument("--max_meetings", type=int, default=10)          # selects ~10 meetings
    ap.add_argument("--max_queries_per_meeting", type=int, default=2) # 2 queries each
    ap.add_argument("--seed", type=int, default=42)

    # Behavior toggles (defaults chosen for ROUGE overlap)
    ap.add_argument("--few_shot", choices=["on","off"], default="off")
    ap.add_argument("--revise", choices=["on","off"], default="off")
    ap.add_argument("--prefilter", choices=["on","off"], default="on")
    ap.add_argument("--preserve_ngrams", choices=["on","off"], default="on")

    # Decoding knobs
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=320)

    args = ap.parse_args()
    random.seed(args.seed)

    ms = MeetingSummarizer(cache_dir="cache_qmsum")
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    # 1) Load all examples, group by meeting_id
    groups: DefaultDict[str, List[Dict]] = defaultdict(list)
    total_items = 0
    for ex in iter_qmsum(args.qmsum_split_dir):
        groups[ex["meeting_id"]].append(ex)
        total_items += 1

    all_meetings = list(groups.keys())
    n_meetings_target = min(len(all_meetings),
                            max(1, min(args.max_meetings, ceil(len(all_meetings) * args.sample_ratio))))
    sampled_meetings = random.sample(all_meetings, n_meetings_target)

    print(f"Meetings total: {len(all_meetings)}  -> using: {n_meetings_target}")
    print(f"Max queries per meeting: {args.max_queries_per_meeting}")

    # 2) Build the run list with per-meeting query cap
    run_items: List[Dict] = []
    for mid in sampled_meetings:
        run_items.extend(groups[mid][: args.max_queries_per_meeting])

    print(f"Queries to process: {len(run_items)} (of ~{total_items} total)")

    # 3) Run map–reduce per query with tuned coverage + behavior knobs
    written = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for ex in run_items:
            query = ex["query"]
            full_transcript = extract_transcript_from_input_text(ex["input_text"])

            # (a) Optional prefilter (cheap retrieval)
            text_for_chunking = keyword_prefilter(full_transcript, query, window=2) if args.prefilter == "on" else full_transcript

            # (b) Optional phrase preservation
            phrases = top_bigrams(text_for_chunking, k=8) if args.preserve_ngrams == "on" else None

            # (c) Chunk + summarize
            chunks = split_chunks(text_for_chunking, chunk_chars=args.chunk_chars, overlap_chars=args.overlap_chars)
            if len(chunks) > args.max_chunks:
                chunks = chunks[:args.max_chunks]

            partials = []
            for ch in chunks:
                map_prompt = build_query_prompt(query, ch, phrases_to_preserve=phrases, use_few_shot=(args.few_shot == "on"))
                part = ms.run_summarizer(
                    map_prompt,
                    model=args.model,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
                ).strip()
                partials.append(part)

            if len(partials) == 1:
                pred = partials[0]
            else:
                reduce_prompt = build_reduce_prompt(query, partials, phrases_to_preserve=phrases)
                pred = ms.run_summarizer(
                    reduce_prompt,
                    model=args.model,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens
                ).strip()

            if args.revise == "on":
                pred = ms.revise_summary(pred, query, model=args.model)

            row = {
                "meeting_id": ex["meeting_id"],
                "query_id": ex["query_id"],
                "query": query,
                "prediction": pred,
                "reference": ex["reference"],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} predictions to {args.out_jsonl}")


if __name__ == "__main__":
    main()
