import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

# Phase 2's causal_events and Phase 4's kg_triples are both extracted from the
# same 10 transcripts in data/synthetic_transcripts/ and keyed by the same
# meeting_id (the transcript filename stem). Neither phase knows about the
# other's output, so this module is the one place that actually connects them:
# a causal event and a KG triple from the same meeting are linked when their
# text overlaps or their timestamps match, on the assumption that overlapping
# vocabulary/timing means they're describing the same real moment in the
# meeting. This is a heuristic (token overlap), not a learned entity linker —
# see docs/gaps_toward_academic_deliverable.md for that residual gap.

_WORD_RE = re.compile(r"[a-z0-9]+")


def _token_set(*texts: str) -> set:
    words = set()
    for text in texts:
        words.update(_WORD_RE.findall((text or "").lower()))
    return words


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _meeting_id_from_filename(path: str, suffix: str) -> str:
    base = os.path.basename(path)
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return base


def load_causal_events(glob_pattern: str) -> Dict[str, List[Dict]]:
    """Group causal events by meeting_id, inferring meeting_id from the
    filename when a given event doesn't carry one (older files may not)."""
    by_meeting: Dict[str, List[Dict]] = defaultdict(list)
    for path in sorted(glob.glob(glob_pattern)):
        meeting_id = _meeting_id_from_filename(path, ".causal_events.json")
        with open(path, "r", encoding="utf-8") as f:
            events = json.load(f)
        for e in events:
            e.setdefault("meeting_id", meeting_id)
            by_meeting[e["meeting_id"]].append(e)
    return dict(by_meeting)


def load_kg_triples(glob_pattern: str) -> Dict[str, List[Dict]]:
    by_meeting: Dict[str, List[Dict]] = defaultdict(list)
    for path in sorted(glob.glob(glob_pattern)):
        meeting_id = _meeting_id_from_filename(path, ".kg_triples.json")
        with open(path, "r", encoding="utf-8") as f:
            triples = json.load(f)
        for t in triples:
            t.setdefault("meeting_id", meeting_id)
            by_meeting[t["meeting_id"]].append(t)
    return dict(by_meeting)


def link_causal_and_kg(
    causal_events: List[Dict],
    kg_triples: List[Dict],
    min_score: float = 0.25,
) -> List[Dict]:
    """Pair up causal events and KG triples from the same meeting whose text
    overlaps (or whose timestamps match exactly), scored by Jaccard overlap
    over lowercased word tokens. Returns pairs sorted by score, descending."""
    links = []
    for event in causal_events:
        event_tokens = _token_set(event.get("cause", ""), event.get("effect", ""))
        for triple in kg_triples:
            triple_tokens = _token_set(
                triple.get("subject_text", ""),
                triple.get("object_text", ""),
                triple.get("quote", ""),
            )
            score = _jaccard(event_tokens, triple_tokens)
            timestamp_match = bool(event.get("timestamp")) and event.get("timestamp") == triple.get("timestamp")
            if timestamp_match:
                score = min(1.0, score + 0.25)
            if score >= min_score:
                links.append({
                    "meeting_id": event.get("meeting_id") or triple.get("meeting_id"),
                    "causal_event": event,
                    "kg_triple": triple,
                    "score": round(score, 3),
                    "timestamp_match": timestamp_match,
                })
    links.sort(key=lambda l: l["score"], reverse=True)
    return links


def link_all_meetings(
    causal_glob: str,
    kg_glob: str,
    min_score: float = 0.25,
) -> Dict[str, List[Dict]]:
    causal_by_meeting = load_causal_events(causal_glob)
    kg_by_meeting = load_kg_triples(kg_glob)
    meeting_ids = sorted(set(causal_by_meeting) & set(kg_by_meeting))
    return {
        meeting_id: link_causal_and_kg(
            causal_by_meeting[meeting_id], kg_by_meeting[meeting_id], min_score=min_score
        )
        for meeting_id in meeting_ids
    }


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
    causal_glob = os.path.join(REPO_ROOT, "output", "causal_events", "*.causal_events.json")
    kg_glob = os.path.join(REPO_ROOT, "output", "kg_triples", "*.kg_triples.json")

    all_links = link_all_meetings(causal_glob, kg_glob)
    total = sum(len(v) for v in all_links.values())
    print(f"Meetings with cross-links: {len(all_links)} / links found: {total}")
    for meeting_id, links in all_links.items():
        print(f"\n{meeting_id}: {len(links)} link(s)")
        for link in links[:3]:
            print(
                f"  score={link['score']} ts_match={link['timestamp_match']}\n"
                f"    causal: {link['causal_event']['cause']} -> {link['causal_event']['effect']}\n"
                f"    kg:     {link['kg_triple']['subject_text']} --{link['kg_triple']['relation']}--> {link['kg_triple']['object_text']}"
            )
