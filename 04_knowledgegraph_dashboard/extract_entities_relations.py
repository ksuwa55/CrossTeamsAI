import json
import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Set

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
SUMMARIZER_PKG = os.path.join(REPO_ROOT, "01_summarization")
CAUSAL_PKG = os.path.join(REPO_ROOT, "02_causal_modeling")
sys.path.insert(0, SUMMARIZER_PKG)
sys.path.insert(0, CAUSAL_PKG)

from summarizer import MeetingSummarizer
# Reuse Phase 2's regex prefilter + candidate-window selection unmodified: the
# same blocker/decision/next_action flags that surface causal events are also
# where decisions get made, issues get raised, and tasks get assigned.
from extract_variables import enrich_transcript, select_candidate_windows  # noqa: E402


ENTITY_TYPES: Set[str] = {"person", "decision", "issue", "task", "topic"}
RELATIONS: Set[str] = {
    "makes_decision", "raises_issue", "blocks", "causes", "assigned_to",
    "owns", "discusses", "resolves", "depends_on", "related_to",
}


# ----------------------------
# Entity normalization / resolution
# ----------------------------
_LEADING_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)

def normalize_entity(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = _LEADING_ARTICLES.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def node_key(entity_type: str, text: str) -> str:
    return f"{entity_type}:{normalize_entity(text)}"

def extract_speakers(enriched: List[Dict]) -> List[str]:
    seen = []
    for u in enriched:
        speaker = u.get("speaker", "")
        if speaker and speaker not in seen:
            seen.append(speaker)
    return seen

def resolve_person_alias(text: str, known_speakers: List[str]) -> str:
    """Collapse a person mention onto a transcript's known speaker name when
    one is a case-insensitive substring of the other (e.g. "Aiko" <-> "Aiko -
    PM"), so the same person doesn't fragment into multiple nodes."""
    mention = (text or "").strip().lower()
    if not mention:
        return text
    best = None
    for speaker in known_speakers:
        candidate = speaker.strip().lower()
        if mention == candidate:
            return speaker
        if mention in candidate or candidate in mention:
            if best is None or len(candidate) < len(best.strip().lower()):
                best = speaker
    return best if best else text


# ----------------------------
# LLM entity/relation-triple extraction
# ----------------------------
KG_SYSTEM_PROMPT = (
    "You are analyzing a team meeting or chat transcript for a software project. "
    "Your job is to extract a knowledge graph of (subject, relation, object) triples "
    "connecting people, decisions, issues/blockers, tasks, and topics. "
    "Only report a triple if the transcript excerpt gives clear evidence for it — do "
    "not invent entities or relations that aren't grounded in the text. "
    "Respond with ONLY a JSON array, no prose, no markdown fences."
)

_ENTITY_TYPES_STR = ", ".join(sorted(ENTITY_TYPES))
_RELATIONS_STR = ", ".join(sorted(RELATIONS))

def build_kg_prompt(window: List[Dict]) -> str:
    dialogue = "\n".join(
        f"{u['speaker']} [{u['timestamp']}]: {u['text']}" for u in window
    )
    return (
        "Transcript excerpt:\n"
        f"{dialogue}\n\n"
        "If this excerpt shows one or more entities related to each other, respond with "
        "a JSON array where each item has exactly these keys:\n"
        '  "subject_text": short name/phrase for the subject entity\n'
        f'  "subject_type": one of [{_ENTITY_TYPES_STR}]\n'
        f'  "relation": one of [{_RELATIONS_STR}]\n'
        '  "object_text": short name/phrase for the object entity\n'
        f'  "object_type": one of [{_ENTITY_TYPES_STR}]\n'
        '  "timestamp": the timestamp of the utterance where this is evidenced\n'
        '  "quote": the exact utterance text that grounds this triple\n'
        "If there is no clear relationship in this excerpt, respond with: []"
    )

def parse_kg_response(raw: str) -> List[Dict]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    triples = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subject_text = item.get("subject_text")
        object_text = item.get("object_text")
        relation = str(item.get("relation", "")).strip().lower()
        subject_type = str(item.get("subject_type", "")).strip().lower()
        object_type = str(item.get("object_type", "")).strip().lower()
        if not (subject_text and object_text):
            continue
        if relation not in RELATIONS or subject_type not in ENTITY_TYPES or object_type not in ENTITY_TYPES:
            continue
        triples.append({
            "subject_text": str(subject_text).strip(),
            "subject_type": subject_type,
            "relation": relation,
            "object_text": str(object_text).strip(),
            "object_type": object_type,
            "timestamp": str(item.get("timestamp", "")).strip(),
            "quote": str(item.get("quote", "")).strip(),
        })
    return triples


def extract_kg_triples(
    enriched: List[Dict],
    model: str = "gpt-3.5-turbo",
    window_before: int = 2,
    window_after: int = 1,
    cache_dir: str = "cache_kg",
    meeting_id: Optional[str] = None,
) -> List[Dict]:
    summarizer = MeetingSummarizer(cache_dir=cache_dir)
    windows = select_candidate_windows(enriched, window_before, window_after)
    known_speakers = extract_speakers(enriched)

    all_triples = []
    for window in windows:
        prompt = build_kg_prompt(window)
        raw = summarizer.run_summarizer(
            prompt,
            model=model,
            system_prompt=KG_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=500,
        )
        all_triples.extend(parse_kg_response(raw))

    unique_triples = []
    seen = set()
    for t in all_triples:
        if t["subject_type"] == "person":
            t["subject_text"] = resolve_person_alias(t["subject_text"], known_speakers)
        if t["object_type"] == "person":
            t["object_text"] = resolve_person_alias(t["object_text"], known_speakers)

        subject_key = node_key(t["subject_type"], t["subject_text"])
        object_key = node_key(t["object_type"], t["object_text"])
        if subject_key == object_key:
            continue
        key = (subject_key, t["relation"], object_key)
        if key in seen:
            continue
        seen.add(key)
        if meeting_id:
            t["meeting_id"] = meeting_id
        unique_triples.append(t)
    return unique_triples


def extract_topic_timeline(enriched: List[Dict], meeting_id: Optional[str] = None) -> List[Dict]:
    """No extra LLM calls: reuses enrich_transcript()'s regex-based `topic` tag
    per utterance to build a per-meeting topic-over-time timeline for the
    dashboard's topic-drift view."""
    timeline = []
    for u in enriched:
        entry = {"timestamp": u.get("timestamp", ""), "topic": u.get("topic", "general")}
        if meeting_id:
            entry["meeting_id"] = meeting_id
        timeline.append(entry)
    return timeline


def save_output(data, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved output to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input transcript JSON")
    parser.add_argument("--output", required=True, help="Path to save extracted triples JSON")
    parser.add_argument("--topic-output", default=None, help="Optional path to save the per-utterance topic timeline")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--window-before", type=int, default=2)
    parser.add_argument("--window-after", type=int, default=1)
    parser.add_argument("--skip-llm", action="store_true", help="Only run the regex prefilter + topic timeline; skip the LLM triple extraction call")
    args = parser.parse_args()

    enriched = enrich_transcript(args.input)
    meeting_id = os.path.splitext(os.path.basename(args.input))[0]

    if args.topic_output:
        save_output(extract_topic_timeline(enriched, meeting_id=meeting_id), args.topic_output)

    if args.skip_llm:
        save_output(enriched, args.output)
    else:
        triples = extract_kg_triples(
            enriched,
            model=args.model,
            window_before=args.window_before,
            window_after=args.window_after,
            meeting_id=meeting_id,
        )
        save_output(triples, args.output)
