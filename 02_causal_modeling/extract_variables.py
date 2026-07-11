import json
import argparse
import os
import re
import sys
from typing import List, Dict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
SUMMARIZER_PKG = os.path.join(REPO_ROOT, "01_summarization")
sys.path.insert(0, SUMMARIZER_PKG)

from summarizer import MeetingSummarizer


# ----------------------------
# Cheap regex prefilter (unchanged logic, used to pick candidate utterances
# before spending an LLM call on them)
# ----------------------------
def classify_speaker_role(speaker):
    speaker = speaker.lower()
    if "pm" in speaker:
        return "pm"
    elif "engineer" in speaker or "dev" in speaker:
        return "engineer"
    elif "host" in speaker:
        return "facilitator"
    else:
        return "unknown"

def detect_topic(text):
    text = text.lower()
    if "login" in text or "bug" in text:
        return "bug fix update"
    elif "qa" in text or "testing" in text:
        return "testing schedule"
    elif "release notes" in text:
        return "release preparation"
    elif "welcome" in text:
        return "meeting kickoff"
    else:
        return "general"

def detect_utterance_type(text):
    if re.search(r"\b(what|can|do|should|could|will)\b", text.lower()):
        return "question"
    elif re.search(r"\b(please|start|make sure|prepare)\b", text.lower()):
        return "instruction"
    elif re.search(r"\b(fixed|done|completed|progress|working on)\b", text.lower()):
        return "status report"
    elif re.search(r"\b(yes|i will|i’ll|sure|okay)\b", text.lower()):
        return "commitment"
    elif re.search(r"\b(hello|hi|welcome)\b", text.lower()):
        return "greeting"
    else:
        return "statement"

def detect_emotion(text):
    if re.search(r"\b(great|awesome|amazing|nice)\b", text.lower()):
        return "positive"
    elif re.search(r"\b(sorry|problem|issue|blocked)\b", text.lower()):
        return "negative"
    else:
        return "neutral"

def detect_decision(text):
    return any(keyword in text.lower() for keyword in ["yes", "i will", "we decided", "it’s fixed", "let’s go ahead", "let's push", "push the launch"])

def detect_blocker(text):
    return any(keyword in text.lower() for keyword in ["blocked", "blocker", "can’t", "can't", "issue", "problem", "delay"])

def detect_next_action(text):
    return any(keyword in text.lower() for keyword in ["please", "need to", "make sure", "start", "schedule", "ask", "prepare", "update the jira"])

def enrich_transcript(input_path: str) -> List[Dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    enriched = []
    for entry in raw:
        text = entry.get("text", "") or ""
        speaker = entry.get("speaker") or entry.get("user") or "unknown"
        enriched.append({
            "speaker": speaker,
            "timestamp": entry.get("timestamp", ""),
            "text": text,
            "speaker_role": classify_speaker_role(speaker),
            "topic": detect_topic(text),
            "utterance_type": detect_utterance_type(text),
            "emotion": detect_emotion(text),
            "decision": detect_decision(text),
            "blocker": detect_blocker(text),
            "next_action": detect_next_action(text)
        })
    return enriched


# ----------------------------
# LLM causal event extraction (Cause -> Effect -> Timestamp)
# ----------------------------
CAUSAL_SYSTEM_PROMPT = (
    "You are analyzing a team meeting or chat transcript for a software project. "
    "Your job is to find causal relationships that explain project bottlenecks: a Cause "
    "(e.g. an ambiguous requirement, a missing resource, a blocked dependency) that leads "
    "to an Effect (e.g. an integration delay, a missed deadline, stalled decision-making). "
    "Only report a relationship if the transcript excerpt gives clear evidence for it — do "
    "not invent causes or effects that aren't grounded in the text. "
    "Respond with ONLY a JSON array, no prose, no markdown fences."
)

def build_causal_prompt(window: List[Dict]) -> str:
    dialogue = "\n".join(
        f"{u['speaker']} [{u['timestamp']}]: {u['text']}" for u in window
    )
    return (
        "Transcript excerpt:\n"
        f"{dialogue}\n\n"
        "If this excerpt shows one or more cause-effect relationships, respond with a JSON "
        "array where each item has exactly these keys:\n"
        '  "cause": short phrase describing the cause\n'
        '  "effect": short phrase describing the effect\n'
        '  "timestamp": the timestamp of the utterance where the effect becomes visible\n'
        "If there is no clear causal relationship in this excerpt, respond with: []"
    )

def parse_causal_response(raw: str) -> List[Dict]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    events = []
    for item in data:
        if not isinstance(item, dict):
            continue
        cause = item.get("cause")
        effect = item.get("effect")
        if cause and effect:
            events.append({
                "cause": str(cause).strip(),
                "effect": str(effect).strip(),
                "timestamp": str(item.get("timestamp", "")).strip(),
            })
    return events

def select_candidate_windows(enriched: List[Dict], window_before: int = 2, window_after: int = 1) -> List[List[Dict]]:
    candidate_idxs = [i for i, u in enumerate(enriched) if u["blocker"] or u["decision"] or u["next_action"]]
    windows = []
    seen_ranges = set()
    for i in candidate_idxs:
        start = max(0, i - window_before)
        end = min(len(enriched), i + window_after + 1)
        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))
        windows.append(enriched[start:end])
    return windows

def extract_causal_events(
    enriched: List[Dict],
    model: str = "gpt-3.5-turbo",
    window_before: int = 2,
    window_after: int = 1,
    cache_dir: str = "cache_causal",
    meeting_id: str = None,
) -> List[Dict]:
    summarizer = MeetingSummarizer(cache_dir=cache_dir)
    windows = select_candidate_windows(enriched, window_before, window_after)

    all_events = []
    for window in windows:
        prompt = build_causal_prompt(window)
        raw = summarizer.run_summarizer(
            prompt,
            model=model,
            system_prompt=CAUSAL_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=300,
        )
        all_events.extend(parse_causal_response(raw))

    unique_events = []
    seen = set()
    for e in all_events:
        key = (e["cause"].lower(), e["effect"].lower())
        if key in seen:
            continue
        seen.add(key)
        if meeting_id:
            e["meeting_id"] = meeting_id
        unique_events.append(e)
    return unique_events


def save_output(data, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved output to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input transcript JSON")
    parser.add_argument("--output", required=True, help="Path to save causal events JSON")
    parser.add_argument("--enriched-output", default=None, help="Optional path to also save the per-utterance prefilter variables")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--window-before", type=int, default=2)
    parser.add_argument("--window-after", type=int, default=1)
    parser.add_argument("--skip-llm", action="store_true", help="Only run the regex prefilter; skip the LLM causal extraction call")
    args = parser.parse_args()

    enriched = enrich_transcript(args.input)
    if args.enriched_output:
        save_output(enriched, args.enriched_output)

    if args.skip_llm:
        save_output(enriched, args.output)
    else:
        meeting_id = os.path.splitext(os.path.basename(args.input))[0]
        events = extract_causal_events(
            enriched,
            model=args.model,
            window_before=args.window_before,
            window_after=args.window_after,
            meeting_id=meeting_id,
        )
        save_output(events, args.output)
