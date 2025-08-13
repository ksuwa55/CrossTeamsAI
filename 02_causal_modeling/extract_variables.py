import json
import argparse
import os
import re

def classify_speaker_role(speaker):
    speaker = speaker.lower()
    if "pm" in speaker:
        return "pm"
    elif "engineer" in speaker:
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
    return any(keyword in text.lower() for keyword in ["yes", "i will", "we decided", "it’s fixed", "let’s go ahead"])

def detect_blocker(text):
    return any(keyword in text.lower() for keyword in ["blocked", "can’t", "issue", "problem", "delay"])

def detect_next_action(text):
    return any(keyword in text.lower() for keyword in ["please", "need to", "make sure", "start", "schedule", "ask", "prepare"])

def enrich_transcript(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    enriched = []
    for entry in raw:
        text = entry.get("text", "")
        enriched.append({
            "speaker": entry.get("speaker", "unknown"),
            "timestamp": entry.get("timestamp", ""),
            "text": text,
            "speaker_role": classify_speaker_role(entry.get("speaker", "")),
            "topic": detect_topic(text),
            "utterance_type": detect_utterance_type(text),
            "emotion": detect_emotion(text),
            "decision": detect_decision(text),
            "blocker": detect_blocker(text),
            "next_action": detect_next_action(text)
        })
    return enriched

def save_output(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved enriched variables to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input transcript JSON")
    parser.add_argument("--output", required=True, help="Path to save enriched variable output JSON")
    args = parser.parse_args()

    enriched = enrich_transcript(args.input)
    save_output(enriched, args.output)
