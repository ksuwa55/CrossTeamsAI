import json
import re
from datetime import datetime
from openai import OpenAI
import hashlib
from dotenv import load_dotenv
import os
import argparse

# Setup Directories and Environment
os.makedirs("output", exist_ok=True)
os.makedirs("logs", exist_ok=True)
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
load_dotenv()

# Preprocess Transcript
def load_and_preprocess_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed = []
    for entry in raw_data:
        text = entry.get("text", "")
        text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]+', '', text)  # remove emojis
        text = re.sub(r'\b(uh+|um+)\b', '', text, flags=re.IGNORECASE)  # remove fillers
        text = text.strip()

        processed.append({
            "speaker": entry.get("user") or entry.get("speaker"),
            "timestamp": entry.get("timestamp"),
            "text": text
        })
    return processed

# Prompt Construction with Type Switching
# Modes: general / decision / blocker / query
def build_prompt(transcript, mode="general", query=None):
    dialogue = "\n".join(
        f"{msg.get('speaker')} [{msg.get('timestamp')}]: {msg.get('text').strip()}"
        for msg in transcript
    )

    if mode == "decision":
        instruction = "Summarize key decisions made in the meeting.\n\n"
    elif mode == "blocker":
        instruction = "Summarize any issues or blockers raised by participants.\n\n"
    elif mode == "query" and query:
        instruction = f"Summarize the meeting in response to this query: '{query}'\n\n"
    else:
        instruction = "Summarize this meeting. Include key points and action items.\n\n"

    return instruction + dialogue.strip()

# Caching
def get_cache_path(prompt, model="gpt-3.5-turbo"):
    key = hashlib.md5((model + prompt).encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")

# Run LLM Summarizer
def run_summarizer(prompt, model="gpt-3.5-turbo"):
    cache_path = get_cache_path(prompt, model)

    if os.path.exists(cache_path):
        print("[Cache] Using cached response")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["response"]

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    print("[API] Using called OpenAI API")
    output = response.choices[0].message.content

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "response": output}, f, ensure_ascii=False, indent=2)

    return output

# Recursive Summary
def recursive_summarize(chunks):
    summaries = [run_summarizer(chunk) for chunk in chunks]
    return run_summarizer("\n".join(summaries))

# Postprocess LLM Output
def parse_output(raw_llm_response):
    sections = raw_llm_response.strip().split("Action Items:")
    summary = sections[0].strip()
    action_items = [line.strip("- ").strip() for line in sections[1].split("\n") if line.strip()] if len(sections) > 1 else []

    return {
        "summary": summary,
        "action_items": action_items,
        "decisions": [],
        "blockers": []
    }

# Save Summary to File
def save_summary(summary_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {output_path}")

# Log Run Metadata
def log_experiment(inputs, outputs, metadata):
    log = {
        "timestamp": datetime.now().isoformat(),
        "inputs": inputs,
        "outputs": outputs,
        "metadata": metadata
    }
    filename = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"📝 Logged experiment to {filename}")

# Run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["general", "decision", "blocker", "query"], default="general")
    parser.add_argument("--query", type=str, help="Optional user query for query-based summarization")
    args = parser.parse_args()

    transcript = load_and_preprocess_transcript("data/zoom_transcript_sample.json")
    prompt = build_prompt(transcript, mode=args.mode, query=args.query)
    raw = run_summarizer(prompt)
    parsed = parse_output(raw)
    save_summary(parsed, "output/summary.json")
    log_experiment({"transcript": transcript, "prompt": prompt}, parsed, {"model": "gpt-3.5-turbo"})
