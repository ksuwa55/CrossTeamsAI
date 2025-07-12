import json
import re
from datetime import datetime
from openai import OpenAI
import hashlib
from dotenv import load_dotenv
import os

os.makedirs("output", exist_ok=True)
os.makedirs("logs", exist_ok=True)
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

load_dotenv()

# ============================
# Input Preprocessing
# ============================
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

# ============================
# Prompt Construction
# ============================
def build_prompt(transcript, mode="general", query=None):
    dialogue = ""
    for msg in transcript:
        speaker = msg.get("user", "Unknown")
        time = msg.get("timestamp", "")
        text = msg.get("text", "").strip()
        dialogue += f"{speaker} [{time}]: {text}\n"

    instruction = "Summarize this meeting. Include key points and action items.\n\n" if mode == "general" else f"Summarize the meeting in response to: \"{query}\"\n\n"
    return instruction + dialogue.strip()

# ============================
# Cache prompt
# ============================
def get_cache_path(prompt, model="gpt-3.5-turbo"):
    key = hashlib.md5((model + prompt).encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")

# ============================
# LLM Inference (Mock)
# ============================
def run_summarizer(prompt, model="gpt-3.5-turbo"):
    cache_path = get_cache_path(prompt, model)

    # Use cache if it exists
    if os.path.exists(cache_path):
        print("[Cache] Using cached response")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["response"]

    # Call OpenAI API if not cached
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    print("[API] Using called OpenAI API")
    output = response.choices[0].message.content

    # Save to cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "response": output}, f, ensure_ascii=False, indent=2)

    return output

# ============================
# Postprocessing Output
# ============================
def parse_output(raw_llm_response):
    sections = raw_llm_response.strip().split("Action Items:")
    summary = sections[0].strip()
    action_items = [line.strip("- ").strip() for line in sections[1].split("\n") if line.strip()] if len(sections) > 1 else []
    return {"summary": summary, "action_items": action_items}

# ============================
# Output Formatting
# ============================
def save_summary(summary_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {output_path}")

# ============================
# Benchmark Logging
# ============================
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
    print(f"Logged experiment to {filename}")

if __name__ == "__main__":
    transcript = load_and_preprocess_transcript("data/zoom_transcript_sample.json")
    prompt = build_prompt(transcript)
    raw = run_summarizer(prompt)
    parsed = parse_output(raw)
    save_summary(parsed, "output/summary.json")
    log_experiment({"transcript": transcript, "prompt": prompt}, parsed, {"model": "gpt-3.5-turbo"})
