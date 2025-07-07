import json
import re
# ============================
# 1. Input Preprocessing
# ============================

# - Load input transcript (Slack/Zoom) from file
# - Parse speakers, timestamps, messages
# - Optionally: Normalize text (remove emojis, filler, noise)
def load_and_preprocess_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)  # ✅ Parse JSON into Python list/dict

    processed = []
    for entry in raw_data:
        text = entry.get("text", "")

        # remove emojis
        text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]+', '', text)  
        # remove fillers
        text = re.sub(r'\b(uh+|um+)\b', '', text, flags=re.IGNORECASE)  
        text = text.strip()

        processed.append({
            "speaker": entry.get("user") or entry.get("speaker"),
            "timestamp": entry.get("timestamp"),
            "text": text
        })
    return processed


# ============================
# 2. Prompt Construction
# ============================

# - Create general summarization prompt template
# - Create query-based summarization prompt (if query provided)
# - Inject preprocessed transcript into prompt

def build_prompt(transcript, mode="general", query=None):
    pass


# ============================
# 3. LLM Inference
# ============================

# - Call OpenAI or other LLM API with prompt
# - Handle token limits via chunking if needed
# - Parse response (text summary)

def run_summarizer(prompt):
    pass


# ============================
# 4. Postprocessing Output
# ============================

# - Parse LLM output into:
#   - summary (str)
#   - action_items (list)
#   - optional metadata (e.g., detected decisions, timestamps)
# - Clean up bullet point formatting

def parse_output(raw_llm_response):
    pass


# ============================
# 5. Output Formatting
# ============================

# - Save final structured summary to Markdown or JSON
# - Optional: simulate Slack message format or print to console

def save_summary(summary_dict, output_path):
    pass


# ============================
# 6. Benchmark Logging
# ============================

# - Store input, prompt, and LLM response for later comparison
# - Save to local file or Notion/Sheets API

def log_experiment(inputs, outputs, metadata):
    pass

if __name__ == "__main__":
    load_and_preprocess_transcript("data/zoom_transcript_sample.json")