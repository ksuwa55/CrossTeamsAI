import json
import re
import os
import hashlib
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

class MeetingSummarizer:
    def __init__(self, cache_dir="cache"):
        load_dotenv()
        self.client = OpenAI()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs("output", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    def load_and_preprocess_transcript(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        processed = []
        for entry in raw_data:
            text = entry.get("text", "")
            text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]+', '', text)  # emojis
            text = re.sub(r'\b(uh+|um+)\b', '', text, flags=re.IGNORECASE)  # fillers
            text = text.strip()

            processed.append({
                "speaker": entry.get("user") or entry.get("speaker"),
                "timestamp": entry.get("timestamp"),
                "text": text
            })
        return processed

    def build_prompt(self, transcript, mode="general", query=None):
        dialogue = "\n".join(
            f"{msg['speaker']} [{msg['timestamp']}]: {msg['text']}"
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

        return instruction + dialogue

    def get_cache_path(self, prompt, model="gpt-3.5-turbo"):
        key = hashlib.md5((model + prompt).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.json")

    def run_summarizer(self, prompt, model="gpt-3.5-turbo"):
        cache_path = self.get_cache_path(prompt, model)
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)["response"]

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.choices[0].message.content

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "response": output}, f, ensure_ascii=False, indent=2)

        return output

    def parse_output(self, raw_llm_response):
        sections = raw_llm_response.strip().split("Action Items:")
        summary = sections[0].strip()
        action_items = [line.strip("- ").strip() for line in sections[1].split("\n") if line.strip()] if len(sections) > 1 else []
        return {
            "summary": summary,
            "action_items": action_items,
            "decisions": [],
            "blockers": []
        }

    def save_summary(self, summary_dict, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, ensure_ascii=False, indent=2)

    def log_experiment(self, inputs, outputs, metadata):
        log = {
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata
        }
        filename = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
