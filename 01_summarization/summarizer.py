import json
import re
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv


class MeetingSummarizer:
    """
    Minimal summarizer with:
      - preprocessing helpers
      - prompt builder (general / decision / blocker / query)
      - OpenAI chat completion with optional system prompt + low temperature
      - simple cache to avoid repeated API calls
      - optional revise() pass to polish final output (for better ROUGE-2/L)
    """
    def __init__(self, cache_dir: str = "cache"):
        load_dotenv()
        self.client = OpenAI()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs("output", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

    # ----------------------------
    # Data loading / preprocessing
    # ----------------------------
    def load_and_preprocess_transcript(self, file_path: str) -> List[Dict]:
        """
        Expects a JSON transcript: list of dicts with keys like {user/speaker, timestamp, text}.
        Cleans emoji and filler words ('uh', 'um').
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        processed = []
        for entry in raw_data:
            text = entry.get("text", "") or ""
            # strip emoji symbols
            text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]+', '', text)
            # strip simple fillers
            text = re.sub(r'\b(uh+|um+)\b', '', text, flags=re.IGNORECASE)
            text = text.strip()

            processed.append({
                "speaker": entry.get("user") or entry.get("speaker") or "UNK",
                "timestamp": entry.get("timestamp"),
                "text": text
            })
        return processed

    # ----------------------------
    # Prompt building
    # ----------------------------
    def build_prompt(self, transcript: List[Dict], mode: str = "general", query: Optional[str] = None) -> str:
        dialogue = "\n".join(
            f"{msg.get('speaker','UNK')} [{msg.get('timestamp','')}]: {msg.get('text','')}"
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

    # ----------------------------
    # Caching helpers
    # ----------------------------
    def get_cache_path(self, prompt: str, model: str = "gpt-3.5-turbo", system_prompt: str = "") -> str:
        """
        Include system_prompt in the key so different behaviors don't collide in cache.
        """
        key = hashlib.md5((model + (system_prompt or "") + prompt).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.json")

    # ----------------------------
    # Core model call
    # ----------------------------
    def run_summarizer(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 220,
    ) -> str:
        """
        Chat completion with:
          - optional system prompt (improves style, structure, faithfulness)
          - low temperature to stabilize n-grams (helps ROUGE-2/L)
          - modest max_tokens to reduce drift and cost
        Backward compatible: if system_prompt is None, behavior matches older calls.
        """
        cache_path = self.get_cache_path(prompt, model=model, system_prompt=(system_prompt or ""))
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)["response"]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        output = response.choices[0].message.content

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {"prompt": prompt, "system_prompt": system_prompt, "response": output},
                f,
                ensure_ascii=False,
                indent=2,
            )

        return output

    # ----------------------------
    # Optional polishing step
    # ----------------------------
    def revise_summary(self, draft_text: str, query: str, model: str = "gpt-3.5-turbo") -> str:
        """
        One short edit pass to improve coherence and phrasing of the final answer.
        Use on the final (reduced) output only. Adds ~1 extra API call per query.
        """
        system_prompt = (
            "You are a precise editor for query-focused meeting summaries. "
            "Return only the revised summary, 4–6 sentences, factual, no preamble."
        )
        prompt = (
            f"Query: '{query}'\n\n"
            "Revise the draft to be concise, coherent, and strictly supported by the transcript facts. "
            "Edit wording; do not introduce new information.\n\n"
            f"Draft:\n{draft_text}"
        )
        return self.run_summarizer(
            prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=220,
        ).strip()

    # ----------------------------
    # Output parsing / logging
    # ----------------------------
    def parse_output(self, raw_llm_response: str) -> Dict:
        """
        Backward-compatible parser for the single-transcript flow used by main.py.
        """
        sections = raw_llm_response.strip().split("Action Items:")
        summary = sections[0].strip()
        action_items = (
            [line.strip("- ").strip() for line in sections[1].split("\n") if line.strip()]
            if len(sections) > 1 else []
        )
        return {
            "summary": summary,
            "action_items": action_items,
            "decisions": [],
            "blockers": []
        }

    def save_summary(self, summary_dict: Dict, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, ensure_ascii=False, indent=2)

    def log_experiment(self, inputs: Dict, outputs: Dict, metadata: Dict) -> None:
        log = {
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata
        }
        filename = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
