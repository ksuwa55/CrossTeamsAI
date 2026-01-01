# app/ui_runner.py
import os, sys, re
from typing import List, Optional
import gradio as gr

# --- locate repo root and import code in 01_summarization ---
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PKG_DIR = os.path.join(REPO_ROOT, "01_summarization")
sys.path.insert(0, PKG_DIR)

from summarizer import MeetingSummarizer  # existing class

# ===== Minimal helpers (copied from CLI for consistency) =====
SYSTEM_PROMPT = (
    "You are a precise, query-focused meeting summarizer. "
    "Write concise, factual summaries. "
    "Do not add any information that is not supported by the transcript."
)

TASK_TEMPLATES = {
    "general": {
        "map": ("Task: write a concise but complete summary focused on the query using ONLY facts in the transcript slice. "
                "Cover: objectives, main points, outcomes, next steps. Avoid speculation. No preamble."),
        "reduce": ("Task: merge partial answers into one coherent summary. Remove redundancy, keep all key facts, "
                   "preserve wording when correct. Cover: objectives, main points, outcomes, next steps. No preamble."),
    },
    "decisions": {
        "map": ("Task: extract DECISIONS relevant to the query. For each: WHAT, WHO decided, WHEN(if stated), "
                "RATIONALE(if stated). Only facts in slice. No preamble."),
        "reduce": ("Task: merge DECISIONS. For each: WHAT, WHO, WHEN(if stated), RATIONALE(if stated). "
                   "Deduplicate; keep supported facts. No preamble."),
    },
    "blockers": {
        "map": ("Task: extract BLOCKERS. For each: WHAT is blocked, ROOT CAUSE(if stated), OWNER, "
                "UNBLOCKING ACTIONS(if stated). Only facts in slice. No preamble."),
        "reduce": ("Task: merge BLOCKERS. For each: WHAT, ROOT CAUSE(if stated), OWNER, UNBLOCKING ACTIONS(if stated), "
                   "STATUS. Deduplicate; keep supported facts. No preamble."),
    },
    "action_items": {
        "map": ("Task: extract ACTION ITEMS. For each: WHO, WHAT, WHEN/DUE(if stated). Only facts in slice. No preamble."),
        "reduce": ("Task: merge ACTION ITEMS. For each: WHO, WHAT, WHEN/DUE(if stated). Deduplicate; keep atomic/factual. No preamble."),
    },
    "risks": {
        "map": ("Task: extract RISKS. For each: DESCRIPTION, LIKELIHOOD(if stated), IMPACT(if stated), "
                "MITIGATION(if stated). Only facts in slice. No preamble."),
        "reduce": ("Task: merge RISKS. For each: DESCRIPTION, LIKELIHOOD(if stated), IMPACT(if stated), "
                   "MITIGATION(if stated). Deduplicate; keep supported facts. No preamble."),
    },
    "followups": {
        "map": ("Task: extract FOLLOW-UPS/QUESTIONS. For each: WHO asked/owns, WHAT is requested, "
                "DUE(if stated). Only facts in slice. No preamble."),
        "reduce": ("Task: merge FOLLOW-UPS/QUESTIONS. For each: WHO asked/owns, WHAT is requested, "
                   "DUE(if stated), CURRENT STATUS(if stated). Remove duplicates; keep supported facts. No preamble."),
    },
}

def split_chunks(text: str, chunk_chars: int = 9000, overlap_chars: int = 800) -> List[str]:
    if chunk_chars <= 0:
        return [text]
    out = []
    n = len(text); start = 0
    while start < n:
        end = min(start + chunk_chars, n)
        out.append(text[start:end])
        if end == n: break
        start = max(0, end - overlap_chars)
    return out

def build_query_prompt(query: str, transcript_slice: str, task: str = "general") -> str:
    ins = TASK_TEMPLATES.get(task, TASK_TEMPLATES["general"])["map"]
    return (
        ins + "\n\n"
        f"Query: '{(query or '').strip()}'\n\n"
        f"Transcript slice:\n{transcript_slice}"
    )

def build_reduce_prompt(query: str, partials: List[str], task: str = "general") -> str:
    ins = TASK_TEMPLATES.get(task, TASK_TEMPLATES["general"])["reduce"]
    joined = "\n\n--- PART ---\n\n".join(partials)
    return (
        ins + "\n\n"
        f"Query: '{(query or '').strip()}'\n\n"
        f"Partial answers:\n{joined}"
    )

# ===== UI callback =====
ms = MeetingSummarizer(cache_dir="cache_ui")

def ui_summarize(transcript: str, query: str, task: str,
                 chunk_chars: int, overlap_chars: int,
                 temperature: float, max_tokens: int):
    transcript = (transcript or "").strip()
    if not transcript:
        return "Please paste a transcript.", {}

    chunks = split_chunks(transcript, chunk_chars=chunk_chars, overlap_chars=overlap_chars)

    partials = []
    for ch in chunks:
        mp = build_query_prompt(query, ch, task=task)
        part = ms.run_summarizer(
            mp, model="gpt-3.5-turbo", system_prompt=SYSTEM_PROMPT,
            temperature=temperature, max_tokens=max_tokens
        ).strip()
        partials.append(part)

    if len(partials) == 1:
        final = partials[0]
    else:
        rp = build_reduce_prompt(query, partials, task=task)
        final = ms.run_summarizer(
            rp, model="gpt-3.5-turbo", system_prompt=SYSTEM_PROMPT,
            temperature=temperature, max_tokens=max_tokens
        ).strip()

    # minimal structured parse (very light)
    structured = {
        "task": task,
        "query": query or None,
        "chunks": len(chunks),
        "summary": final
    }
    return final, structured

# ===== Gradio app =====
with gr.Blocks(title="Meeting Summarizer") as demo:
    gr.Markdown("## Meeting Summarizer — Simple Demo")

    with gr.Row():
        transcript = gr.Textbox(label="Transcript", lines=16, placeholder="Paste meeting transcript here...")
        with gr.Column():
            query = gr.Textbox(label="Query (optional)")
            task = gr.Dropdown(
                ["general", "decisions", "blockers", "action_items", "risks", "followups"],
                value="general", label="Discussion Type"
            )
            chunk_chars = gr.Slider(3000, 16000, value=9000, step=500, label="Chunk size (chars)")
            overlap_chars = gr.Slider(0, 2000, value=800, step=100, label="Overlap (chars)")
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
            max_tokens = gr.Slider(64, 512, value=320, step=16, label="Max tokens per call")
            run_btn = gr.Button("Generate Summary", variant="primary")

    out_text = gr.Textbox(label="Summary", lines=12)
    out_json = gr.JSON(label="Structured (minimal)")

    run_btn.click(
        ui_summarize,
        inputs=[transcript, query, task, chunk_chars, overlap_chars, temperature, max_tokens],
        outputs=[out_text, out_json]
    )

if __name__ == "__main__":
    # share=False keeps it local; server_port must match Docker port mapping
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
