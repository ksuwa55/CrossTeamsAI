import os
import sys

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
INTEGRATION_PKG = os.path.join(REPO_ROOT, "05_integration")
SUMMARIZER_PKG = os.path.join(REPO_ROOT, "01_summarization")
sys.path.insert(0, HERE)
sys.path.insert(0, INTEGRATION_PKG)
sys.path.insert(0, SUMMARIZER_PKG)

# Each of these builds its Gradio Blocks object at import time but only
# calls .launch() inside `if __name__ == "__main__"`, so importing them here
# is safe and doesn't start extra servers.
import ui_runner  # noqa: E402
import intervention_dashboard  # noqa: E402
import kg_dashboard  # noqa: E402

from cross_link import link_all_meetings  # noqa: E402
from summarizer import MeetingSummarizer  # noqa: E402

CAUSAL_EVENTS_GLOB = os.path.join(REPO_ROOT, "output", "causal_events", "*.causal_events.json")
KG_TRIPLES_GLOB = os.path.join(REPO_ROOT, "output", "kg_triples", "*.kg_triples.json")
TRANSCRIPTS_DIR = os.path.join(REPO_ROOT, "data", "synthetic_transcripts")

_links_by_meeting = link_all_meetings(CAUSAL_EVENTS_GLOB, KG_TRIPLES_GLOB)
_meeting_ids = sorted(_links_by_meeting.keys())
_ms = MeetingSummarizer(cache_dir="cache")


def render_cross_links(meeting_id: str):
    links = _links_by_meeting.get(meeting_id, [])
    if not links:
        return "No cross-links found for this meeting."
    lines = [f"**{len(links)} cross-link(s) for `{meeting_id}`** (causal event <-> KG triple, matched by shared wording/timestamp)\n"]
    for link in links:
        ce = link["causal_event"]
        kg = link["kg_triple"]
        lines.append(
            f"- **score {link['score']}** (timestamp match: {link['timestamp_match']})\n"
            f"  - Causal: _{ce['cause']}_ &rarr; _{ce['effect']}_\n"
            f"  - KG: **{kg['subject_text']}** --{kg['relation']}--> **{kg['object_text']}**\n"
            f"    > \"{kg.get('quote', '')}\""
        )
    return "\n".join(lines)


def generate_live_summary(meeting_id: str):
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{meeting_id}.json")
    if not os.path.exists(transcript_path):
        return f"No transcript found for {meeting_id}."
    try:
        transcript = _ms.load_and_preprocess_transcript(transcript_path)
        prompt = _ms.build_prompt(transcript, mode="general")
        return _ms.run_summarizer(prompt, model="gpt-3.5-turbo").strip()
    except Exception as exc:
        return (
            f"Live summary generation failed ({exc}). This requires a live OpenAI API "
            "key/credits — the cross-links above only need output/causal_events/ and "
            "output/kg_triples/ to already exist locally, no live API call."
        )


with gr.Blocks(title="Integrated View") as integrated_demo:
    gr.Markdown("## Integrated View — Causal Events &harr; Knowledge Graph")
    gr.Markdown(
        "Cross-links between Phase 2's causal events (`output/causal_events/`) and "
        "Phase 4's KG triples (`output/kg_triples/`), both extracted from the same "
        "meeting transcripts. Linking is by shared wording + timestamp "
        "(`05_integration/cross_link.py`) — a heuristic, not a learned entity linker."
    )
    if not _meeting_ids:
        gr.Markdown(
            "**No data found.** `output/causal_events/` and `output/kg_triples/` are "
            "gitignored (regenerable, not source) — run `python eval/evaluate_causal_extraction.py` "
            "and `python eval/evaluate_kg_extraction.py` first (see root README) to populate them."
        )
    meeting_dropdown = gr.Dropdown(_meeting_ids, value=_meeting_ids[0] if _meeting_ids else None, label="Meeting")
    links_md = gr.Markdown()
    meeting_dropdown.change(render_cross_links, inputs=[meeting_dropdown], outputs=[links_md])
    integrated_demo.load(render_cross_links, inputs=[meeting_dropdown], outputs=[links_md])

    gr.Markdown("---\n### Optional: generate a live Phase 1 summary for this meeting")
    summary_btn = gr.Button("Generate live summary")
    summary_out = gr.Textbox(label="Summary", lines=6)
    summary_btn.click(generate_live_summary, inputs=[meeting_dropdown], outputs=[summary_out])


demo = gr.TabbedInterface(
    [ui_runner.demo, intervention_dashboard.demo, kg_dashboard.demo, integrated_demo],
    tab_names=["Summarizer", "Causal What-If", "Knowledge Graph Explorer", "Integrated View"],
    title="CrossTeamsAI — Unified Dashboard",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
