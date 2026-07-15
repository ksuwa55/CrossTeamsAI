import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
SUMMARIZER_PKG = os.path.join(REPO_ROOT, "01_summarization")
CAUSAL_PKG = os.path.join(REPO_ROOT, "02_causal_modeling")
KG_PKG = os.path.join(REPO_ROOT, "04_knowledgegraph_dashboard")
sys.path.insert(0, SUMMARIZER_PKG)
sys.path.insert(0, CAUSAL_PKG)
sys.path.insert(0, KG_PKG)
sys.path.insert(0, HERE)

from summarizer import MeetingSummarizer  # noqa: E402
from extract_variables import enrich_transcript, extract_causal_events  # noqa: E402
from extract_entities_relations import extract_kg_triples  # noqa: E402
from cross_link import link_causal_and_kg  # noqa: E402


def iter_transcript_paths(transcripts_dir: str):
    for path in sorted(glob.glob(os.path.join(transcripts_dir, "*.json"))):
        if path.endswith(".labels.json") or path.endswith(".kg_labels.json"):
            continue
        yield path


def _load_precomputed(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def run_pipeline_for_transcript(
    transcript_path: str,
    model: str = "gpt-3.5-turbo",
    skip_llm: bool = True,
    causal_events_dir: str = None,
    kg_triples_dir: str = None,
) -> dict:
    """Run Phase 1 (summary), Phase 2 (causal events), and Phase 4 (KG
    triples) for one transcript and cross-link the Phase 2/4 outputs.

    skip_llm=True (default): reuses already-extracted output/causal_events
    and output/kg_triples for this meeting if present, and skips the
    summarizer entirely, so this runs with zero API calls. skip_llm=False
    extracts fresh (each phase's own disk cache still avoids re-paying for
    already-seen prompts).
    """
    meeting_id = os.path.splitext(os.path.basename(transcript_path))[0]

    causal_events = None
    if causal_events_dir:
        causal_events = _load_precomputed(
            os.path.join(causal_events_dir, f"{meeting_id}.causal_events.json")
        )
    kg_triples = None
    if kg_triples_dir:
        kg_triples = _load_precomputed(
            os.path.join(kg_triples_dir, f"{meeting_id}.kg_triples.json")
        )

    summary = None
    enriched = None
    if not skip_llm:
        enriched = enrich_transcript(transcript_path)
        if causal_events is None:
            causal_events = extract_causal_events(enriched, model=model, meeting_id=meeting_id)
        if kg_triples is None:
            kg_triples = extract_kg_triples(enriched, model=model, meeting_id=meeting_id)

        summarizer = MeetingSummarizer(cache_dir="cache")
        transcript = summarizer.load_and_preprocess_transcript(transcript_path)
        prompt = summarizer.build_prompt(transcript, mode="general")
        summary = summarizer.run_summarizer(prompt, model=model).strip()

    causal_events = causal_events or []
    kg_triples = kg_triples or []
    cross_links = link_causal_and_kg(causal_events, kg_triples)

    return {
        "meeting_id": meeting_id,
        "summary": summary,
        "causal_events": causal_events,
        "kg_triples": kg_triples,
        "cross_links": cross_links,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Phases 1/2/4 over a transcript directory and cross-link the results.")
    parser.add_argument("--transcripts-dir", default=os.path.join(REPO_ROOT, "data", "synthetic_transcripts"))
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "output", "integrated"))
    parser.add_argument("--causal-events-dir", default=os.path.join(REPO_ROOT, "output", "causal_events"),
                         help="Existing per-meeting causal_events to reuse instead of re-extracting.")
    parser.add_argument("--kg-triples-dir", default=os.path.join(REPO_ROOT, "output", "kg_triples"),
                         help="Existing per-meeting kg_triples to reuse instead of re-extracting.")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--skip-llm", action="store_true", default=True,
                         help="Reuse existing extraction outputs and skip all LLM calls (default).")
    parser.add_argument("--with-llm", dest="skip_llm", action="store_false",
                         help="Extract fresh (summary + causal events + KG triples) instead of reusing existing outputs.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for transcript_path in iter_transcript_paths(args.transcripts_dir):
        result = run_pipeline_for_transcript(
            transcript_path,
            model=args.model,
            skip_llm=args.skip_llm,
            causal_events_dir=args.causal_events_dir,
            kg_triples_dir=args.kg_triples_dir,
        )
        out_path = os.path.join(args.out_dir, f"{result['meeting_id']}.integrated.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(
            f"{result['meeting_id']}: {len(result['causal_events'])} causal event(s), "
            f"{len(result['kg_triples'])} triple(s), {len(result['cross_links'])} cross-link(s) "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
