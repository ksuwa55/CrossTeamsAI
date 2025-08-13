import os, json, re, glob
from typing import Iterable, Tuple

def _concat_transcript(js) -> str:
    mt = js.get("meeting_transcripts", [])
    parts = []
    for i, seg in enumerate(mt):
        spk = seg.get("speaker") or "UNK"
        txt = seg.get("content") or ""
        txt = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]+', '', txt)  # strip emoji
        txt = re.sub(r'\b(uh+|um+)\b', '', txt, flags=re.IGNORECASE)
        txt = txt.strip()
        if txt:
            parts.append(f"{spk} [{i}]: {txt}")
    return "\n".join(parts)

def _extract_queries(js) -> Iterable[Tuple[str, str, str]]:
    """
    Yields (query_id, query_text, reference_summary).
    Works for general/specific query lists and single-query forms.
    """
    def pull(lst, prefix):
        if isinstance(lst, list):
            for idx, q in enumerate(lst):
                if not isinstance(q, dict):
                    continue
                query = q.get("query") or q.get("question")
                ref = q.get("answer") or q.get("summary") or q.get("reference")
                if isinstance(ref, dict) and "text" in ref:
                    ref = ref["text"]
                if isinstance(ref, list):
                    ref = " ".join(x["text"] if isinstance(x, dict) and "text" in x else str(x) for x in ref)
                if query and ref:
                    qid = str(q.get("query_id") or q.get("id") or f"{prefix}-{idx}")
                    yield (qid, query.strip(), str(ref).strip())

    yield from pull(js.get("general_query_list"), "gen")
    yield from pull(js.get("specific_query_list"), "spec")

    # fallback single-query form
    if js.get("query") and (js.get("answer") or js.get("summary")):
        ref = js.get("answer") or js.get("summary")
        if isinstance(ref, dict) and "text" in ref:
            ref = ref["text"]
        yield ("single", js["query"].strip(), str(ref).strip())

def iter_qmsum(split_dir: str):
    """
    Iterates over ALL files in the split (train/val/test) and all queries in each file.
    Yields dict with: meeting_id, query_id, query, input_text, reference
    """
    files = sorted(glob.glob(os.path.join(split_dir, "*.json")))
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            js = json.load(f)
        meeting_id = js.get("meeting_id") or os.path.splitext(os.path.basename(fp))[0]
        transcript = _concat_transcript(js)
        for qid, query, ref in _extract_queries(js):
            input_text = f"Summarize the meeting in response to this query: '{query}'\n\n{transcript}"
            yield {
                "meeting_id": meeting_id,
                "query_id": qid,
                "query": query,
                "input_text": input_text,
                "reference": ref,
            }
