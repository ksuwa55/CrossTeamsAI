"""
Baseline methods for QMSum query-focused meeting summarization.

Produces JSONL output in the same format as cli_runner.py for direct
comparison via evaluate_qmsum.py.

Baselines implemented:
  1. lead-n      — first N sentences of the transcript
  2. tfidf       — TF-IDF cosine extractive: pick sentences closest to the query
  3. random      — random N sentences (lower-bound sanity check)
"""

import os, sys, json, re, argparse, random, math
from collections import Counter
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PKG_DIR = os.path.join(REPO_ROOT, "01_summarization")
sys.path.insert(0, PKG_DIR)

from qmsum_loader import iter_qmsum

# ─── helpers ───────────────────────────────────────────────────────

def _extract_transcript(input_text: str) -> str:
    """Same logic as cli_runner.extract_transcript_from_input_text."""
    t_prefix = "Transcript:\n"
    idx = input_text.find(t_prefix)
    if idx == -1:
        sep = "\n\n"
        j = input_text.find(sep)
        return input_text if j == -1 else input_text[j + len(sep):]
    return input_text[idx + len(t_prefix):]


def _sent_tokenize(text: str) -> List[str]:
    """Simple sentence splitter (avoids nltk dependency for baselines)."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


_STOP = set("the a an is are was were be been being have has had do does did "
            "will would shall should may might can could of in on at to for "
            "with by from and or but not no nor so yet".split())


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower()) if t not in _STOP and len(t) > 1]


def _tfidf_vectors(docs: List[List[str]]):
    """Returns list of {term: tfidf} dicts, one per doc."""
    n = len(docs)
    df = Counter()
    for d in docs:
        df.update(set(d))
    vecs = []
    for d in docs:
        tf = Counter(d)
        total = len(d) or 1
        vec = {}
        for t, c in tf.items():
            idf = math.log((n + 1) / (df[t] + 1)) + 1
            vec[t] = (c / total) * idf
        vecs.append(vec)
    return vecs


def _cosine(a: dict, b: dict) -> float:
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ─── baselines ─────────────────────────────────────────────────────

def lead_n(transcript: str, n_sents: int = 5, **_) -> str:
    """Return first n_sents sentences of the transcript."""
    lines = [l.strip() for l in transcript.splitlines() if l.strip()]
    # Each speaker turn is a "sentence" in meeting transcripts
    return " ".join(lines[:n_sents])


def tfidf_extractive(transcript: str, query: str, n_sents: int = 5, **_) -> str:
    """Select the n_sents transcript lines most similar to the query (TF-IDF cosine)."""
    lines = [l.strip() for l in transcript.splitlines() if l.strip()]
    if not lines:
        return ""
    docs = [_tokenize(l) for l in lines]
    q_tokens = _tokenize(query)
    all_docs = docs + [q_tokens]
    vecs = _tfidf_vectors(all_docs)
    q_vec = vecs[-1]
    scored = [(i, _cosine(vecs[i], q_vec)) for i in range(len(lines))]
    scored.sort(key=lambda x: x[1], reverse=True)
    # Return in original order for coherence
    top_indices = sorted([i for i, _ in scored[:n_sents]])
    return " ".join(lines[i] for i in top_indices)


def random_extractive(transcript: str, n_sents: int = 5, seed: int = 42, **_) -> str:
    """Random n_sents lines (lower bound)."""
    lines = [l.strip() for l in transcript.splitlines() if l.strip()]
    if len(lines) <= n_sents:
        return " ".join(lines)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(lines)), n_sents))
    return " ".join(lines[i] for i in chosen)


BASELINES = {
    "lead-n": lead_n,
    "tfidf": tfidf_extractive,
    "random": random_extractive,
}

# ─── main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run extractive baselines on QMSum")
    ap.add_argument("--qmsum_split_dir", required=True)
    ap.add_argument("--method", choices=list(BASELINES.keys()), default="tfidf")
    ap.add_argument("--n_sents", type=int, default=5, help="Number of sentences/lines to extract")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    fn = BASELINES[args.method]

    written = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for ex in iter_qmsum(args.qmsum_split_dir):
            transcript = _extract_transcript(ex["input_text"])
            pred = fn(transcript, query=ex["query"], n_sents=args.n_sents, seed=args.seed)
            row = {
                "meeting_id": ex["meeting_id"],
                "query_id": ex["query_id"],
                "query": ex["query"],
                "prediction": pred,
                "reference": ex["reference"],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"[{args.method}] Wrote {written} predictions to {args.out_jsonl}")


if __name__ == "__main__":
    main()
