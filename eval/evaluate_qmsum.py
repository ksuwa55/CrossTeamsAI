"""
Comprehensive evaluation for QMSum predictions.

Metrics:
  - ROUGE-1/2/L/Lsum  (lexical overlap)
  - BERTScore F1       (semantic similarity)

Analysis:
  - Aggregate scores with bootstrap 95% confidence intervals
  - Per-query scores saved to CSV
  - Length statistics (pred vs ref)
  - General vs specific query breakdown
  - Comprehensive results JSON
"""

import json, argparse, os, csv, random
from datetime import datetime
from typing import List, Dict, Tuple

import evaluate


# ─── I/O ───────────────────────────────────────────────────────────

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ─── Bootstrap CI ──────────────────────────────────────────────────

def bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """Returns (mean, lower, upper) for the given confidence level."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return sum(values) / n, lo, hi


# ─── Metrics ───────────────────────────────────────────────────────

def compute_rouge_per_instance(preds: List[str], refs: List[str]) -> List[Dict[str, float]]:
    """Compute ROUGE per instance (one at a time for per-query breakdown)."""
    rouge = evaluate.load("rouge")
    per_instance = []
    for p, r in zip(preds, refs):
        scores = rouge.compute(predictions=[p], references=[r], use_stemmer=True)
        per_instance.append({
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"],
            "rougeLsum": scores.get("rougeLsum", scores["rougeL"]),
        })
    return per_instance


def compute_bertscore(preds: List[str], refs: List[str]) -> List[float]:
    """BERTScore F1 per instance. Returns list of F1 scores."""
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=preds, references=refs, lang="en", model_type="microsoft/deberta-xlarge-mnli")
    return results["f1"]


# ─── Length analysis ───────────────────────────────────────────────

def length_stats(preds: List[str], refs: List[str]) -> Dict:
    pred_lens = [len(p.split()) for p in preds]
    ref_lens = [len(r.split()) for r in refs]
    pred_char_lens = [len(p) for p in preds]
    ref_char_lens = [len(r) for r in refs]
    return {
        "pred_words_mean": sum(pred_lens) / len(pred_lens),
        "ref_words_mean": sum(ref_lens) / len(ref_lens),
        "pred_chars_mean": sum(pred_char_lens) / len(pred_char_lens),
        "ref_chars_mean": sum(ref_char_lens) / len(ref_char_lens),
        "length_ratio_words": (sum(pred_lens) / len(pred_lens)) / max(1, sum(ref_lens) / len(ref_lens)),
        "length_ratio_chars": (sum(pred_char_lens) / len(pred_char_lens)) / max(1, sum(ref_char_lens) / len(ref_char_lens)),
    }


# ─── Main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Comprehensive QMSum evaluation")
    ap.add_argument("--preds_jsonl", required=True)
    ap.add_argument("--out_dir", default="eval/results", help="Directory for result files")
    ap.add_argument("--tag", default="", help="Optional tag for result filenames")
    ap.add_argument("--skip_bertscore", action="store_true", help="Skip BERTScore (slow)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_jsonl(args.preds_jsonl)
    preds = [r["prediction"] for r in rows]
    refs = [r["reference"] for r in rows]

    print(f"Evaluating {len(rows)} instances from {args.preds_jsonl}")

    # ── 1. Per-instance ROUGE ──
    print("Computing per-instance ROUGE...")
    per_rouge = compute_rouge_per_instance(preds, refs)

    # ── 2. BERTScore ──
    bert_f1s = []
    if not args.skip_bertscore:
        print("Computing BERTScore (this may take a minute)...")
        bert_f1s = compute_bertscore(preds, refs)
    else:
        print("Skipping BERTScore.")

    # ── 3. Aggregate with bootstrap CI ──
    metrics_agg = {}
    for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        vals = [d[key] for d in per_rouge]
        mean, lo, hi = bootstrap_ci(vals)
        metrics_agg[key] = {"mean": round(mean, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)}

    if bert_f1s:
        mean, lo, hi = bootstrap_ci(bert_f1s)
        metrics_agg["bertscore_f1"] = {"mean": round(mean, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)}

    # ── 4. Length stats ──
    lens = length_stats(preds, refs)

    # ── 5. General vs specific breakdown ──
    gen_idx = [i for i, r in enumerate(rows) if r.get("query_id", "").startswith("gen")]
    spec_idx = [i for i, r in enumerate(rows) if r.get("query_id", "").startswith("spec")]

    breakdown = {}
    for label, indices in [("general", gen_idx), ("specific", spec_idx)]:
        if not indices:
            continue
        sub = {}
        for key in ["rouge1", "rouge2", "rougeL"]:
            vals = [per_rouge[i][key] for i in indices]
            m, lo, hi = bootstrap_ci(vals)
            sub[key] = round(m, 4)
        if bert_f1s:
            vals = [bert_f1s[i] for i in indices]
            m, _, _ = bootstrap_ci(vals)
            sub["bertscore_f1"] = round(m, 4)
        sub["count"] = len(indices)
        breakdown[label] = sub

    # ── 6. Print summary ──
    print("\n" + "=" * 50)
    print(f"{'Metric':<16} {'Mean':>8}  {'95% CI':>16}")
    print("-" * 50)
    for key, v in metrics_agg.items():
        print(f"{key:<16} {v['mean']:>8.4f}  [{v['ci95_lo']:.4f}, {v['ci95_hi']:.4f}]")
    print("-" * 50)
    print(f"{'pred_words_avg':<16} {lens['pred_words_mean']:>8.1f}")
    print(f"{'ref_words_avg':<16} {lens['ref_words_mean']:>8.1f}")
    print(f"{'length_ratio':<16} {lens['length_ratio_words']:>8.2f}x")
    print(f"{'instances':<16} {len(rows):>8}")
    print("=" * 50)

    if breakdown:
        print("\nBreakdown by query type:")
        for label, sub in breakdown.items():
            counts = sub.pop("count")
            scores = "  ".join(f"{k}={v:.4f}" for k, v in sub.items())
            print(f"  {label} (n={counts}): {scores}")

    # ── 7. Save per-query CSV ──
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    tag = f"_{args.tag}" if args.tag else ""
    csv_path = os.path.join(args.out_dir, f"per_query{tag}_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["meeting_id", "query_id", "query_type", "rouge1", "rouge2", "rougeL",
                  "pred_words", "ref_words"]
        if bert_f1s:
            header.append("bertscore_f1")
        writer.writerow(header)
        for i, row in enumerate(rows):
            qtype = "general" if row.get("query_id", "").startswith("gen") else "specific"
            r = per_rouge[i]
            out_row = [
                row["meeting_id"], row["query_id"], qtype,
                f"{r['rouge1']:.4f}", f"{r['rouge2']:.4f}", f"{r['rougeL']:.4f}",
                len(preds[i].split()), len(refs[i].split()),
            ]
            if bert_f1s:
                out_row.append(f"{bert_f1s[i]:.4f}")
            writer.writerow(out_row)
    print(f"\nPer-query CSV saved to: {csv_path}")

    # ── 8. Save comprehensive JSON ──
    json_path = os.path.join(args.out_dir, f"results{tag}_{timestamp}.json")
    results = {
        "dataset": "QMSum",
        "source": args.preds_jsonl,
        "tag": args.tag or None,
        "instances": len(rows),
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics_agg,
        "length": {k: round(v, 2) for k, v in lens.items()},
        "breakdown": breakdown,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
