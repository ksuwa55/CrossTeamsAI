import json, argparse
import evaluate

def read_jsonl(path):
    preds, refs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            preds.append(j["prediction"])
            refs.append(j["reference"])
    return preds, refs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_jsonl", required=True, help="output/qmsum_test_preds.jsonl")
    args = ap.parse_args()

    preds, refs = read_jsonl(args.preds_jsonl)

    rouge = evaluate.load("rouge")
    r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)

    print("=== QMSum TEST ROUGE (F1) ===")
    print(f"ROUGE1: {r['rouge1']:.4f}")
    print(f"ROUGE2: {r['rouge2']:.4f}")
    print(f"ROUGEL: {r['rougeL']:.4f}")
    if "rougeLsum" in r:
        print(f"ROUGELSUM: {r['rougeLsum']:.4f}")
    print(f"Instances evaluated: {len(preds)}")

if __name__ == "__main__":
    main()
