import json
import evaluate

# Load evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

# Load model output and reference summary
try:
    with open("output/summary.json", encoding="utf-8") as f:
        model_output = json.load(f)["summary"]
    with open("references/ref_summary_general.json", encoding="utf-8") as f:
        reference = json.load(f)["summary"]
except FileNotFoundError as e:
    print(f"[Error] Missing file: {e.filename}")
    exit(1)
except KeyError:
    print("[Error] JSON file missing required 'summary' field.")
    exit(1)

# Compute ROUGE Scores
rouge_scores = rouge.compute(
    predictions=[model_output],
    references=[reference],
    use_stemmer=True
)

# Compute BLEU Score (tokenized input)
bleu_scores = bleu.compute(
    predictions=[model_output],
    references=[[reference]]
)

# Compute METEOR Score
meteor_score = meteor.compute(
    predictions=[model_output],
    references=[reference]
)

# Display the results
print("\n=== ROUGE Scores ===")
for key, val in rouge_scores.items():
    print(f"{key.upper()}: {val:.4f}")

print("\n=== BLEU Score ===")
print(f"BLEU: {bleu_scores['bleu']:.4f}")

print("\n=== METEOR Score ===")
print(f"METEOR: {meteor_score['meteor']:.4f}")
