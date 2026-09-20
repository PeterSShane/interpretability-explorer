from datasets import load_dataset
from extraction import get_attention_and_prediction
import json

NEGATION_WORDS = {"not", "no", "never", "n't", "nothing", "neither", "nor", "none"}

def has_negation(sentence):
    words = sentence.lower().split()
    return any(neg in word for word in words for neg in NEGATION_WORDS)

def run_full_analysis():
    dataset = load_dataset("stanfordnlp/sst2", split="validation")

    all_results = []
    misclassified = []

    total = len(dataset)
    for i, example in enumerate(dataset):
        sentence = example["sentence"]
        true_label = "POSITIVE" if example["label"] == 1 else "NEGATIVE"
        result = get_attention_and_prediction(sentence)
        correct = result["prediction"] == true_label

        entry = {
            "sentence": sentence,
            "true_label": true_label,
            "predicted": result["prediction"],
            "confidence": result["confidence"],
            "correct": correct,
            "has_negation": has_negation(sentence),
            "word_count": len(sentence.split())
        }
        all_results.append(entry)

        if not correct:
            misclassified.append(entry)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total}")

    # Overall accuracy
    overall_accuracy = sum(1 for r in all_results if r["correct"]) / len(all_results)

    # Accuracy on negation vs non-negation sentences
    negation_results = [r for r in all_results if r["has_negation"]]
    non_negation_results = [r for r in all_results if not r["has_negation"]]
    negation_accuracy = sum(1 for r in negation_results if r["correct"]) / len(negation_results) if negation_results else 0
    non_negation_accuracy = sum(1 for r in non_negation_results if r["correct"]) / len(non_negation_results) if non_negation_results else 0

    # Accuracy by sentence length (short vs long, split at median-ish threshold)
    short_results = [r for r in all_results if r["word_count"] <= 10]
    long_results = [r for r in all_results if r["word_count"] > 10]
    short_accuracy = sum(1 for r in short_results if r["correct"]) / len(short_results) if short_results else 0
    long_accuracy = sum(1 for r in long_results if r["correct"]) / len(long_results) if long_results else 0

    summary = {
        "total_examples": len(all_results),
        "overall_accuracy": overall_accuracy,
        "negation_accuracy": negation_accuracy,
        "negation_count": len(negation_results),
        "non_negation_accuracy": non_negation_accuracy,
        "non_negation_count": len(non_negation_results),
        "short_sentence_accuracy": short_accuracy,
        "short_sentence_count": len(short_results),
        "long_sentence_accuracy": long_accuracy,
        "long_sentence_count": len(long_results),
        "total_misclassified": len(misclassified)
    }

    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary, misclassified[:20]  # keep only top 20 for the dashboard dropdown

if __name__ == "__main__":
    summary, top_misclassified = run_full_analysis()

    with open("error_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("misclassified_cache.json", "w") as f:
        json.dump(top_misclassified, f, indent=2)

    print("\nSaved error_analysis_summary.json and misclassified_cache.json")