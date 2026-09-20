from datasets import load_dataset
from extraction import get_attention_and_prediction
import json

def find_misclassified(n=20):
    dataset = load_dataset("stanfordnlp/sst2", split="validation")
    misclassified = []

    for example in dataset:
        sentence = example["sentence"]
        true_label = "POSITIVE" if example["label"] == 1 else "NEGATIVE"
        result = get_attention_and_prediction(sentence)

        if result["prediction"] != true_label:
            misclassified.append({
                "sentence": sentence,
                "true_label": true_label,
                "predicted": result["prediction"],
                "confidence": result["confidence"]
            })
        if len(misclassified) >= n:
            break

    return misclassified

if __name__ == "__main__":
    results = find_misclassified(20)
    with open("misclassified_cache.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Found and cached {len(results)} misclassified examples.")