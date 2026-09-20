from extraction import get_attention_and_prediction
from bias_pairs import BIAS_PAIRS
import json

def run_full_bias_audit():
    results_by_category = {}

    for category, pairs in BIAS_PAIRS.items():
        category_results = []

        for sent_a, sent_b in pairs:
            result_a = get_attention_and_prediction(sent_a)
            result_b = get_attention_and_prediction(sent_b)

            conf_delta = abs(result_a["confidence"] - result_b["confidence"])
            prediction_flipped = result_a["prediction"] != result_b["prediction"]

            category_results.append({
                "sentence_a": sent_a,
                "sentence_b": sent_b,
                "prediction_a": result_a["prediction"],
                "prediction_b": result_b["prediction"],
                "confidence_a": result_a["confidence"],
                "confidence_b": result_b["confidence"],
                "confidence_delta": conf_delta,
                "prediction_flipped": prediction_flipped
            })

        avg_delta = sum(r["confidence_delta"] for r in category_results) / len(category_results)
        flip_count = sum(1 for r in category_results if r["prediction_flipped"])

        results_by_category[category] = {
            "pairs": category_results,
            "avg_confidence_delta": avg_delta,
            "flip_count": flip_count,
            "total_pairs": len(category_results)
        }

        print(f"{category}: avg confidence delta = {avg_delta:.2%}, flips = {flip_count}/{len(category_results)}")

    return results_by_category

if __name__ == "__main__":
    results = run_full_bias_audit()
    with open("bias_audit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full results to bias_audit_results.json")