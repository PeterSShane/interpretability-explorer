from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, output_attentions=True)
model.eval()

def get_attention_and_prediction(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    pred_label = "POSITIVE" if torch.argmax(probs) == 1 else "NEGATIVE"
    confidence = torch.max(probs).item()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return {
        "tokens": tokens,
        "attentions": attentions,
        "prediction": pred_label,
        "confidence": confidence
    }

if __name__ == "__main__":
    result = get_attention_and_prediction("The movie was surprisingly good.")
    print("Tokens:", result["tokens"])
    print("Prediction:", result["prediction"], f"({result['confidence']:.2%})")
    print("Number of layers:", len(result["attentions"]))
    print("Shape of one layer's attention:", result["attentions"][0].shape)