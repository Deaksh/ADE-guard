from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_PATH = "/Users/deakshshetty/Documents/ADE-Guard/backend/models/severity_roberta/checkpoint-112"

def test_severity(text):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    results = classifier(text, top_k=3)
    for r in results:
        print(f"Label: {r['label']}, Score: {r['score']:.3f}")

if __name__ == "__main__":
    sample_text = "Patient experienced severe headache and nausea after vaccination."
    test_severity(sample_text)
