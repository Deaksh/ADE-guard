from transformers import pipeline

# Load once at module level for performance
MODEL_PATH = "/Users/deakshshetty/Documents/ADE-Guard/backend/models/ner_biobert_output/checkpoint-200"
ner_pipeline = pipeline("ner", model=MODEL_PATH, tokenizer=MODEL_PATH, aggregation_strategy="simple")

def extract_entities(text: str):
    results = ner_pipeline(text)
    entities = []
    # Postprocess possible '##' subword tokens and merge them if needed
    for r in results:
        # Clean token text from tokenizer artifacts if needed
        word = r["word"].replace("##", "") if r["word"].startswith("##") else r["word"]
        entities.append({
            "text": word,
            "label": r["entity_group"],
            "start": r["start"],
            "end": r["end"],
            "score": float(r["score"]),
        })
    return entities
