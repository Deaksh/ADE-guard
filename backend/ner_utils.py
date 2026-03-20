import os
from functools import lru_cache
from transformers import pipeline

# Load once at module level for performance
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(BASE_DIR, "models", "ner_biobert_output")
ENV_PATH = os.environ.get("NER_MODEL_PATH")


def _resolve_model_path() -> str:
    candidates = []
    if ENV_PATH:
        candidates.append(ENV_PATH)
    candidates.extend([
        DEFAULT_DIR,
        os.path.join(DEFAULT_DIR, "checkpoint-400"),
        os.path.join(DEFAULT_DIR, "checkpoint-200"),
    ])
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return DEFAULT_DIR


MODEL_PATH = _resolve_model_path()
ner_pipeline = pipeline(
    "ner",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    aggregation_strategy="simple",
)


@lru_cache(maxsize=1024)
def _extract_entities_cached(text: str):
    # Older pipeline versions don't accept truncation params; do manual truncation.
    tokens = ner_pipeline.tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = tokens["input_ids"][0]
    truncated_text = ner_pipeline.tokenizer.decode(input_ids, skip_special_tokens=True)
    results = ner_pipeline(truncated_text)
    entities = []
    for r in results:
        word = r["word"].replace("##", "") if r["word"].startswith("##") else r["word"]
        entities.append({
            "text": word,
            "label": r["entity_group"],
            "start": r["start"],
            "end": r["end"],
            "score": float(r["score"]),
        })
    return entities


def extract_entities(text: str):
    return _extract_entities_cached(text)
