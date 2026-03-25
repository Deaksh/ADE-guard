import os
from functools import lru_cache
from transformers import pipeline
from dotenv import load_dotenv

# Load once at module level for performance
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
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
        # If user provided a HF repo id, use it directly
        if path and not os.path.exists(path) and isinstance(path, str) and "/" in path and not path.startswith("/"):
            return path
        if not path or not os.path.exists(path):
            continue
        # Ensure a model file exists in the directory
        has_model = any(
            os.path.exists(os.path.join(path, fname))
            for fname in ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]
        )
        if has_model:
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
    stopwords = {"and", "or", "the", "a", "an", "to", "of", "in", "after", "for", "on", "at", "with",
                 "without", "from", "by", "as", "is", "was", "were", "be", "been", "being"}
    for r in results:
        word = r["word"].replace("##", "") if r["word"].startswith("##") else r["word"]
        cleaned = word.strip()
        if len(cleaned) < 3 or cleaned.lower() in stopwords:
            continue
        entities.append({
            "text": cleaned,
            "label": r["entity_group"],
            "start": r["start"],
            "end": r["end"],
            "score": float(r["score"]),
        })
    return entities


def extract_entities(text: str):
    return _extract_entities_cached(text)
