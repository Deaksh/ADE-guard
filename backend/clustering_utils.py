import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
from .ner_utils import extract_entities  # make sure import matches your project structure
import warnings
from transformers import logging

warnings.filterwarnings("ignore", message="Asking to truncate to max_length but no maximum length is provided")
logging.set_verbosity_error()

# Load tokenizer and model once globally
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = 512
tokenizer._max_len_single_sentence = 512
tokenizer._max_len_sentences_pair = 512

model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def get_age_group(age):
    if age is None or pd.isna(age):
        return "Unknown"
    try:
        age = float(age)
    except:
        return "Unknown"
    if age <= 17:
        return "0-17"
    elif age <= 30:
        return "18-30"
    elif age <= 50:
        return "31-50"
    else:
        return "51+"

def detect_modifier(text: str):
    text_lower = text.lower()
    if "severe" in text_lower:
        return "Severe"
    elif "moderate" in text_lower:
        return "Moderate"
    elif "mild" in text_lower:
        return "Mild"
    else:
        return "Unknown"

def embed_texts(texts):
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation='longest_first',
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**encoded)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS embeddings
    return embeddings

def cluster_ades(df: pd.DataFrame, max_clusters=50):
    if "AGE_YRS" not in df.columns:
        df["AGE_YRS"] = pd.NA
    df["AGE_GROUP"] = df["AGE_YRS"].apply(get_age_group)

    texts = []
    meta = []
    for _, row in df.iterrows():
        text = str(row.get("SYMPTOM_TEXT", "")).strip()
        if not text:
            continue
        texts.append(text)
        age_group = row["AGE_GROUP"]
        modifier = detect_modifier(text)
        meta.append((age_group, modifier))

    if not texts:
        return []

    embeddings = embed_texts(texts)

    n_clusters = min(max_clusters, len(embeddings))
    if n_clusters == 0:
        return []

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for idx, label in enumerate(labels):
        age_group, modifier = meta[idx]
        entities = extract_entities(texts[idx])
        ade_symptoms = {e["text"].lower() for e in entities if e["label"].upper() in ["ADE", "DRUG"]}

        if label not in clusters:
            clusters[label] = {
                "cluster_id": label + 1,
                "age_group": age_group,
                "modifier": modifier,
                "symptoms": ade_symptoms.copy()
            }
        else:
            clusters[label]["symptoms"].update(ade_symptoms)

    # Convert symptom sets to lists for serialization
    for c in clusters.values():
        c["symptoms"] = list(c["symptoms"])

    return [clusters[k] for k in sorted(clusters.keys())[:max_clusters]]
