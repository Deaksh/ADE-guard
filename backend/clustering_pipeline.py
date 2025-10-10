# backend/clustering_pipeline.py
from sentence_transformers import SentenceTransformer
import hdbscan
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap

model = SentenceTransformer("all-MiniLM-L6-v2")  # fast; switch to bio-sbert if needed

def get_symptom_texts(df):
    return df["SYMPTOM_TEXT"].fillna("").tolist()

def cluster_symptoms(df, min_cluster_size=15):
    texts = get_symptom_texts(df)
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb2 = reducer.fit_transform(emb)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(emb)
    df_out = df.copy()
    df_out["cluster"] = labels
    df_out["emb_x"] = emb2[:,0]
    df_out["emb_y"] = emb2[:,1]
    return df_out, clusterer
