"""
vector_store.py — Numpy-based vector store (no ChromaDB / onnxruntime needed).

Stores embeddings and metadata as numpy arrays on disk.
Supports filtered retrieval by dominant_cluster.
"""

import logging
import os
import json
import numpy as np
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PERSIST_DIR = os.environ.get("VECTOR_STORE_DIR", "data/vecstore")


def _paths():
    return {
        "embeddings": os.path.join(PERSIST_DIR, "embeddings.npy"),
        "texts":      os.path.join(PERSIST_DIR, "texts.json"),
        "metadata":   os.path.join(PERSIST_DIR, "metadata.json"),
    }


def add_documents(
    texts: List[str],
    embeddings: np.ndarray,
    labels: List[int],
    target_names: List[str],
    dominant_clusters: Optional[List[int]] = None,
    soft_memberships=None,
    batch_size: int = 500,
) -> None:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    p = _paths()

    metadata = []
    for i, (label, text) in enumerate(zip(labels, texts)):
        meta: Dict[str, Any] = {
            "label": label,
            "category": target_names[label],
            "doc_id": i,
        }
        if dominant_clusters is not None:
            meta["dominant_cluster"] = int(dominant_clusters[i])
        metadata.append(meta)

    np.save(p["embeddings"], embeddings.astype(np.float32))
    with open(p["texts"], "w", encoding="utf-8") as f:
        json.dump(texts, f)
    with open(p["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    logger.info(f"Saved {len(texts)} documents to {PERSIST_DIR}")


def _load():
    p = _paths()
    embeddings = np.load(p["embeddings"])
    with open(p["texts"], "r", encoding="utf-8") as f:
        texts = json.load(f)
    with open(p["metadata"], "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return embeddings, texts, metadata


def query(
    query_embedding: np.ndarray,
    top_k: int = 5,
    cluster_filter: Optional[int] = None,
) -> List[Dict[str, Any]]:
    embeddings, texts, metadata = _load()

    # Apply cluster filter
    if cluster_filter is not None:
        indices = [i for i, m in enumerate(metadata)
                   if m.get("dominant_cluster") == cluster_filter]
        if len(indices) < top_k:
            indices = list(range(len(texts)))  # fallback to all
    else:
        indices = list(range(len(texts)))

    sub_emb = embeddings[indices]

    # Cosine similarity (embeddings are already normalised)
    sims = sub_emb @ query_embedding

    top_local = np.argsort(sims)[::-1][:top_k]

    results = []
    for local_idx in top_local:
        global_idx = indices[local_idx]
        results.append({
            "text": texts[global_idx][:300],
            "metadata": metadata[global_idx],
            "similarity": round(float(sims[local_idx]), 4),
        })
    return results


def count() -> int:
    try:
        _, texts, _ = _load()
        return len(texts)
    except Exception:
        return 0
