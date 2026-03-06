"""
embedder.py — Sentence embedding wrapper.

DESIGN DECISION — model choice: all-MiniLM-L6-v2
    Alternatives considered:
      • all-mpnet-base-v2: Higher quality but 4× slower and 3× larger.
        Overkill for a 20K-doc corpus where inference speed matters more
        than marginal quality gains.
      • text-embedding-ada-002 (OpenAI): Requires API key + cost per call.
        Inappropriate for a self-contained system.
      • TF-IDF vectors: Fast but purely lexical — cannot detect that
        "gun control" and "firearm legislation" are semantically close.
        The entire point of this system is semantic search, so bag-of-words
        embeddings defeat the purpose.
    all-MiniLM-L6-v2 hits the sweet spot: 384-dim, ~80 MB, fast CPU
    inference, and strong semantic similarity benchmarks on STS tasks.

DECISION — batch_size=64:
    Empirically balances GPU/CPU memory against throughput. Too large and
    CPU runs out of working memory; too small and overhead dominates.

DECISION — normalize_embeddings=True:
    Cosine similarity on L2-normalised vectors is equivalent to dot product,
    which is faster (no division) and required by ChromaDB's cosine metric.
    It also makes the cache's cosine threshold directly interpretable.
"""

import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Singleton so the heavy model loads once per process.
_model: SentenceTransformer | None = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(
    texts: Union[str, List[str]],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed one or more texts. Returns float32 array of shape (n, 384).
    Embeddings are L2-normalised so cosine similarity == dot product.
    """
    if isinstance(texts, str):
        texts = [texts]

    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # key decision — see module docstring
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Convenience wrapper for single-query embedding. Returns shape (384,)."""
    return embed([query])[0]
