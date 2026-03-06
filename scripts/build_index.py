#!/usr/bin/env python3
"""
build_index.py — One-time setup script.

Run this before starting the API server:
    python scripts/build_index.py

What it does:
  1. Loads and cleans the 20 Newsgroups corpus
  2. Embeds all documents with sentence-transformers
  3. Fits GMM fuzzy clustering (includes BIC-based cluster selection)
  4. Upserts documents + embeddings into ChromaDB with cluster metadata
  5. Saves GMM / PCA / scaler artifacts to models/

Estimated runtime: 5–15 minutes depending on CPU (embedding is the bottleneck).
"""

import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_and_clean
from src.embedder import embed
from src.clustering import (
    reduce_dimensions,
    select_n_clusters,
    fit_gmm,
    get_soft_memberships,
    get_dominant_clusters,
    save_artifacts,
)
from src.vector_store import add_documents, count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    t0 = time.time()

    # ── 1. Load and clean corpus ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1/4: Loading and cleaning corpus")
    logger.info("=" * 60)
    texts, labels, target_names = load_and_clean(subset="all")
    logger.info(f"Corpus size after cleaning: {len(texts)} documents")

    # ── 2. Embed ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2/4: Embedding documents")
    logger.info("(This is the slow step — ~5-10 mins on CPU)")
    logger.info("=" * 60)
    emb_path = "data/embeddings.npy"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(emb_path):
        logger.info(f"Loading cached embeddings from {emb_path}")
        embeddings = np.load(emb_path)
        if len(embeddings) != len(texts):
            logger.warning("Cached embedding count mismatch — re-embedding")
            embeddings = None

    if not os.path.exists(emb_path) or embeddings is None:
        embeddings = embed(texts, batch_size=64, show_progress=True)
        np.save(emb_path, embeddings)
        logger.info(f"Embeddings saved to {emb_path} (shape={embeddings.shape})")

    # ── 3. Fuzzy clustering ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 3/4: Fitting fuzzy clusters (GMM)")
    logger.info("=" * 60)

    X_reduced, scaler, pca = reduce_dimensions(embeddings, n_components=50, fit=True)

    logger.info("Sweeping cluster counts [10, 15, 20, 25, 30] via BIC…")
    best_k, bic_scores = select_n_clusters(X_reduced, candidates=[10, 15, 20, 25, 30])
    logger.info(f"Selected k={best_k} clusters by BIC")

    logger.info(f"Fitting final GMM with k={best_k} (n_init=5 for stability)…")
    gmm = fit_gmm(X_reduced, n_components=best_k, n_init=5)

    memberships = get_soft_memberships(gmm, X_reduced)   # (n_docs, k)
    dominant = get_dominant_clusters(memberships)         # (n_docs,)

    # Log some stats about cluster distribution
    cluster_sizes = np.bincount(dominant, minlength=best_k)
    logger.info("Cluster size distribution:")
    for i, size in enumerate(cluster_sizes):
        logger.info(f"  Cluster {i:2d}: {size:5d} docs ({size / len(texts):.1%})")

    # Log a few high-uncertainty (boundary) documents
    entropy = -np.sum(memberships * np.log(memberships + 1e-10), axis=1)
    boundary_indices = np.argsort(entropy)[-5:]
    logger.info("\nMost uncertain documents (boundary cases):")
    for idx in boundary_indices:
        top2 = np.argsort(memberships[idx])[::-1][:2]
        logger.info(
            f"  doc_{idx} | cat={target_names[labels[idx]]} "
            f"| top clusters: {top2[0]}({memberships[idx][top2[0]]:.2f}), "
            f"{top2[1]}({memberships[idx][top2[1]]:.2f})"
        )

    save_artifacts(gmm, scaler, pca, memberships, bic_scores, best_k)

    # ── 4. Build vector store ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 4/4: Upserting documents into ChromaDB")
    logger.info("=" * 60)
    add_documents(
        texts=texts,
        embeddings=embeddings,
        labels=labels,
        target_names=target_names,
        dominant_clusters=dominant.tolist(),
    )
    logger.info(f"Vector store now contains {count()} documents")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"Index build complete in {elapsed / 60:.1f} minutes.")
    logger.info("You can now start the API with:")
    logger.info("  uvicorn api.main:app --host 0.0.0.0 --port 8000")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
