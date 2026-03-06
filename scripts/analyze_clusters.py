#!/usr/bin/env python3
"""
analyze_clusters.py — Convince a sceptical reader that the clusters are meaningful.

This script produces a semantic audit of the GMM clustering, including:
  - Top representative documents per cluster
  - Category label distribution per cluster
  - Boundary/uncertain documents
  - BIC curve
  - Cluster entropy distribution (how "fuzzy" is each document)

Run AFTER build_index.py:
    python scripts/analyze_clusters.py
"""

import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_and_clean

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
EMBEDDINGS_PATH = "data/embeddings.npy"


def load_all():
    texts, labels, target_names = load_and_clean()
    embeddings = np.load(EMBEDDINGS_PATH)
    memberships = np.load(f"{MODELS_DIR}/memberships.npy")
    with open(f"{MODELS_DIR}/config.pkl", "rb") as f:
        config = pickle.load(f)
    bic_data = np.load(f"{MODELS_DIR}/bic_scores.npy")
    return texts, labels, target_names, embeddings, memberships, config, bic_data


def print_separator(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def main():
    print("Loading artifacts…")
    texts, labels, target_names, embeddings, memberships, config, bic_data = load_all()
    n_clusters = config["n_clusters"]
    dominant = np.argmax(memberships, axis=1)

    # ── BIC Curve ──────────────────────────────────────────────────────
    print_separator("BIC SCORES (cluster selection justification)")
    print(f"  {'k':>4}  {'BIC':>14}  {'Δ BIC':>12}")
    bic_sorted = bic_data[bic_data[:, 0].argsort()]
    prev_bic = None
    for k, bic in bic_sorted:
        delta = f"{bic - prev_bic:+.1f}" if prev_bic is not None else "      —"
        marker = " ◄ SELECTED" if int(k) == n_clusters else ""
        print(f"  {int(k):>4}  {bic:>14.1f}  {delta:>12}{marker}")
        prev_bic = bic

    # ── Cluster compositions ───────────────────────────────────────────
    print_separator(f"CLUSTER LABEL COMPOSITIONS ({n_clusters} clusters)")
    for c in range(n_clusters):
        mask = dominant == c
        if mask.sum() == 0:
            continue
        cluster_labels = [labels[i] for i in range(len(labels)) if mask[i]]
        from collections import Counter
        top_cats = Counter(cluster_labels).most_common(4)
        top_str = ", ".join(f"{target_names[lbl]}({cnt})" for lbl, cnt in top_cats)
        avg_entropy = float(-np.sum(
            memberships[mask] * np.log(memberships[mask] + 1e-10), axis=1
        ).mean())
        print(f"  Cluster {c:2d} | {mask.sum():5d} docs | entropy={avg_entropy:.2f} | {top_str}")

    # ── Representative documents per cluster ───────────────────────────
    print_separator("REPRESENTATIVE DOCUMENTS (highest membership confidence)")
    for c in range(min(n_clusters, 5)):   # show first 5 clusters
        mask = dominant == c
        if mask.sum() == 0:
            continue
        confidence = memberships[mask, c]
        idxs = np.where(mask)[0]
        top_idx = idxs[np.argsort(confidence)[::-1][:2]]
        print(f"\n  -- Cluster {c} --")
        for idx in top_idx:
            snippet = " ".join(texts[idx].split()[:30])
            print(f"    [{target_names[labels[idx]]}] {snippet}…")

    # ── Boundary / uncertain documents ────────────────────────────────
    print_separator("BOUNDARY DOCUMENTS (most semantically ambiguous)")
    entropy = -np.sum(memberships * np.log(memberships + 1e-10), axis=1)
    boundary = np.argsort(entropy)[::-1][:10]
    print(f"  {'doc':>6}  {'category':30}  {'top cluster (p)':20}  {'2nd cluster (p)':20}")
    for idx in boundary:
        top2 = np.argsort(memberships[idx])[::-1][:2]
        print(
            f"  {idx:>6}  {target_names[labels[idx]]:30}  "
            f"C{top2[0]}({memberships[idx][top2[0]]:.3f})            "
            f"C{top2[1]}({memberships[idx][top2[1]]:.3f})"
        )

    # ── Entropy distribution ──────────────────────────────────────────
    print_separator("ENTROPY DISTRIBUTION (how 'fuzzy' is the corpus?)")
    print(f"  Mean entropy:   {entropy.mean():.3f}")
    print(f"  Median entropy: {np.median(entropy):.3f}")
    print(f"  Max entropy:    {entropy.max():.3f}  (max possible = {np.log(n_clusters):.3f})")
    print(f"  % of docs with entropy > 2.0: {(entropy > 2.0).mean():.1%}")
    print(
        "\n  Interpretation: A high mean entropy confirms that many documents "
        "genuinely belong to multiple clusters — validating the choice of "
        "fuzzy over hard clustering."
    )

    # ── Cross-cluster docs ────────────────────────────────────────────
    print_separator("MULTI-CLUSTER MEMBERSHIP EXAMPLES")
    print("  Documents with >20% probability mass in 2+ clusters:\n")
    multi_cluster = [(i, memberships[i]) for i in range(len(texts))
                     if (memberships[i] > 0.20).sum() >= 2]
    for idx, m in multi_cluster[:8]:
        top3 = np.argsort(m)[::-1][:3]
        membership_str = "  ".join(f"C{c}={m[c]:.2f}" for c in top3 if m[c] > 0.05)
        snippet = " ".join(texts[idx].split()[:15])
        print(f"  [{target_names[labels[idx]]}] {snippet}…")
        print(f"    {membership_str}\n")


if __name__ == "__main__":
    main()
