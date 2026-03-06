"""
cache.py — Semantic cache built from first principles.

ARCHITECTURE OVERVIEW:
─────────────────────
A traditional exact-match cache (dict keyed on query string) is useless
for NL queries: "best sci-fi books" and "top science fiction novels"
would both miss. Our cache instead uses *embedding cosine similarity*:
a new query is a hit if we already have a cached query whose embedding
is within threshold θ.

THE CLUSTER-BUCKETED DATA STRUCTURE:
──────────────────────────────────────
Naïve approach: scan all N cached entries, compute cosine similarity with
each, return the best hit above θ. Cost: O(N).

Our approach: bucket cache entries by their dominant cluster. For a new
query, compute its soft membership vector, then only scan entries in its
top-C clusters (C=2 by default). Expected cost: O(N/K) where K is the
number of clusters.

This is the "real work" that Part 2 clustering does for Part 3:
  • With K=20 clusters and N=1000 cache entries → ~50 comparisons vs 1000.
  • As the cache grows, the speedup grows proportionally.

The structure:
    _buckets: Dict[int, List[CacheEntry]]
        cluster_id → list of cached entries whose dominant cluster = cluster_id

    _stats: hit/miss counters

THE THRESHOLD θ — THE CORE TUNABLE:
─────────────────────────────────────
θ ∈ (0, 1) controls the "similarity radius" for a cache hit.

  θ = 0.95 — very strict: only nearly-identical rephrasing hits.
    Behaviour: cache rarely helps; system is essentially stateless.
    What it reveals: the query distribution has low paraphrase overlap.

  θ = 0.85 — moderate: catches most paraphrases and synonym swaps.
    Behaviour: good hit rate without returning semantically distant results.
    This is the recommended default.

  θ = 0.70 — loose: catches topic-similar queries regardless of intent.
    Behaviour: "what is gun control" might hit "second amendment rights" —
    same topic, different question. Cache becomes incorrect: it returns an
    answer that doesn't actually address the new query.

  θ < 0.60 — dangerously loose: almost everything hits. The cache becomes
    a noise generator. Correctness collapses.

This trade-off is fundamental and no heuristic can resolve it — the right
θ depends on the application's tolerance for false hits vs. recomputation.
We expose it as a runtime parameter and log the threshold in every response
so callers can tune it empirically with real traffic.

DESIGN DECISIONS — what is NOT used:
  • Redis: excluded by spec. Also overkill: Redis adds a network hop and
    process boundary for something that can live in-process with equal
    correctness and better latency.
  • FAISS for cache lookup: The cache typically holds tens-to-hundreds of
    entries — linear scan with numpy dot product is faster than FAISS
    index build/query overhead at that scale.
  • LRU eviction: Not implemented. The spec is silent on cache eviction, and
    adding arbitrary eviction would change cache semantics. In production,
    this would be parameterised by memory budget.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.85  # see docstring for analysis


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray          # shape (384,)
    soft_memberships: np.ndarray   # shape (n_clusters,)
    dominant_cluster: int
    result: Any
    timestamp: float = field(default_factory=time.time)


class SemanticCache:
    """
    Cluster-bucketed semantic cache.

    Thread-safe via a single RLock. In a production multi-worker setup,
    you would serialise the cache to shared memory or a fast kv store —
    but for a single-process uvicorn deployment (which the spec implies),
    this is correct and sufficient.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, n_clusters: int = 20):
        """
        Parameters
        ----------
        threshold : float
            Cosine similarity above which a new query is considered a hit.
            See module docstring for a detailed analysis of different values.
        n_clusters : int
            Number of GMM clusters (used to size the bucket dict).
        """
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")

        self.threshold = threshold
        self.n_clusters = n_clusters

        # Core data structure: cluster_id → list of entries
        self._buckets: Dict[int, List[CacheEntry]] = {
            k: [] for k in range(n_clusters)
        }
        self._total_entries: int = 0
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def lookup(
        self,
        query_embedding: np.ndarray,
        soft_memberships: np.ndarray,
        top_clusters: int = 2,
    ) -> Tuple[bool, Optional[CacheEntry], float]:
        """
        Search the cache for a semantically similar query.

        Strategy:
          1. Identify the top-C clusters by membership probability.
          2. Scan only those C buckets (O(N/K) on average).
          3. Return the entry with highest cosine similarity if above threshold.

        Returns
        -------
        hit : bool
        best_entry : CacheEntry or None
        best_score : float  (similarity of the best match, 0 if miss)
        """
        candidate_clusters = np.argsort(soft_memberships)[::-1][:top_clusters]

        best_score = 0.0
        best_entry: Optional[CacheEntry] = None

        with self._lock:
            for cluster_id in candidate_clusters:
                for entry in self._buckets[cluster_id]:
                    # Cosine similarity: embeddings are L2-normalised so this
                    # is just a dot product.
                    sim = float(np.dot(query_embedding, entry.embedding))
                    if sim > best_score:
                        best_score = sim
                        best_entry = entry

        hit = best_score >= self.threshold
        with self._lock:
            if hit:
                self._hit_count += 1
            else:
                self._miss_count += 1

        return hit, best_entry, best_score

    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        soft_memberships: np.ndarray,
        result: Any,
    ) -> CacheEntry:
        """
        Insert a new entry into the appropriate cluster bucket.
        """
        dominant_cluster = int(np.argmax(soft_memberships))
        entry = CacheEntry(
            query=query,
            embedding=query_embedding.copy(),
            soft_memberships=soft_memberships.copy(),
            dominant_cluster=dominant_cluster,
            result=result,
        )
        with self._lock:
            self._buckets[dominant_cluster].append(entry)
            self._total_entries += 1

        logger.debug(
            f"Cached query '{query[:50]}' in cluster {dominant_cluster} "
            f"(bucket now has {len(self._buckets[dominant_cluster])} entries)"
        )
        return entry

    def flush(self) -> None:
        """Clear all entries and reset stats."""
        with self._lock:
            for k in self._buckets:
                self._buckets[k] = []
            self._total_entries = 0
            self._hit_count = 0
            self._miss_count = 0
        logger.info("Cache flushed.")

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total if total > 0 else 0.0
            bucket_sizes = {k: len(v) for k, v in self._buckets.items() if v}
            return {
                "total_entries": self._total_entries,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(hit_rate, 4),
                "threshold": self.threshold,
                "n_clusters": self.n_clusters,
                "bucket_distribution": bucket_sizes,
            }

    def set_threshold(self, new_threshold: float) -> None:
        """
        Runtime threshold adjustment for exploration / A-B testing.
        Existing entries remain; only future lookups use the new value.
        """
        if not (0.0 < new_threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {new_threshold}")
        with self._lock:
            old = self.threshold
            self.threshold = new_threshold
        logger.info(f"Cache threshold changed: {old} → {new_threshold}")

    def __len__(self) -> int:
        return self._total_entries
