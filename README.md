# Semantic Search System — 20 Newsgroups

A lightweight semantic search system with fuzzy clustering and a cluster-bucketed semantic cache.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        POST /query                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                   embed query
                 (all-MiniLM-L6-v2)
                          │
               get soft GMM memberships
              (P(cluster_k | query), sums to 1)
                          │
                 ┌────────▼────────┐
                 │  Semantic Cache │  ← cluster-bucketed, O(N/K) lookup
                 └────────┬────────┘
             HIT ◄────────┤────────► MISS
              │           │               │
         return cached    │          query ChromaDB
           result         │        (filtered by dominant cluster)
                          │               │
                          │          cache result
                          │               │
                          └───────────────┘
                                  │
                           return response
```

## Design Decisions

### Part 1 — Preprocessing
- **Strip headers/footers/quotes**: Raw 20NG headers contain `Newsgroup: rec.sport.hockey` — trivially cheating. All content-irrelevant metadata removed.
- **min_words=30**: Sub-30-word posts are noise (bounced mail, one-liners). No semantic signal.
- **max_words=500**: sentence-transformers truncates at ~512 tokens anyway; we truncate early for consistent representation.
- **Embedding model**: `all-MiniLM-L6-v2` — 384 dims, fast CPU inference, strong STS benchmarks. Beats TF-IDF (lexical only) without the cost/dependency of commercial APIs.

### Part 2 — Fuzzy Clustering
- **GMM over fuzzy c-means**: GMM is a proper probabilistic model. `predict_proba()` gives P(cluster | doc) directly, with BIC for principled cluster count selection.
- **PCA to 50 dims before GMM**: GMM covariance estimation is unstable in 384 dims (curse of dimensionality). PCA retains >85% variance while making EM tractable.
- **Cluster count via BIC**: Sweep [10, 15, 20, 25, 30], pick the minimum BIC. Penalises complexity — not chosen for convenience.

### Part 3 — Semantic Cache
- **Cluster-bucketed structure**: `{cluster_id: [CacheEntry]}`. Lookup scans only the top-2 clusters by membership probability → O(N/K) vs O(N).
- **Threshold θ analysis**:
  - `θ=0.95`: Near-identical rephrasing only. Cache rarely helps.
  - `θ=0.85` (**default**): Catches paraphrases and synonym swaps. Correct and efficient.
  - `θ=0.70`: Topic-similar queries hit even with different intent. Correctness degrades.
  - `θ<0.60`: Almost everything hits. Cache becomes a noise generator.
- **No Redis/Memcached**: Built from first principles. In-process dict with threading.RLock.

## Quickstart

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the index (embed + cluster + store) — ~5-15 min on CPU
python scripts/build_index.py

# 4. (Optional) Analyse cluster quality
python scripts/analyze_clusters.py

# 5. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API

### `POST /query`
```json
{
  "query": "What are the health risks of smoking?",
  "top_k": 5,
  "threshold_override": 0.85
}
```
Response:
```json
{
  "query": "What are the health risks of smoking?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.71,
  "result": { "top_documents": [...], "n_results": 5 },
  "dominant_cluster": 7,
  "cluster_memberships": [[7, 0.62], [12, 0.21], [3, 0.09], ...]
}
```

### `GET /cache/stats`
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`
Flushes cache and resets all stats.

### `PATCH /cache/threshold?threshold=0.9`
Update similarity threshold at runtime (no restart needed).

### `GET /health`
Returns service status.

## Docker

```bash
# Build index first (needs to run outside Docker for data persistence)
python scripts/build_index.py

# Then build and run
docker compose up --build
```

The container mounts `./data` and `./models` as volumes for persistence.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CACHE_THRESHOLD` | `0.85` | Cosine similarity threshold for cache hit |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB persistence directory |
| `MODELS_DIR` | `models` | Directory for GMM/PCA/scaler artifacts |

## Project Structure

```
newsgroups_search/
├── src/
│   ├── preprocess.py      # corpus loading and cleaning
│   ├── embedder.py        # sentence-transformers wrapper
│   ├── vector_store.py    # ChromaDB client
│   ├── clustering.py      # GMM fuzzy clustering
│   └── cache.py           # semantic cache (built from scratch)
├── api/
│   └── main.py            # FastAPI application
├── scripts/
│   ├── build_index.py     # one-time index builder
│   └── analyze_clusters.py  # cluster quality analysis
├── data/                  # embeddings + ChromaDB (gitignored)
├── models/                # GMM + PCA artifacts (gitignored)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
