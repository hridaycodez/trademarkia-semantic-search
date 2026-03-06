"""
main.py — FastAPI service for semantic search with cache.

Startup sequence:
  1. Load embedding model (sentence-transformers, downloaded on first run)
  2. Load GMM + PCA + scaler from models/
  3. Initialise SemanticCache
  4. ChromaDB client lazily initialised on first query

All heavy state is held in module-level singletons, so the entire service
stays stateful within a single uvicorn process.

NOTE: If you run with multiple workers (--workers N), each worker has its
own SemanticCache instance — caches won't share state across workers.
For this assignment, run with a single worker (the default).
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path so imports work regardless of working directory
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedder import embed_query
from src.clustering import load_artifacts, get_query_memberships
from src.cache import SemanticCache
from src.vector_store import query as vdb_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Global state                                                         #
# ------------------------------------------------------------------ #

_gmm = None
_scaler = None
_pca = None
_config: Dict = {}
_cache: Optional[SemanticCache] = None

CACHE_THRESHOLD = float(os.environ.get("CACHE_THRESHOLD", "0.85"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, clean up at shutdown."""
    global _gmm, _scaler, _pca, _config, _cache

    logger.info("Loading clustering artifacts…")
    try:
        _gmm, _scaler, _pca, _config = load_artifacts()
        n_clusters = _config["n_clusters"]
        logger.info(f"Clustering artifacts loaded (n_clusters={n_clusters})")
    except FileNotFoundError:
        logger.error(
            "Clustering models not found! Run scripts/build_index.py first."
        )
        raise

    _cache = SemanticCache(threshold=CACHE_THRESHOLD, n_clusters=_config["n_clusters"])
    logger.info(f"SemanticCache initialised (threshold={CACHE_THRESHOLD})")

    # Warm up the embedding model (first inference loads the ONNX/torch model)
    logger.info("Warming up embedding model…")
    embed_query("warmup")
    logger.info("Service ready.")

    yield  # --- app runs here ---

    logger.info("Shutting down.")


app = FastAPI(
    title="Semantic Search with Semantic Cache",
    description=(
        "20 Newsgroups semantic search backed by GMM fuzzy clustering "
        "and a cluster-bucketed semantic cache."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------ #
#  Pydantic schemas                                                     #
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, example="What are the health risks of smoking?")
    top_k: int = Field(default=5, ge=1, le=20)
    threshold_override: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=0.99,
        description="Override the default similarity threshold for this request only.",
    )


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: float
    result: Any
    dominant_cluster: int
    cluster_memberships: list  # top-5 (cluster_id, probability) pairs


class CacheStats(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


# ------------------------------------------------------------------ #
#  Helpers                                                              #
# ------------------------------------------------------------------ #

def _get_top_cluster_memberships(memberships, top_n=5):
    indices = memberships.argsort()[::-1][:top_n]
    return [(int(i), round(float(memberships[i]), 4)) for i in indices]


def _compute_result(query: str, query_embedding, dominant_cluster: int, top_k: int) -> Dict:
    """
    Compute the actual search result by querying the vector DB.
    This is what gets cached — the raw VDB result.
    """
    docs = vdb_query(
        query_embedding=query_embedding,
        top_k=top_k,
        cluster_filter=dominant_cluster,
    )
    # If cluster filter returns too few results (small cluster), fall back
    if len(docs) < top_k:
        docs = vdb_query(query_embedding=query_embedding, top_k=top_k)

    return {
        "top_documents": docs,
        "n_results": len(docs),
    }


# ------------------------------------------------------------------ #
#  Endpoints                                                            #
# ------------------------------------------------------------------ #

@app.post("/query", response_model=QueryResponse)
async def semantic_query(body: QueryRequest):
    """
    Main search endpoint.

    Flow:
      1. Embed the query.
      2. Get soft GMM cluster memberships.
      3. Check the semantic cache (cluster-bucketed O(N/K) lookup).
      4a. Cache HIT: return cached result immediately.
      4b. Cache MISS: query the vector DB, cache the result, return it.
    """
    if _cache is None or _gmm is None:
        raise HTTPException(status_code=503, detail="Service not initialised")

    # Step 1: embed
    q_emb = embed_query(body.query)

    # Step 2: soft cluster memberships
    memberships = get_query_memberships(q_emb, _gmm, _scaler, _pca)
    dominant_cluster = int(memberships.argmax())
    top_memberships = _get_top_cluster_memberships(memberships)

    # Step 3: cache lookup (use per-request threshold override if provided)
    effective_threshold = body.threshold_override or _cache.threshold
    old_threshold = None
    if body.threshold_override is not None:
        old_threshold = _cache.threshold
        _cache.set_threshold(body.threshold_override)

    hit, entry, sim_score = _cache.lookup(q_emb, memberships)

    if old_threshold is not None:
        _cache.set_threshold(old_threshold)

    if hit and entry is not None:
        # Cache HIT
        logger.info(f"Cache HIT: '{body.query[:50]}' matched '{entry.query[:50]}' (sim={sim_score:.3f})")
        return QueryResponse(
            query=body.query,
            cache_hit=True,
            matched_query=entry.query,
            similarity_score=round(sim_score, 4),
            result=entry.result,
            dominant_cluster=dominant_cluster,
            cluster_memberships=top_memberships,
        )

    # Cache MISS — compute result
    logger.info(f"Cache MISS: '{body.query[:50]}' (best sim={sim_score:.3f} < θ={effective_threshold})")
    result = _compute_result(body.query, q_emb, dominant_cluster, body.top_k)

    # Store in cache
    _cache.store(
        query=body.query,
        query_embedding=q_emb,
        soft_memberships=memberships,
        result=result,
    )

    return QueryResponse(
        query=body.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=round(sim_score, 4),
        result=result,
        dominant_cluster=dominant_cluster,
        cluster_memberships=top_memberships,
    )


@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    """Return current cache statistics."""
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialised")
    s = _cache.stats()
    return CacheStats(
        total_entries=s["total_entries"],
        hit_count=s["hit_count"],
        miss_count=s["miss_count"],
        hit_rate=s["hit_rate"],
    )


@app.delete("/cache")
async def flush_cache():
    """Flush the entire cache and reset stats."""
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialised")
    _cache.flush()
    return JSONResponse(content={"message": "Cache flushed.", "status": "ok"})


@app.get("/cache/detail")
async def cache_detail():
    """Extended cache info including per-cluster bucket sizes (debugging)."""
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialised")
    return _cache.stats()


@app.patch("/cache/threshold")
async def set_threshold(threshold: float):
    """
    Update the similarity threshold at runtime.
    Useful for live exploration of threshold behaviour without restart.
    """
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialised")
    if not (0.0 < threshold < 1.0):
        raise HTTPException(status_code=422, detail="threshold must be in (0, 1)")
    _cache.set_threshold(threshold)
    return {"threshold": threshold, "message": "Threshold updated"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cache_entries": len(_cache) if _cache else 0,
        "n_clusters": _config.get("n_clusters"),
        "cache_threshold": _cache.threshold if _cache else None,
    }
