"""FastAPI application for the Multimodal Product Reranker."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from mmrerank.config import Config, load_config
from mmrerank.evaluation.metrics import ndcg_at_k
from mmrerank.models.baselines import BM25OnlyRanker, RandomRanker
from mmrerank.models.ranker import LambdaRankRanker
from mmrerank.utils.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI."""
    startup_event()
    yield


app = FastAPI(
    title="Multimodal Product Reranker API",
    description="Interactive API and UI for reranking e-commerce product candidates with LightGBM LambdaRank & CLIP embeddings",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and config
CFG: Config | None = None
MODELS: dict[str, LambdaRankRanker] = {}
SAMPLE_QUERIES: list[dict] = []
HTML_CONTENT: str = ""


class ProductInput(BaseModel):
    """Input representation of a candidate product."""

    product_id: str = Field(default="prod_1")
    title: str = Field(..., description="Product title")
    esci_label: str = Field(default="I", description="Relevance label (E, S, C, I)")
    has_image: int = Field(default=1, description="1 if product has an image, 0 if missing")
    clip_similarity: float | None = Field(
        default=None, description="CLIP query-image similarity (0-1)"
    )


class RerankRequest(BaseModel):
    """Request payload for product reranking."""

    query: str = Field(..., description="Search query")
    products: list[ProductInput] = Field(..., description="Candidate products to rerank")
    model_variant: str = Field(
        default="text_plus_image",
        description="Model variant: text_plus_image, text_only, bm25_only, image_only, random",
    )


class RankedProduct(BaseModel):
    """Output representation of a reranked product."""

    rank: int
    product_id: str
    title: str
    esci_label: str
    score: float
    bm25_score: float
    token_overlap: float
    title_length: int
    clip_similarity: float | None
    has_image: int
    gain: float
    rank_delta: int | None = None


class RerankResponse(BaseModel):
    """Response payload for reranking."""

    query: str
    model_variant: str
    ranked_products: list[RankedProduct]
    ndcg_10: float
    is_visual_query: bool
    visual_keywords: list[str]
    latency_ms: float


class CompareResponse(BaseModel):
    """Side-by-side comparison response."""

    query: str
    is_visual_query: bool
    visual_keywords: list[str]
    text_only: list[RankedProduct]
    text_only_ndcg: float
    text_plus_image: list[RankedProduct]
    text_plus_image_ndcg: float
    delta_ndcg: float
    latency_ms: float


def startup_event() -> None:
    """Load configuration, sample queries, and trained models on startup."""
    global CFG, MODELS, SAMPLE_QUERIES, HTML_CONTENT
    setup_logging()
    CFG = load_config()

    # Load sample queries
    sample_path = Path(__file__).parent / "data" / "sample_queries.json"
    if sample_path.exists():
        with open(sample_path, encoding="utf-8") as f:
            SAMPLE_QUERIES = json.load(f)
        logger.info("Loaded %d sample queries from %s", len(SAMPLE_QUERIES), sample_path)

    # Load models
    models_dir = Path("reports/models")
    for variant in ["text_only", "text_plus_image"]:
        model_file = models_dir / f"{variant}.pkl"
        if model_file.exists():
            ranker = LambdaRankRanker(CFG.lightgbm, seed=CFG.seed)
            ranker.load(model_file)
            MODELS[variant] = ranker
            logger.info("Loaded %s model from %s", variant, model_file)
        else:
            logger.warning("Model file not found at %s", model_file)

    # Load UI HTML template
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        with open(template_path, encoding="utf-8") as f:
            HTML_CONTENT = f.read()


def _check_visual_query(query: str, lexicon: list[str]) -> tuple[bool, list[str]]:
    """Check if query contains visual lexicon terms."""
    query_tokens = set(query.lower().split())
    matched = [w for w in lexicon if w in query_tokens]
    return len(matched) > 0, matched


def _extract_features(
    query: str,
    products: list[ProductInput],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract the 5 standard features for candidates of a query."""
    n = len(products)
    titles = [p.title for p in products]

    # 1. BM25 score
    tokenized_titles = [t.lower().split() for t in titles]
    tokenized_query = query.lower().split()
    if any(tokenized_titles):
        bm25_index = BM25Okapi(tokenized_titles)
        bm25_scores = np.array(bm25_index.get_scores(tokenized_query), dtype=np.float32)
    else:
        bm25_scores = np.zeros(n, dtype=np.float32)

    # 2. Token overlap
    query_set = set(tokenized_query)
    token_overlaps = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(titles):
        t_tokens = set(t.lower().split())
        token_overlaps[i] = len(query_set & t_tokens) / max(len(query_set), 1)

    # 3. Title length
    title_lengths = np.array([len(t) for t in titles], dtype=np.float32)

    # 4. CLIP similarity & 5. has_image
    clip_sims = np.full(n, np.nan, dtype=np.float32)
    has_images = np.zeros(n, dtype=np.float32)
    for i, p in enumerate(products):
        has_images[i] = float(p.has_image)
        if p.has_image and p.clip_similarity is not None:
            clip_sims[i] = float(p.clip_similarity)

    feat_names = [
        "bm25_score",
        "token_overlap",
        "title_length",
        "clip_similarity",
        "has_image",
    ]
    X_full = np.column_stack(
        [bm25_scores, token_overlaps, title_lengths, clip_sims, has_images]
    ).astype(np.float32)

    return X_full, bm25_scores, feat_names


def _score_products(
    X_full: np.ndarray,
    bm25_scores: np.ndarray,
    variant: str,
) -> np.ndarray:
    """Predict ranking scores given features and selected variant."""
    if variant == "random":
        return RandomRanker(seed=42).predict(X_full)

    if variant == "bm25_only":
        return BM25OnlyRanker().predict(X_full[:, [0]])

    if variant == "image_only":
        # Rank by clip_similarity, NaNs to lowest score
        sims = X_full[:, 3].copy()
        sims = np.nan_to_num(sims, nan=-999.0)
        return sims

    if variant == "text_only":
        X_text = np.nan_to_num(X_full[:, :3], nan=0.0)
        if "text_only" in MODELS:
            return MODELS["text_only"].predict(X_text)
        # Fallback to BM25 if model not loaded
        return bm25_scores

    if variant == "text_plus_image":
        X_all = np.nan_to_num(X_full, nan=0.0)
        if "text_plus_image" in MODELS:
            return MODELS["text_plus_image"].predict(X_all)
        return bm25_scores

    raise HTTPException(status_code=400, detail=f"Unknown model variant: {variant}")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve web user interface."""
    global HTML_CONTENT
    if not HTML_CONTENT:
        template_path = Path(__file__).parent / "templates" / "index.html"
        if template_path.exists():
            with open(template_path, encoding="utf-8") as f:
                HTML_CONTENT = f.read()
    return HTML_CONTENT


@app.get("/api/sample_queries")
def get_sample_queries() -> list[dict]:
    """Return preloaded sample queries from the ESCI test set."""
    return SAMPLE_QUERIES


@app.post("/api/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    """Rerank candidate products for a search query using the specified model."""
    start_time = time.perf_counter()
    if not req.products:
        raise HTTPException(status_code=400, detail="Products list cannot be empty")

    cfg = CFG or load_config()
    is_visual, visual_keywords = _check_visual_query(req.query, cfg.visual_lexicon.words)

    # Feature extraction
    X_full, bm25_scores, _ = _extract_features(req.query, req.products)

    # Score candidates
    scores = _score_products(X_full, bm25_scores, req.model_variant)

    # ESCI gains mapping
    gains_map = {"E": cfg.gains.E, "S": cfg.gains.S, "C": cfg.gains.C, "I": cfg.gains.I}

    # Sort descending by score
    sorted_indices = np.argsort(-scores)

    ranked_items: list[RankedProduct] = []
    y_true_sorted: list[float] = []

    for rank_idx, orig_idx in enumerate(sorted_indices, start=1):
        p = req.products[orig_idx]
        gain = gains_map.get(p.esci_label.upper(), 0.0)
        y_true_sorted.append(gain)

        ranked_items.append(
            RankedProduct(
                rank=rank_idx,
                product_id=p.product_id,
                title=p.title,
                esci_label=p.esci_label.upper(),
                score=round(float(scores[orig_idx]), 4),
                bm25_score=round(float(X_full[orig_idx, 0]), 3),
                token_overlap=round(float(X_full[orig_idx, 1]), 3),
                title_length=int(X_full[orig_idx, 2]),
                clip_similarity=(
                    round(float(X_full[orig_idx, 3]), 4)
                    if not np.isnan(X_full[orig_idx, 3])
                    else None
                ),
                has_image=int(X_full[orig_idx, 4]),
                gain=gain,
            )
        )

    # Compute NDCG@10 for this query's ranking
    ndcg_val = ndcg_at_k(np.array(y_true_sorted), k=cfg.ndcg_k)
    latency = (time.perf_counter() - start_time) * 1000.0

    return RerankResponse(
        query=req.query,
        model_variant=req.model_variant,
        ranked_products=ranked_items,
        ndcg_10=round(float(ndcg_val), 4),
        is_visual_query=is_visual,
        visual_keywords=visual_keywords,
        latency_ms=round(latency, 2),
    )


@app.post("/api/compare", response_model=CompareResponse)
def compare_rerankers(req: RerankRequest) -> CompareResponse:
    """Run text_only and text_plus_image side by side, calculating rank shifts."""
    start_time = time.perf_counter()
    if not req.products:
        raise HTTPException(status_code=400, detail="Products list cannot be empty")

    cfg = CFG or load_config()
    is_visual, visual_keywords = _check_visual_query(req.query, cfg.visual_lexicon.words)
    gains_map = {"E": cfg.gains.E, "S": cfg.gains.S, "C": cfg.gains.C, "I": cfg.gains.I}

    X_full, bm25_scores, _ = _extract_features(req.query, req.products)

    # 1. Text Only
    text_scores = _score_products(X_full, bm25_scores, "text_only")
    text_order = np.argsort(-text_scores)
    text_ranks: dict[str, int] = {}
    text_items: list[RankedProduct] = []
    y_text: list[float] = []

    for rank_idx, orig_idx in enumerate(text_order, start=1):
        p = req.products[orig_idx]
        gain = gains_map.get(p.esci_label.upper(), 0.0)
        y_text.append(gain)
        text_ranks[p.product_id] = rank_idx
        text_items.append(
            RankedProduct(
                rank=rank_idx,
                product_id=p.product_id,
                title=p.title,
                esci_label=p.esci_label.upper(),
                score=round(float(text_scores[orig_idx]), 4),
                bm25_score=round(float(X_full[orig_idx, 0]), 3),
                token_overlap=round(float(X_full[orig_idx, 1]), 3),
                title_length=int(X_full[orig_idx, 2]),
                clip_similarity=(
                    round(float(X_full[orig_idx, 3]), 4)
                    if not np.isnan(X_full[orig_idx, 3])
                    else None
                ),
                has_image=int(X_full[orig_idx, 4]),
                gain=gain,
            )
        )
    text_ndcg = ndcg_at_k(np.array(y_text), k=cfg.ndcg_k)

    # 2. Text + Image Multimodal
    mm_scores = _score_products(X_full, bm25_scores, "text_plus_image")
    mm_order = np.argsort(-mm_scores)
    mm_items: list[RankedProduct] = []
    y_mm: list[float] = []

    for rank_idx, orig_idx in enumerate(mm_order, start=1):
        p = req.products[orig_idx]
        gain = gains_map.get(p.esci_label.upper(), 0.0)
        y_mm.append(gain)
        old_rank = text_ranks.get(p.product_id, rank_idx)
        # Positive delta means moved UP (better position)
        rank_delta = old_rank - rank_idx

        mm_items.append(
            RankedProduct(
                rank=rank_idx,
                product_id=p.product_id,
                title=p.title,
                esci_label=p.esci_label.upper(),
                score=round(float(mm_scores[orig_idx]), 4),
                bm25_score=round(float(X_full[orig_idx, 0]), 3),
                token_overlap=round(float(X_full[orig_idx, 1]), 3),
                title_length=int(X_full[orig_idx, 2]),
                clip_similarity=(
                    round(float(X_full[orig_idx, 3]), 4)
                    if not np.isnan(X_full[orig_idx, 3])
                    else None
                ),
                has_image=int(X_full[orig_idx, 4]),
                gain=gain,
                rank_delta=rank_delta,
            )
        )
    mm_ndcg = ndcg_at_k(np.array(y_mm), k=cfg.ndcg_k)

    latency = (time.perf_counter() - start_time) * 1000.0

    return CompareResponse(
        query=req.query,
        is_visual_query=is_visual,
        visual_keywords=visual_keywords,
        text_only=text_items,
        text_only_ndcg=round(float(text_ndcg), 4),
        text_plus_image=mm_items,
        text_plus_image_ndcg=round(float(mm_ndcg), 4),
        delta_ndcg=round(float(mm_ndcg - text_ndcg), 4),
        latency_ms=round(latency, 2),
    )
