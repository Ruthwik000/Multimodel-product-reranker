# Interview Notes & Project Walkthrough

## 1. 30-Second Elevator Pitch

"Most e-commerce product search engines rank items primarily using textual keyword matching (BM25 or dense retrieval), but consumers make purchasing decisions heavily influenced by visual attributes like style, color, pattern, and silhouette. In this project, we built a multimodal reranker using **LightGBM LambdaRank** on the Amazon ESCI benchmark combined with precomputed **CLIP image embeddings** from the SQID dataset. We demonstrated an end-to-end reproducible research workflow on a laptop-scale CPU setup, evaluated with query-grouped **NDCG@10** and rigorous **paired bootstrap significance testing** (95% CI) to quantify exactly when and why visual features improve search ranking."

---

## 2. 2-Minute Technical Walkthrough

### The Problem
Text search often fails when users type ambiguous or visually descriptive queries (e.g., "floral summer dress" or "vintage leather bag"). Traditional text retrieval either over-matches irrelevant text tokens or fails to capture visual nuance.

### Pipeline Architecture
1. **Data Ingestion & Filtering**: We ingest the Amazon ESCI dataset (Task 1 / US locale, reduced small version) and attach SQID CLIP query and product image embeddings.
2. **Leakage-Free Splitting**: We enforce strict query-level isolation using `GroupShuffleSplit` on `query_id`. A query and its candidate products never span across train, validation, and test splits.
3. **Feature Engineering**:
   - **Text features**: Candidate-level BM25 score, query-product token overlap ratio, and product title character length.
   - **Image features**: Cosine similarity between SQID CLIP query embedding and product image embedding, accompanied by a binary indicator `has_image` (flagging missing images, assigned NaN similarity).
4. **Learning-to-Rank Model**: LightGBM LambdaRank with NDCG objective and configurable ESCI gain weights (`Exact=1.0`, `Substitute=0.1`, `Complement=0.01`, `Irrelevant=0.0`). We handle integer mapping and `label_gain` configuration for LambdaRank.
5. **Ablation & Statistical Validation**:
   - Evaluated 5 variants across 3 seeds: `random`, `bm25_only`, `text_only`, `image_only`, and `text_plus_image`.
   - Paired bootstrap test (10,000 resamples) computing the 95% confidence interval of $\Delta \text{NDCG@10} = \text{NDCG}_{\text{text+image}} - \text{NDCG}_{\text{text}}$.
6. **Error Analysis**: We classify queries into *visual* vs. *non-visual* using a curated color/material/pattern lexicon to uncover where image signals provide the strongest lift versus where they introduce noise.

---

## 3. Core Architectural & ML Questions

### Q1: Why use NDCG@10 rather than Accuracy, Precision@k, or MAP?
- **Graded Relevance**: ESCI labels are inherently multi-grade (`Exact`, `Substitute`, `Complement`, `Irrelevant`). Precision@k and MAP treat relevance as binary (0/1), discarding the critical business nuance that a *substitute* item (e.g., an alternative brand phone case) is moderately useful, whereas an *irrelevant* item is completely detrimental.
- **Position Discounting**: Users pay disproportionate attention to the top results (positions 1–3). NDCG applies a logarithmic discount ($\log_2(i+1)$), penalizing high-ranked irrelevant items heavily.
- **Query-Level Invariance**: Normalization by the Ideal DCG (IDCG) ensures that queries with differing numbers of relevant items contribute equally to the final metric, avoiding domination by queries with large candidate sets.

### Q2: Why split strictly by `query_id` instead of row-level random splitting?
- **Data Leakage & Memorization**: If query-product pairs from the same query appear in both training and test sets, the model can simply memorize query-specific candidate distributions or title vocabulary rather than learning a generalizable ranking function.
- **Real-World Simulation**: In production search, the system must rank candidate products for both seen and unseen user queries. Query-level grouping simulates out-of-sample generalization.
- **Code Assertion**: We explicitly assert `set(train_queries).isdisjoint(test_queries)` in the pipeline to prevent silent regression.

### Q3: When do product images *hurt* ranking performance?
1. **Ambiguous or Non-Visual Queries**: For queries where physical appearance is irrelevant (e.g., "AA batteries", "USB-C cable 6ft 100W", "replacement screws M4"), CLIP embedding similarity can introduce misleading semantic drift.
2. **Low-Quality or Misleading Product Images**: Multi-pack items shown with a single item image, products shown with unrelated packaging, or stylized promotional banners can distort visual similarity.
3. **Missing Image Penalty**: Missing product embeddings must be handled carefully. If missingness correlates with product quality or catalog completeness, improper imputation (or arbitrary zeros instead of tree-handled NaNs) biases ranking scores.

### Q4: How would this multimodal reranking architecture serve at scale in production?
- **Two-Stage (or Three-Stage) Retrieval Funnel**:
  1. *Stage 1 (Retrieval)*: Approximate Nearest Neighbor (ANN) index (e.g., HNSW or Milvus) on text + BM25 retrieves top 500 candidates in <10ms.
  2. *Stage 2 (LightGBM LambdaRank Reranking)*: Precomputed offline CLIP embeddings for product catalog stored in a low-latency key-value cache (e.g., Redis or Aerospike). The query CLIP embedding is computed in real time (~5ms via ONNX Runtime / TensorRT on GPU or optimized CPU). Fast tabular feature extraction and LightGBM inference rerank top 500 down to top 50 in <15ms.
  3. *Stage 3 (Cross-Attention or Heavy Transformer, Optional)*: Late-stage reranking on top 20 candidates.
- **Embedding Cache Invalidation**: Catalog product embeddings are generated asynchronously during catalog ingestion/updates, meaning zero online image encoding overhead.
- **Fallback Resilience**: When a product image is unavailable or inference times out, the LambdaRank tree handles `NaN` image features natively, falling back gracefully to text BM25 and token overlap signals.
