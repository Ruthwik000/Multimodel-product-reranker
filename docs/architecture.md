# Architecture

## Pipeline Overview

```mermaid
flowchart LR
    A[Raw ESCI + SQID] --> B[Loaders]
    B --> C[Filter: US locale, reduced]
    C --> D[Sample queries]
    D --> E[Join image embeddings]
    E --> F[Query-level split]
    F --> G[Feature builder]
    G --> H[LightGBM LambdaRank]
    H --> I[NDCG@10]
    I --> J[Ablations]
    J --> K[Error analysis]
    K --> L[Reports]
```

## Components

- **Loaders** (`src/mmrerank/data/loaders.py`): Read raw ESCI and SQID parquet files.
- **Sampling** (`src/mmrerank/data/sampling.py`): Seeded query sampling preserving all candidate products per query.
- **Join** (`src/mmrerank/data/join.py`): Attach SQID CLIP embeddings to products, flag missing.
- **Splits** (`src/mmrerank/data/splits.py`): Query-grouped train/val/test split with leakage assertions.
- **Features** (`src/mmrerank/features/`): BM25 score, token overlap, title length (text); CLIP cosine similarity, has_image flag (image).
- **Models** (`src/mmrerank/models/`): LGBMRanker wrapper with LambdaRank and baselines (random, BM25-only).
- **Evaluation** (`src/mmrerank/evaluation/`): NDCG@k, ablation framework, paired bootstrap CI, error analysis.
