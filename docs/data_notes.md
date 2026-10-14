# Data Notes & Dataset Observations

Observations recorded from the real Amazon ESCI dataset and SQID CLIP embeddings.

## 1. Raw Dataset Overview

| Dataset / Table | Source | Row Count | Key Columns |
|---|---|---:|---|
| **ESCI Examples** | Amazon ESCI | 2,621,288 | `example_id`, `query`, `query_id`, `product_id`, `product_locale`, `esci_label`, `small_version`, `large_version`, `split` |
| **ESCI Products** | Amazon ESCI | 1,814,924 | `product_id`, `product_title`, `product_description`, `product_brand`, `product_color`, `product_locale` |
| **SQID Product Features** | Crossing-Minds (HF) | 164,900 | `product_id`, `clip_text_features`, `clip_image_features` (768-dim numpy float32) |
| **SQID Query Features** | Crossing-Minds (HF) | 8,956 | `query_id`, `clip_text_features` (768-dim numpy float32) |

---

## 2. Filtering & Sampling Funnel

1. **Initial ESCI Examples:** 2,621,288 rows across all international locales.
2. **Locale Filter (`product_locale == 'us'`):** Reduced to 1,818,825 rows. Note that the column name in raw data is `product_locale` (not `query_locale` or `locale`).
3. **Reduced Version Filter (`small_version == 1`):** Filtered to 601,354 query-product pairs across 29,844 distinct queries.
4. **Configured Sampling (Task 4):** Seeded query-level sample of 4,000 queries (3,000 train + 1,000 test) preserving all candidate products per query, resulting in **80,217 rows** (average ~20.05 candidate products per query).

---

## 3. Label Distribution in Processed Subset

| Label | Meaning | Gain (YAML) | Count | Percentage |
|:---:|:---|:---:|---:|---:|
| **E** | Exact match | 1.00 | 35,057 | 43.70% |
| **S** | Substitute product | 0.10 | 27,976 | 34.88% |
| **I** | Irrelevant | 0.00 | 13,484 | 16.81% |
| **C** | Complementary item | 0.01 | 3,700 | 4.61% |
| **Total** | | | **80,217** | **100.00%** |

---

## 4. Embedding Alignment & Missingness

- **Product Image Embedding Coverage:** 31,397 / 80,217 pairs (39.14%) have valid CLIP image embeddings in SQID.
- **Missing Product Embedding Rate:** 60.86% (48,820 rows). Products without embeddings receive `has_image = 0` and `NaN` cosine similarity.
- **Embedding Dimensions:** Real SQID CLIP embeddings have dimension **768** (OpenAI CLIP ViT-L/14).
