# Results

## Ablation Summary

| Variant | NDCG@10 Mean | NDCG@10 Std | Runs |
|---------|-------------|-------------|------|
| text_only | 0.6804 | 0.0000 | 3 |
| text_plus_image | 0.6798 | 0.0000 | 3 |
| bm25_only | 0.6735 | 0.0000 | 3 |
| image_only | 0.5996 | 0.0000 | 3 |
| random | 0.5833 | 0.0041 | 3 |

## Statistical Test: text_plus_image vs text_only

- **Delta (mean NDCG@10)**: -0.0006
- **95% CI**: [-0.0062, 0.0050]
- **p-value**: 0.4103
- **N queries**: 1206
- **N bootstrap samples**: 10000

## Visual vs Non-Visual Query Analysis

| Group | N Queries | NDCG@10 Mean | NDCG@10 Std |
|-------|-----------|-------------|-------------|
| visual | 116 | 0.5637 | 0.2131 |
| non-visual | 1090 | 0.5988 | 0.2089 |

## Top 15 Improved Queries (text_plus_image vs text_only)

| Query | NDCG Before | NDCG After | Delta |
|-------|------------|------------|-------|
| 18x34 pool cover | 0.4507 | 0.8564 | 0.4057 |
| neumann tlm 103 | 0.3892 | 0.7781 | 0.3889 |
| chicago bears gifts | 0.1915 | 0.5788 | 0.3873 |
| silk bodysuit long sleevetops | 0.0000 | 0.3333 | 0.3333 |
| educated book tara westover | 0.2590 | 0.5699 | 0.3109 |
| deltran battery charger 5 | 0.3401 | 0.6497 | 0.3096 |
| eero | 0.2968 | 0.5953 | 0.2985 |
| g key chain | 0.6460 | 0.9427 | 0.2967 |
| grimm season1 | 0.3355 | 0.6217 | 0.2862 |
| christmas disney long sleeve shirts for girls | 0.5127 | 0.7973 | 0.2845 |
| hydraulic oil for log splitter | 0.4832 | 0.7528 | 0.2697 |
| modern warfare cod | 0.6131 | 0.8772 | 0.2641 |
| huda beauty 420 | 0.0797 | 0.3394 | 0.2597 |
| waterproof hking coat | 0.4814 | 0.7343 | 0.2529 |
| vans sweatshirt | 0.4616 | 0.7117 | 0.2501 |

## Top 15 Degraded Queries (text_plus_image vs text_only)

| Query | NDCG Before | NDCG After | Delta |
|-------|------------|------------|-------|
| marvelous mrs maisel | 0.8289 | 0.2832 | -0.5458 |
| 112 aio cooler without fans | 0.6954 | 0.1979 | -0.4974 |
| inside christmas projector | 0.8878 | 0.4406 | -0.4471 |
| iphone 11 pri | 0.4376 | 0.0274 | -0.4102 |
| instant pot 8qt | 0.8420 | 0.4634 | -0.3786 |
| kids toys $5 or less | 1.0000 | 0.6677 | -0.3323 |
| delta wall faucet spout with shower control | 0.8439 | 0.5170 | -0.3269 |
| silver hair gema | 0.8045 | 0.4913 | -0.3132 |
| gem frost | 0.9340 | 0.6236 | -0.3104 |
| gel polish remover for toes | 0.6471 | 0.3455 | -0.3016 |
| black and white dishes farmhouse | 0.5494 | 0.2502 | -0.2992 |
| female bounty hunter costume | 0.2967 | 0.0000 | -0.2967 |
| boho dress casual | 0.8803 | 0.6046 | -0.2757 |
| swarovski bezel bracelet | 0.9146 | 0.6489 | -0.2657 |
| blue suit for baby boy | 1.0000 | 0.7411 | -0.2589 |

