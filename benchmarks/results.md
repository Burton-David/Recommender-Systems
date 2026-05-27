# MovieLens 100k — top-10 evaluation

Reproduce with `python scripts/benchmark.py` (seed = 0, 80/20 split).

|             |   precision@10 |   recall@10 |   MAP@10 |   NDCG@10 |   coverage@10 |
|:------------|---------------:|------------:|---------:|----------:|--------------:|
| MostPopular |         0.1863 |      0.1191 |   0.1104 |    0.2141 |        0.0315 |
| MeanRating  |         0.049  |      0.0194 |   0.014  |    0.0428 |        0.0161 |
| ItemKNN     |         0.3188 |      0.201  |   0.2486 |    0.3786 |        0.2866 |
| UserKNN     |         0.3175 |      0.2123 |   0.2503 |    0.3881 |        0.2134 |
| SVD         |         0.3016 |      0.2134 |   0.2283 |    0.3675 |        0.2717 |
