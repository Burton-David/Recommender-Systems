# goodbooks-10k — top-10 evaluation

Reproduce with `python -m scripts.benchmark_goodbooks` (seed = 20260527, 80/20 split).
Subsampled to 2500 users (~281k interactions) so the dense user-user similarity needed by UserKNN fits in memory.


|             |   precision@10 |   recall@10 |   MAP@10 |   NDCG@10 |   coverage@10 |
|:------------|---------------:|------------:|---------:|----------:|--------------:|
| MostPopular |         0.0979 |      0.0437 |   0.0517 |    0.1109 |        0.0036 |
| MeanRating  |         0.0036 |      0.0016 |   0.0014 |    0.0040 |        0.0014 |
| ItemKNN     |         0.3355 |      0.1534 |   0.2425 |    0.3841 |        0.3589 |
| UserKNN     |         0.2370 |      0.1085 |   0.1539 |    0.2729 |        0.1423 |
| SVD         |         0.2756 |      0.1241 |   0.1858 |    0.3173 |        0.0759 |
