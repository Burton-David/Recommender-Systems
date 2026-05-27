# goodbooks-10k — top-10 evaluation

Reproduce with `python -m scripts.benchmark_goodbooks` (seed = 0, 80/20 split).
Subsampled to 2500 users (~279k interactions) so the dense user-user similarity needed by UserKNN fits in memory.


|             |   precision@10 |   recall@10 |   MAP@10 |   NDCG@10 |   coverage@10 |
|:------------|---------------:|------------:|---------:|----------:|--------------:|
| MostPopular |         0.0985 |      0.0434 |   0.0482 |    0.108  |        0.0035 |
| MeanRating  |         0.0042 |      0.0019 |   0.0011 |    0.004  |        0.0014 |
| ItemKNN     |         0.3256 |      0.1511 |   0.2314 |    0.3719 |        0.3413 |
| UserKNN     |         0.2414 |      0.1113 |   0.1552 |    0.2766 |        0.1286 |
| SVD         |         0.2714 |      0.1229 |   0.184  |    0.3142 |        0.0739 |
