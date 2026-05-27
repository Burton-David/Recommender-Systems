# Recommender Systems Library

[![CI](https://github.com/Burton-David/Recommender-Systems/actions/workflows/ci.yml/badge.svg)](https://github.com/Burton-David/Recommender-Systems/actions/workflows/ci.yml)
[![Docs](https://github.com/Burton-David/Recommender-Systems/actions/workflows/docs.yml/badge.svg)](https://burton-david.github.io/Recommender-Systems/)
[![codecov](https://codecov.io/gh/Burton-David/Recommender-Systems/branch/main/graph/badge.svg)](https://codecov.io/gh/Burton-David/Recommender-Systems)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

**This library contains a variety of different recommender systems implemented in Python.**

## Benchmarks

Top-10 evaluation on MovieLens 100k (80/20 seeded split). Reproduce with
`pip install -e ".[dev,benchmarks]" && python scripts/benchmark.py`.

![MovieLens 100k benchmark](benchmarks/results.png)

|             | precision@10 | recall@10 |  MAP@10 | NDCG@10 | coverage@10 |
|:------------|-------------:|----------:|--------:|--------:|------------:|
| MostPopular |       0.1863 |    0.1191 |  0.1104 |  0.2141 |      0.0315 |
| MeanRating  |       0.0490 |    0.0194 |  0.0140 |  0.0428 |      0.0161 |
| ItemKNN     |       0.3188 |    0.2010 |  0.2486 |  0.3786 |      0.2866 |
| UserKNN     |       0.3175 |    0.2123 |  0.2503 |  0.3881 |      0.2134 |
| SVD         |       0.3016 |    0.2134 |  0.2283 |  0.3675 |      0.2717 |

See [`benchmarks/results.md`](benchmarks/results.md) for the same table generated
from a fresh run.

1. user_based_cosine_similarity.py
2. item_based_dot_product.py
3. demographic_based_mean_rating.py
4. context_aware_mean_rating.py
5. hybrid_context_user.py
6. content_based_binary_filter.py
7. content_based_cosine_similarity.py
8. content_based_word_counts.py
9. content_based_word_embeddings.py
10. demographic_based_filtering.py

## Using the Library
To use any of the included files, simply import the file and call the get_recommended_items() function with the appropriate arguments.

For example, to use the user-based collaborative filtering recommender using cosine similarity:
```
import user_based_cosine_similarity

recommendations = user_based_cosine_similarity.get_recommended_items(df, 1)
```
The **df** parameter should be a pandas DataFrame containing ratings data with columns for 'user_id', 'item_id', and 'rating'. The second parameter is the ID of the user or item for which you want to get recommendations.

Each file has its own specific requirements for the data and parameters, so be sure to read the docstrings for more information.

### File Descriptions
user_based_cosine_similarity.py
This file contains a user-based collaborative filtering recommender that uses cosine similarity to compute the similarity

### item_based_dot_product.py
This file contains an item-based collaborative filtering recommender that uses the dot product of ratings to compute the similarity between items. Given an item's ID, the function returns the top N recommended items for that item based on the dot product of their ratings with other items.

### demographic_based_mean_rating.py
This file contains a demographic-based recommender that uses the age and gender of users to recommend items. Given a user's ID, the function returns the top N recommended items for that user based on the average rating of those items by users of the same age and gender.

### context_aware_mean_rating.py
This file contains a context-aware recommender that uses the context in which the recommendations will be used to recommend items. Given a context, the function returns the top N recommended items for that context based on the average rating of those items in that context.

### hybrid_context_user.py
This file contains a hybrid recommender that combines both context-aware and user-based collaborative filtering. Given a user's ID and a context, the function returns the top N recommended items for that user and context by combining the recommendations from both the context-aware and user-based collaborative filtering approaches.

### content_based_binary_filter.py
This file contains a content-based recommender that uses a binary filter to recommend items. Given a list of keywords, the function returns the top N recommended items that contain those keywords in their description.

### content_based_cosine_similarity.py
This file contains a content-based recommender that uses cosine similarity to recommend items. Given a list of keywords, the function returns the top N recommended items that are most similar to the keywords based on the cosine similarity of their descriptions.

### content_based_word_counts.py
This file contains a content-based recommender that uses word counts to recommend items. Given a list of keywords, the function returns the top N recommended items that contain the most occurrences of those keywords in their description.

### content_based_word_embeddings.py
This file contains a content-based recommender that uses word embeddings to recommend items. Given a list of keywords, the function returns the top N recommended items that are most similar to the keywords based on the cosine similarity of their word embeddings.

## demographic_based_filtering.py
This file contains a demographic-based recommender that uses the age and gender of users to recommend items. Given a user's ID, the function returns the top N recommended items for that user based on the average rating of those items by users of the same age and gender.
