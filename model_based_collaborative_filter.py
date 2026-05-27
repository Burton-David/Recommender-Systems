"""Legacy SVD-from-CSV demo script.

Kept for historical continuity with the project's early commits. For new code use
``recommender_systems.svd.SVD`` (same algorithm, on the unified ``Recommender``
interface). The body of this file was previously executed at import time, which
crashed with ``FileNotFoundError`` whenever the module was imported without the
expected CSV present — now everything runs only via the ``__main__`` guard.
"""

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer


def get_recommended_items(user, ratings_path="user-item-data.csv", N=10):
    """Top-``N`` item ids for ``user``, trained from a CSV of (user_id, item_id, rating)."""
    df = pd.read_csv(ratings_path)
    df_pivot = df.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0)
    svd = TruncatedSVD(n_components=10, random_state=42)
    normalized = Normalizer().fit_transform(svd.fit_transform(df_pivot.values))
    user_row = normalized[user - 1, :]
    scores = normalized.dot(user_row)
    return df_pivot.columns[scores.argsort()[::-1][:N]]


if __name__ == "__main__":
    print(get_recommended_items(1))
