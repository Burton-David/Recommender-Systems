"""Two-tower neural collaborative filtering (PyTorch).

Optional — requires the ``neural`` extra:

    pip install 'recommender-systems[neural]'
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "TwoTowerCF requires torch. Install with: pip install 'recommender-systems[neural]'"
    ) from exc

from recommender_systems.base import _MatrixBackedRecommender
from recommender_systems.data import build_user_item_matrix

__all__ = ["TwoTowerCF"]


class _TwoTowerNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int) -> None:
        super().__init__()
        self.user_embed = nn.Embedding(n_users, n_factors)
        self.item_embed = nn.Embedding(n_items, n_factors)
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        scores: torch.Tensor = (self.user_embed(users) * self.item_embed(items)).sum(dim=-1)
        return scores


class TwoTowerCF(_MatrixBackedRecommender):
    """Two-tower neural CF trained with a BPR-style ranking loss.

    Each user and item is a learned embedding; the score for a (user, item) pair is
    the dot product of their vectors. Training samples observed (positive) and random
    unobserved (negative) items per user and maximizes the sigmoid margin between
    them — the same objective as :class:`recommender_systems.bpr.BPR`, lifted onto
    PyTorch for autograd, batched SGD, and easy extension (e.g. adding side
    information into either tower).

    Parameters
    ----------
    n_factors
        Embedding dimensionality.
    epochs
        Full passes over the observed interactions.
    learning_rate
        Adam optimizer step size.
    batch_size
        Triples per gradient step.
    random_state
        Seed for embedding init and negative sampling.
    """

    def __init__(
        self,
        n_factors: int = 32,
        epochs: int = 20,
        learning_rate: float = 0.01,
        batch_size: int = 256,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self._predicted: np.ndarray = np.empty((0, 0))

    def fit(self, ratings: pd.DataFrame) -> TwoTowerCF:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        self._matrix = build_user_item_matrix(ratings, fill_value=0.0)
        observed = self._matrix.to_numpy() > 0
        n_users, n_items = observed.shape

        # Same termination guard as BPR: skip users with nothing to sample negatives from.
        has_negatives = observed.sum(axis=1) < n_items
        positives = np.argwhere(observed & has_negatives[:, None])
        if len(positives) == 0:
            self._predicted = np.zeros((n_users, n_items))
            return self

        rng = np.random.default_rng(self.random_state)
        model = _TwoTowerNet(n_users, n_items, self.n_factors)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        users_arr = positives[:, 0]
        items_arr = positives[:, 1]
        for _ in range(self.epochs):
            order = rng.permutation(len(positives))
            for start in range(0, len(positives), self.batch_size):
                batch = order[start : start + self.batch_size]
                u = users_arr[batch]
                i = items_arr[batch]
                j = rng.integers(0, n_items, size=len(batch))
                # Resample any negatives the user has actually interacted with.
                mask = observed[u, j]
                while mask.any():
                    j[mask] = rng.integers(0, n_items, size=int(mask.sum()))
                    mask = observed[u, j]
                pos = model(torch.from_numpy(u), torch.from_numpy(i))
                neg = model(torch.from_numpy(u), torch.from_numpy(j))
                loss = -torch.nn.functional.logsigmoid(pos - neg).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            scores = model.user_embed.weight @ model.item_embed.weight.T
            self._predicted = scores.numpy()
        return self

    def _user_scores(self, user_id: Hashable) -> np.ndarray:
        return np.asarray(self._predicted[self._matrix.index.get_loc(user_id)])
