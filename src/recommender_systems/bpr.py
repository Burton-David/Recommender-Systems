"""Bayesian Personalized Ranking for implicit feedback."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd

from recommender_systems.base import _MatrixBackedRecommender
from recommender_systems.data import build_user_item_matrix

__all__ = ["BPR"]


class BPR(_MatrixBackedRecommender):
    """Bayesian Personalized Ranking — learns item embeddings from implicit feedback.

    Every observed (user, item) interaction is treated as a positive signal; random
    unobserved items are sampled as negatives. The objective is to score each positive
    higher than its sampled negative — implemented as SGD on the sigmoid-margin loss
    with L2 regularization. The rating values themselves are ignored; only presence
    of an interaction matters.

    Parameters
    ----------
    n_factors
        Dimensionality of the user and item embeddings.
    epochs
        Number of full passes over the observed interactions.
    learning_rate
        SGD step size.
    reg
        L2 regularization strength applied to user and item factors.
    random_state
        Seed for initialization and negative sampling.
    """

    def __init__(
        self,
        n_factors: int = 32,
        epochs: int = 20,
        learning_rate: float = 0.05,
        reg: float = 0.01,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg = reg
        self.random_state = random_state
        self._user_factors: np.ndarray = np.empty((0, 0))
        self._item_factors: np.ndarray = np.empty((0, 0))

    def fit(self, ratings: pd.DataFrame) -> BPR:
        self._matrix = build_user_item_matrix(ratings, fill_value=0.0)
        observed = self._matrix.to_numpy() > 0
        n_users, n_items = observed.shape

        rng = np.random.default_rng(self.random_state)
        self._user_factors = rng.normal(0.0, 0.01, size=(n_users, self.n_factors))
        self._item_factors = rng.normal(0.0, 0.01, size=(n_items, self.n_factors))

        # Restrict training to users that have at least one unobserved item —
        # without that guarantee the negative-resample loop below never terminates.
        has_negatives = observed.sum(axis=1) < n_items
        positives = np.argwhere(observed & has_negatives[:, None]).astype(np.int64)

        try:
            from recommender_systems import _kernels  # type: ignore[attr-defined]
        except ImportError:
            self._fit_python(observed, positives, n_items, rng)
        else:
            _kernels.bpr_train(
                self._user_factors,
                self._item_factors,
                positives,
                observed.reshape(-1),
                n_items,
                self.epochs,
                self.learning_rate,
                self.reg,
                int(self.random_state if self.random_state is not None else 0),
            )
        return self

    def _fit_python(
        self,
        observed: np.ndarray,
        positives: np.ndarray,
        n_items: int,
        rng: np.random.Generator,
    ) -> None:
        """Pure-Python training loop, kept as a fallback for environments where
        the compiled ``_kernels`` extension isn't available."""
        for _ in range(self.epochs):
            order = rng.permutation(len(positives))
            negatives = rng.integers(0, n_items, size=len(positives))
            for idx, neg in zip(order, negatives, strict=True):
                u, i = positives[idx]
                j = int(neg)
                while observed[u, j]:
                    j = int(rng.integers(0, n_items))
                self._step(int(u), int(i), j)

    def _step(self, u: int, i: int, j: int) -> None:
        u_vec = self._user_factors[u]
        i_vec = self._item_factors[i]
        j_vec = self._item_factors[j]
        margin = u_vec @ (i_vec - j_vec)
        sig = 1.0 / (1.0 + np.exp(margin))  # sigmoid(-margin); saturates safely at the tails
        lr = self.learning_rate
        self._user_factors[u] += lr * (sig * (i_vec - j_vec) - self.reg * u_vec)
        self._item_factors[i] += lr * (sig * u_vec - self.reg * i_vec)
        self._item_factors[j] += lr * (-sig * u_vec - self.reg * j_vec)

    def _user_scores(self, user_id: Hashable) -> np.ndarray:
        u_idx = self._matrix.index.get_loc(user_id)
        return np.asarray(self._user_factors[u_idx] @ self._item_factors.T)
