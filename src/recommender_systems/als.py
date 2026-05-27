"""Alternating Least Squares for implicit-feedback matrix factorization."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd

from recommender_systems.base import _MatrixBackedRecommender
from recommender_systems.data import build_user_item_matrix

__all__ = ["ALS"]


class ALS(_MatrixBackedRecommender):
    """Implicit-feedback matrix factorization via alternating least squares.

    Follows Hu, Koren & Volinsky (2008): binary preferences ``p_ui = 1 if r_ui > 0``
    with confidence ``c_ui = 1 + alpha * r_ui``, then alternate closed-form solves
    for the user factors ``X`` and item factors ``Y`` of the regularized weighted
    least-squares loss.

    Same algorithm family as :class:`recommender_systems.bpr.BPR` but a different
    optimization story — closed-form alternating solves instead of SGD on a sigmoid
    margin — so it tends to converge in many fewer epochs.

    Parameters
    ----------
    n_factors
        Latent factor dimensionality.
    epochs
        Full passes over the (X, Y) update cycle.
    regularization
        L2 regularization strength applied to the factor solves.
    alpha
        Confidence scaling: a rating of ``r`` contributes weight ``1 + alpha * r``.
        Higher ``alpha`` makes the model trust observed ratings more.
    random_state
        Seed for factor initialization.
    """

    def __init__(
        self,
        n_factors: int = 32,
        epochs: int = 15,
        regularization: float = 0.01,
        alpha: float = 40.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.epochs = epochs
        self.regularization = regularization
        self.alpha = alpha
        self.random_state = random_state
        self._predicted: np.ndarray = np.empty((0, 0))

    def fit(self, ratings: pd.DataFrame) -> ALS:
        self._matrix = build_user_item_matrix(ratings, fill_value=0.0)
        ratings_mat = self._matrix.to_numpy()
        preferences = (ratings_mat > 0).astype(np.float64)
        confidence = 1.0 + self.alpha * ratings_mat
        n_users, n_items = ratings_mat.shape

        rng = np.random.default_rng(self.random_state)
        user_factors = rng.normal(0.0, 0.01, size=(n_users, self.n_factors))
        item_factors = rng.normal(0.0, 0.01, size=(n_items, self.n_factors))
        reg_eye = self.regularization * np.eye(self.n_factors)

        for _ in range(self.epochs):
            user_factors = self._solve_side(item_factors, confidence, preferences, reg_eye)
            item_factors = self._solve_side(user_factors, confidence.T, preferences.T, reg_eye)

        self._predicted = np.asarray(user_factors @ item_factors.T)
        return self

    @staticmethod
    def _solve_side(
        other: np.ndarray, confidence: np.ndarray, preferences: np.ndarray, reg_eye: np.ndarray
    ) -> np.ndarray:
        """Closed-form solve for one factor matrix given the other side fixed.

        Solving the side with rows ``u`` means: for each ``u`` find ``x_u`` minimizing
        ``sum_i c_{u,i} (p_{u,i} - x_u . y_i)^2 + lambda ||x_u||^2``.
        """
        gram = other.T @ other  # (f, f)
        target = np.empty((confidence.shape[0], other.shape[1]))
        for row in range(confidence.shape[0]):
            cu_minus_1 = confidence[row] - 1.0
            a = gram + other.T @ (cu_minus_1[:, None] * other) + reg_eye
            b = other.T @ (confidence[row] * preferences[row])
            target[row] = np.linalg.solve(a, b)
        return target

    def _user_scores(self, user_id: Hashable) -> np.ndarray:
        return np.asarray(self._predicted[self._matrix.index.get_loc(user_id)])
