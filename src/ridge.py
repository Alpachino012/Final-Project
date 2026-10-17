"""
Ridge regression (bonus part).

normal equations:
    w_hat = (X^T X + lambda * I)^(-1) X^T y

A column of ones is added to X for the intercept, and the intercept is
not penalised (its diagonal entry in I is 0), like scikit-learn's Ridge.
lambda = 0 gives ordinary least squares, but on our features X^T X is
singular in exact arithmetic: the three category one-hot columns always
add up to 1, the same as the intercept column, so the least-squares
solution is not unique. Any lambda > 0 makes the matrix invertible.

We solve the linear system with np.linalg.solve instead of forming the
inverse explicitly: same w_hat, but numerically safer.
Features should be STANDARDIZED first (see data_prep.standardize), so that
lambda shrinks all coefficients on the same scale.
"""

from __future__ import annotations

import numpy as np


class RidgeRegression:
    def __init__(self, lambda_: float = 1.0):
        self.lambda_ = lambda_
        self.coef_ = None          # one weight per feature
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        X_aug = np.hstack([np.ones((n, 1)), X])     # first column = intercept

        penalty = self.lambda_ * np.eye(d + 1)
        penalty[0, 0] = 0.0                          # do not shrink the intercept

        A = X_aug.T @ X_aug + penalty
        w = np.linalg.solve(A, X_aug.T @ y)
        self.intercept_ = float(w[0])
        self.coef_ = w[1:]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
