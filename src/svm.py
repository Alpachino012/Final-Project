"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

No QP / SMO solver. Pegasos minimises the regularised hinge-loss primal by
stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate eta_t:

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i      (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                      (only regularise)

The bias is folded into w by adding a constant 1 feature to every x (as in
the project statement), so b gets the same update as the other weights.
With batch_size > 1 the hinge part is averaged over the mini-batch
(Shalev-Shwartz et al., 2011, Sec. 2.3); batch_size = 1 is plain Pegasos.

Non-linear extension: RANDOM FOURIER FEATURES (Rahimi & Recht, 2007).
We map X -> phi(X) whose inner products approximate the RBF kernel
exp(-gamma ||x - x'||^2) and then run the same LINEAR Pegasos on phi(X).
Kernelised Pegasos would need a kernel row against all ~40k training
points at every step and keep ~40k coefficients; random features keep the
cost linear in n and reuse the linear solver unchanged.

scikit-learn's SVC is a BASELINE only — not used inside these classes.
Features must be STANDARDIZED first (see data_prep.standardize).
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,       # regularisation strength (C = 1/(n*lambda))
        n_epochs: int = 10,          # passes over the training data
        batch_size: int = 1,         # 1 = stochastic Pegasos, n = full batch
        lr_schedule: str = "pegasos",  # "pegasos" | "constant" | "invsqrt"
        eta0: float = 0.01,          # base rate for "constant" / "invsqrt"
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.eta0 = eta0
        self.random_state = random_state
        self.w = None              # weight vector
        self.b = 0.0               # bias

    def _to_pm1(self, y):
        """Map the 2 class labels (e.g. {0,1}) to {-1, +1}. Store the mapping."""
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("PegasosSVM is a binary classifier")
        return np.where(y == self.classes_[1], 1.0, -1.0)

    def _learning_rate(self, t):
        if self.lr_schedule == "pegasos":
            return 1.0 / (self.lambda_ * t)
        if self.lr_schedule == "constant":
            return self.eta0
        if self.lr_schedule == "invsqrt":
            return self.eta0 / np.sqrt(t)
        raise ValueError("unknown lr_schedule: " + self.lr_schedule)

    def _objective(self, X_aug, y_pm, w):
        """J(w) = lambda/2 ||w||^2 + mean hinge loss."""
        hinge = np.maximum(0.0, 1.0 - y_pm * (X_aug @ w))
        return 0.5 * self.lambda_ * np.dot(w, w) + hinge.mean()

    def fit(self, X, y):
        """
        Run Pegasos. Each step takes one mini-batch, uses eta_t from the
        schedule and applies the update above. The objective J is recorded
        after every epoch in self.objective_history_.
        Return self.
        """
        X = np.asarray(X, dtype=float)
        y_pm = self._to_pm1(np.asarray(y))
        n, d = X.shape
        X_aug = np.hstack([X, np.ones((n, 1))])   # constant 1 -> bias
        w = np.zeros(d + 1)
        rng = np.random.default_rng(self.random_state)

        self.objective_history_ = [self._objective(X_aug, y_pm, w)]
        t = 0
        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = order[start:start + self.batch_size]
                t += 1
                eta = self._learning_rate(t)
                X_b = X_aug[batch]
                y_b = y_pm[batch]
                active = y_b * (X_b @ w) < 1          # inside the margin
                w = (1.0 - eta * self.lambda_) * w
                if np.any(active):
                    w = w + (eta / len(batch)) * (y_b[active] @ X_b[active])
            self.objective_history_.append(self._objective(X_aug, y_pm, w))

        self.w = w[:-1]
        self.b = float(w[-1])
        return self

    def decision_function(self, X):
        """Return the signed margin  w . x + b  for each row."""
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X):
        """Return class labels (map sign of decision_function back to our labels)."""
        positive = self.decision_function(X) > 0
        return self.classes_[positive.astype(int)]

    # --- margin and support vectors -----------------------------------------
    def weight_norm(self) -> float:
        """||w|| (without the bias)."""
        return float(np.linalg.norm(self.w))

    def margin_width(self) -> float:
        """Geometric width of the margin, 2 / ||w||."""
        return 2.0 / self.weight_norm()

    def support_vector_indices(self, X, y):
        """Training points on or inside the margin: y_i (w . x_i + b) <= 1."""
        y_pm = np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0)
        return np.where(y_pm * self.decision_function(X) <= 1.0)[0]


class RandomFourierFeatures:
    """
    phi(x) = sqrt(2/D) * cos(W^T x + c), with W ~ N(0, 2*gamma*I) and
    c ~ Uniform(0, 2*pi). Then phi(x) . phi(x') ~ exp(-gamma ||x - x'||^2).
    """
    def __init__(self, n_components: int = 300, gamma: float = 0.03,
                 random_state: int = 42):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        self.W = rng.normal(0.0, np.sqrt(2.0 * self.gamma),
                            size=(n_features, self.n_components))
        self.c = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.W + self.c)
