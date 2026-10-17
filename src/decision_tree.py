"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

Suggested design:
  - a small Node (feature index, threshold, left, right, value/leaf_value),
  - recursive _grow with a best-split search over features & thresholds,
  - impurity functions (gini / entropy / mse) and the impurity-decrease
    objective used to pick splits,
  - hyperparameters that control overfitting (below).

Design implemented by our team:
  - a small Node (feature index, threshold, left, right, value),
  - recursive _grow with a best-split search over features & thresholds,
  - impurity functions (gini / entropy / mse) and the impurity-decrease
    objective used to pick splits,
  - hyperparameters that control overfitting (max_depth, min_samples_split,
    min_samples_leaf, min_impurity_decrease).

Formulas (p_k = share of class k in a node):
    Gini     G = 1 - sum_k p_k^2
    Entropy  H = - sum_k p_k log2 p_k
    MSE      I = 1/N sum_i (y_i - mean(y))^2      (variance of the node)
    Gain     IG = I(parent) - sum_c N_c/N * I(child c)
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree. Fill in the fields you need."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf prediction (class label/proba or mean)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        if task == "regression" and criterion != "mse":
            raise ValueError("regression trees use criterion='mse'")
        if task == "classification" and criterion not in ("gini", "entropy"):
            raise ValueError("classification trees use 'gini' or 'entropy'")
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression)."""
        if self.task == "regression":
            return float(np.var(y))
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        return float(self._class_impurity(counts[None, :], np.array([len(y)]))[0])

    def _best_split(self, X, y):
        """
        Search features & candidate thresholds; return the split that
        maximizes impurity decrease (feature, threshold, decrease), or
        None if no valid split improves things.
        """
        n, n_features = X.shape
        parent_impurity = self._impurity(y)
        n_left = np.arange(1, n)              # size of the left child
        n_right = n - n_left
        best = None
        best_gain = 0.0                       # a split must strictly help

        for feature in range(n_features):
            order = np.argsort(X[:, feature], kind="stable")
            x_sorted = X[order, feature]
            y_sorted = y[order]

            if self.task == "regression":
                # variance = mean(y^2) - mean(y)^2 for each child
                left_sum = np.cumsum(y_sorted)[:-1]
                left_sq = np.cumsum(y_sorted ** 2)[:-1]
                right_sum = y_sorted.sum() - left_sum
                right_sq = (y_sorted ** 2).sum() - left_sq
                left_imp = left_sq / n_left - (left_sum / n_left) ** 2
                right_imp = right_sq / n_right - (right_sum / n_right) ** 2
                left_imp = np.maximum(left_imp, 0.0)     # tiny negative rounding
                right_imp = np.maximum(right_imp, 0.0)
            else:
                one_hot = np.zeros((n, self.n_classes_))
                one_hot[np.arange(n), y_sorted] = 1.0
                left_counts = np.cumsum(one_hot, axis=0)[:-1]
                right_counts = one_hot.sum(axis=0) - left_counts
                left_imp = self._class_impurity(left_counts, n_left)
                right_imp = self._class_impurity(right_counts, n_right)

            children_imp = (n_left * left_imp + n_right * right_imp) / n
            gain = parent_impurity - children_imp

            # a threshold can only sit between two different values,
            # and both children must keep min_samples_leaf samples
            valid = x_sorted[1:] > x_sorted[:-1]
            valid &= (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            if not np.any(valid):
                continue
            gain = np.where(valid, gain, -np.inf)
            i = int(np.argmax(gain))         # first best position on ties

            if gain[i] > best_gain:
                best_gain = float(gain[i])
                # midpoint between the two neighbouring values (like sklearn)
                threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0
                if threshold == x_sorted[i + 1]:        # float rounding guard
                    threshold = x_sorted[i]
                best = (feature, float(threshold), best_gain)

        return best
    
    def _class_impurity(self, counts, n):
        """
        Gini or entropy for many candidate nodes at once.
        counts: (m, k) class counts, n: (m,) node sizes -> (m,) impurities
        """
        p = counts / n[:, None]
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2, axis=1)
        # entropy, using 0 * log(0) = 0
        p_safe = np.where(p > 0, p, 1.0)
        return -np.sum(p * np.log2(p_safe), axis=1)
    
    # --- grow / predict -----------------------------------------------------
    def _leaf_value(self, y):
        if self.task == "regression":
            return float(np.mean(y))
        counts = np.bincount(y, minlength=self.n_classes_)
        return counts / len(y)
    
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        node = Node()
        node.n_samples = len(y)
        node.impurity = self._impurity(y)
        node.value = self._leaf_value(y)

        # stopping rules
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(y) < self.min_samples_split or len(y) < 2 * self.min_samples_leaf:
            return node
        if node.impurity == 0.0:                 # pure node
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, gain = split

        # impurity decrease weighted by the share of samples in this node
        # (same definition as sklearn's min_impurity_decrease)
        weighted_decrease = len(y) / self.n_samples_total_ * gain
        if weighted_decrease < self.min_impurity_decrease:
            return node

        self.feature_importances_[feature] += weighted_decrease
        go_left = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(X[go_left], y[go_left], depth + 1)
        node.right = self._grow(X[~go_left], y[~go_left], depth + 1)
        return node
        
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == "classification":
            # store the original labels, work with 0..k-1 inside the tree
            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
        else:
            y = y.astype(float)
        self.n_features_ = X.shape[1]
        self.n_samples_total_ = len(y)
        self.feature_importances_ = np.zeros(self.n_features_)

        self.root = self._grow(X, y, depth=0)

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total
        return self

    def _predict_one(self, x, node: Node):
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        values = np.array([self._predict_one(x, self.root) for x in X])
        if self.task == "regression":
            return values
        return self.classes_[np.argmax(values, axis=1)]   # majority class

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        if self.task != "classification":
            raise ValueError("predict_proba is only for classification")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])


    #For sklearn comparison:
    def get_depth(self) -> int:
        def depth(node):
            if node.is_leaf():
                return 0
            return 1 + max(depth(node.left), depth(node.right))
        return depth(self.root)

    def get_n_leaves(self) -> int:
        def leaves(node):
            if node.is_leaf():
                return 1
            return leaves(node.left) + leaves(node.right)
        return leaves(self.root)
