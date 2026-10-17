"""
test_sanity.py — sanity tests for our models against scikit-learn.

Run from the repo root (no extra packages needed):
    python -m tests.test_sanity
(pytest also works: python -m pytest tests)

scikit-learn is used here ONLY as the reference we compare against.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

from src.decision_tree import DecisionTree
from src.svm import PegasosSVM
from src import evaluate
from src import data_prep

# --- tiny, fully-determined toy data ----------------------------------------
X_TOY = np.array([
    [2.0, 7.0],
    [3.0, 1.0],
    [5.0, 6.0],
    [6.0, 3.0],
    [8.0, 8.0],
    [9.0, 2.0],
    [11.0, 5.0],
    [12.0, 4.0],
])
# labels chosen so that no two splits tie: sklearn builds the same tree for
# every random_state (checked inside the tests), so the answer is unique
Y_TOY_CLASS = np.array([0, 1, 1, 1, 0, 0, 1, 0])
Y_TOY_REG = np.array([17.0, 9.0, 18.0, 16.0, 14.0, 5.0, 15.0, 1.0])


def our_splits(tree):
    """Pre-order list of (feature, threshold), leaves as (-2, -2.0) like sklearn."""
    out = []

    def visit(node):
        if node.is_leaf():
            out.append((-2, -2.0))
        else:
            out.append((node.feature, node.threshold))
            visit(node.left)
            visit(node.right)

    visit(tree.root)
    return out


def sklearn_splits(model):
    t = model.tree_
    return [(int(f), float(th)) for f, th in zip(t.feature, t.threshold)]


# --- decision tree ----------------------------------------------------------
def test_tree_matches_sklearn_classification():
    for criterion in ["gini", "entropy"]:
        # the example is fully determined: every sklearn seed gives the same tree
        reference = None
        for seed in range(5):
            sk = DecisionTreeClassifier(criterion=criterion, random_state=seed).fit(X_TOY, Y_TOY_CLASS)
            if reference is None:
                reference = sklearn_splits(sk)
            assert sklearn_splits(sk) == reference

        ours = DecisionTree(task="classification", criterion=criterion).fit(X_TOY, Y_TOY_CLASS)
        assert our_splits(ours) == reference, (criterion, our_splits(ours), reference)
        assert np.array_equal(ours.predict(X_TOY), sk.predict(X_TOY))


def test_tree_matches_sklearn_regression():
    # depth 2: deeper nodes hold 2 samples, where both features tie
    reference = None
    for seed in range(5):
        sk = DecisionTreeRegressor(max_depth=2, random_state=seed).fit(X_TOY, Y_TOY_REG)
        if reference is None:
            reference = sklearn_splits(sk)
        assert sklearn_splits(sk) == reference

    ours = DecisionTree(task="regression", criterion="mse", max_depth=2).fit(X_TOY, Y_TOY_REG)
    assert our_splits(ours) == reference
    assert np.allclose(ours.predict(X_TOY), sk.predict(X_TOY))


def test_predict_proba_from_leaf_frequencies():
    sk = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_TOY, Y_TOY_CLASS)
    ours = DecisionTree(task="classification", max_depth=1).fit(X_TOY, Y_TOY_CLASS)
    proba = ours.predict_proba(X_TOY)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.allclose(proba, sk.predict_proba(X_TOY))


def test_hyperparameters_change_tree():
    """Each stopping rule must change the tree (and agree with sklearn)."""
    rng = np.random.default_rng(0)
    X = np.round(rng.normal(size=(400, 4)), 3)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.7, size=400) > 0).astype(int)
