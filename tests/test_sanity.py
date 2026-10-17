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

    full = DecisionTree(task="classification").fit(X, y)
    settings = [
        {"max_depth": 3},
        {"min_samples_split": 40},
        {"min_samples_leaf": 20},
        {"min_impurity_decrease": 0.01},
    ]
    for params in settings:
        ours = DecisionTree(task="classification", **params).fit(X, y)
        sk = DecisionTreeClassifier(random_state=0, **params).fit(X, y)
        assert ours.get_n_leaves() < full.get_n_leaves(), params
        assert ours.get_n_leaves() == sk.get_n_leaves(), params
        assert np.array_equal(ours.predict(X), sk.predict(X)), params


# --- SVM --------------------------------------------------------------------
def make_blobs(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-1.0, 1.0, size=(n // 2, 2)),
                   rng.normal(1.0, 1.0, size=(n // 2, 2))])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_svm_loss_decreases():
    X, y = make_blobs()
    svm = PegasosSVM(lambda_=0.01, n_epochs=30, batch_size=1).fit(X, y)
    history = svm.objective_history_
    assert history[-1] < history[0]
    assert history[-1] < 0.5          # starts at 1.0 (w = 0)


def test_svm_matches_sklearn_on_toy_split():
    X, y = make_blobs()
    X_test, y_test = make_blobs(seed=1)
    ours = PegasosSVM(lambda_=0.01, n_epochs=30, batch_size=1).fit(X, y)
    sk = SGDClassifier(loss="hinge", alpha=0.01, max_iter=30, tol=None, random_state=0).fit(X, y)
    acc_ours = np.mean(ours.predict(X_test) == y_test)
    acc_sk = np.mean(sk.predict(X_test) == y_test)
    assert abs(acc_ours - acc_sk) < 0.03
    assert np.mean(ours.predict(X_test) == sk.predict(X_test)) > 0.95


def test_svm_margin_and_support_vectors():
    X, y = make_blobs()
    svm = PegasosSVM(lambda_=0.01, n_epochs=30).fit(X, y)
    assert np.isclose(svm.margin_width(), 2.0 / np.linalg.norm(svm.w))
    sv = svm.support_vector_indices(X, y)
    y_pm = np.where(y == 1, 1.0, -1.0)
    margins = y_pm * svm.decision_function(X)
    assert np.all(margins[sv] <= 1.0)
    assert np.all(np.delete(margins, sv) > 1.0)


# --- metrics and data helpers -----------------------------------------------
def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=200)
    y_pred = y_true + rng.normal(scale=0.5, size=200)
    assert np.isclose(evaluate.rmse(y_true, y_pred), np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    assert np.isclose(evaluate.mae(y_true, y_pred), metrics.mean_absolute_error(y_true, y_pred))
    assert np.isclose(evaluate.r2(y_true, y_pred), metrics.r2_score(y_true, y_pred))

    labels = rng.integers(0, 2, size=300)
    predicted = rng.integers(0, 2, size=300)
    scores = np.round(rng.random(300), 1)            # rounded -> many ties
    assert np.array_equal(evaluate.confusion_matrix(labels, predicted),
                          metrics.confusion_matrix(labels, predicted))
    p, r, f1 = evaluate.precision_recall_f1(labels, predicted)
    assert np.isclose(p, metrics.precision_score(labels, predicted))
    assert np.isclose(r, metrics.recall_score(labels, predicted))
    assert np.isclose(f1, metrics.f1_score(labels, predicted))
    assert np.isclose(evaluate.roc_auc(labels, scores), metrics.roc_auc_score(labels, scores))


def test_tier_threshold_comes_from_train_only():
    price_train = np.array([100.0, 200.0, 300.0, 400.0])
    price_test = np.array([1000.0, 1000.0, 50.0])
    tier_train, threshold = data_prep.make_tier_label(price_train)
    tier_test, same = data_prep.make_tier_label(price_test, threshold)
    assert threshold == 250.0 and same == 250.0
    assert tier_train.tolist() == [0, 0, 1, 1]
    assert tier_test.tolist() == [1, 1, 0]


if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_")]
    for name, test in tests:
        test()
        print("PASS", name)
    print(f"All {len(tests)} tests passed.")
