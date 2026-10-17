"""
run_all.py — ONE command reproduces every headline number and figure.

Run from the repo root:
    python -m src.run_all

Steps:
  1. load + clean the data, build features, split, tier label, standardize,
  2. EDA figure (training split only),
  3. tune OUR decision tree (Task A regression, Task B classification)
     and OUR Ridge regression (bonus Problem 8, Task A baseline),
  4. tune OUR Pegasos SVM (Task B, linear + random Fourier features),
  5. train the final models and the scikit-learn baselines (matched
     settings) and score them on the test set,
  6. 5-fold cross-validation (mean ± std) on train+val,
  7. results tables, tree/SVM comparison, importances, error analysis,
     and all figures saved to report/figures/.

Everything is deterministic (fixed seeds). Takes roughly 10 minutes.
"""

from __future__ import annotations

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # save figures, no window
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, Ridge
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import data_prep
from .decision_tree import DecisionTree
from .svm import PegasosSVM, RandomFourierFeatures
from .ridge import RidgeRegression
from . import evaluate

SEED = data_prep.SEED
FIG_DIR = "report/figures"

# hyperparameter grids (searched on the VALIDATION split only)
DEPTHS = [2, 4, 6, 8, 10, 12, 14, 16, 20]
LEAF_SIZES = [1, 5, 10, 20, 50, 100]
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SCHEDULES = ["pegasos", "constant", "invsqrt"]
EPOCH_CHOICES = [1, 5, 20]
GAMMAS = [0.003, 0.01, 0.03, 0.1, 0.3]
RFF_LAMBDAS = [1e-5, 1e-4, 1e-3]
RIDGE_LAMBDAS = [0.01, 0.1, 1, 10, 100, 1000, 10000]

N_EPOCHS = 20        # Pegasos epochs
BATCH_SIZE = 32      # mini-batch Pegasos: ~20x faster per epoch, same objective
ETA0 = 0.01          # base rate for the "constant" / "invsqrt" schedules
N_RFF = 500          # number of random Fourier features
N_FOLDS = 5

# plot colours (fixed order: first series blue, second orange, third green)
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [saved figure] {path}")


# =============================================================================
# 1. Data
# =============================================================================
def load_data():
    section("1. DATA: load, clean, features, split")
    df = data_prep.load_raw()
    print("Cleaning steps:")
    df = data_prep.clean(df)
    X, y, names = data_prep.make_features(df)
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_prep.train_val_test_split(X, y)

    # Task B label: threshold = median of the TRAINING prices only
    tier_tr, threshold = data_prep.make_tier_label(y_tr)
    tier_val, _ = data_prep.make_tier_label(y_val, threshold)
    tier_te, _ = data_prep.make_tier_label(y_te, threshold)

    print(f"\nRows after cleaning : {len(y)}")
    print(f"Features            : {len(names)}")
    print(f"  {names}")
    print(f"Split sizes         : train {len(y_tr)} / val {len(y_val)} / test {len(y_te)}")
    print(f"Tier threshold      : {threshold:,.0f} AZN (median of TRAIN price)")
    print(f"Premium share       : train {tier_tr.mean():.3f} / val {tier_val.mean():.3f}"
          f" / test {tier_te.mean():.3f}")

    return {
        "names": names,
        "X_tr": X_tr, "X_val": X_val, "X_te": X_te,
        "price_tr": y_tr, "price_val": y_val, "price_te": y_te,
        # Task A target: log(price), prices are heavy-tailed
        "log_tr": np.log(y_tr), "log_val": np.log(y_val), "log_te": np.log(y_te),
        "tier_tr": tier_tr, "tier_val": tier_val, "tier_te": tier_te,
        "threshold": threshold,
    }


def plot_eda(d):
    """EDA on the TRAINING split only (no peeking at val/test)."""
    section("2. EDA (training split)")
    names, X, price = d["names"], d["X_tr"], d["price_tr"]
    area = X[:, names.index("area")]
    rooms = X[:, names.index("rooms")]
    lat, lng = X[:, names.index("lat")], X[:, names.index("lng")]
    log_price = np.log(price)

    print(f"Price: median {np.median(price):,.0f}  mean {price.mean():,.0f}"
          f"  max {price.max():,.0f} AZN")
    print(f"corr(log area, log price) = {np.corrcoef(np.log(area), log_price)[0, 1]:.3f}")
    for cat in ["cat_new_build", "cat_old_build", "cat_house"]:
        mask = X[:, names.index(cat)] == 1
        print(f"  {cat:<15s} n={mask.sum():>6d}  median price {np.median(price[mask]):>10,.0f} AZN")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    ax.hist(price / 1000, bins=100, color=BLUE)
    ax.set_xlim(0, np.percentile(price / 1000, 99))
    ax.set_xlabel("price (thousand AZN, up to 99th pct)")
    ax.set_ylabel("listings")
    ax.set_title("(a) Price is heavy-tailed")

    ax = axes[0, 1]
    ax.hist(log_price, bins=60, color=BLUE)
    ax.set_xlabel("log(price)")
    ax.set_ylabel("listings")
    ax.set_title("(b) log(price) is close to symmetric")

    ax = axes[0, 2]
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(price), size=5000, replace=False)
    ax.scatter(area[sample], price[sample] / 1000, s=4, alpha=0.3, color=BLUE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("area (m², log scale)")
    ax.set_ylabel("price (thousand AZN, log scale)")
    ax.set_title("(c) Area vs price (5,000 random listings)")

    ax = axes[1, 0]
    groups = [log_price[rooms == r] for r in range(1, 7)] + [log_price[rooms >= 7]]
    ax.boxplot(groups, showfliers=False)
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7+"])
    ax.set_xlabel("rooms")
    ax.set_ylabel("log(price)")
    ax.set_title("(d) Price by number of rooms")

    ax = axes[1, 1]
    in_baku = (lat > 40.25) & (lat < 40.65) & (lng > 49.65) & (lng < 50.35)
    points = ax.scatter(lng[in_baku], lat[in_baku], c=log_price[in_baku], cmap="Blues",
                        s=2, alpha=0.6)
    fig.colorbar(points, ax=ax, label="log(price)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("(e) Location effect (Baku area)")

    ax = axes[1, 2]
    shares = [d["tier_tr"].mean(), d["tier_val"].mean(), d["tier_te"].mean()]
    ax.bar(["train", "validation", "test"], shares, color=BLUE, width=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of premium listings")
    ax.set_title(f"(f) Tier balance (threshold {d['threshold']:,.0f} AZN)")
    for i, share in enumerate(shares):
        ax.text(i, share + 0.02, f"{share:.3f}", ha="center")

    save(fig, "eda.png")


# =============================================================================
# 2. Decision tree studies (validation only)
# =============================================================================
def study_tree_regression(d):
    section("3a. TASK A - our decision tree (regression on log price)")
    train_rmse, val_rmse = [], []
    for depth in DEPTHS:
        tree = DecisionTree(task="regression", criterion="mse", max_depth=depth)
        tree.fit(d["X_tr"], d["log_tr"])
        train_rmse.append(evaluate.rmse(d["log_tr"], tree.predict(d["X_tr"])))
        val_rmse.append(evaluate.rmse(d["log_val"], tree.predict(d["X_val"])))
        print(f"  max_depth={depth:>2d}  leaves={tree.get_n_leaves():>5d}"
              f"  train RMSE={train_rmse[-1]:.4f}  val RMSE={val_rmse[-1]:.4f}")
    best_depth = DEPTHS[int(np.argmin(val_rmse))]

    leaf_train, leaf_val = [], []
    for leaf in LEAF_SIZES:
        tree = DecisionTree(task="regression", criterion="mse", max_depth=best_depth,
                            min_samples_leaf=leaf)
        tree.fit(d["X_tr"], d["log_tr"])
        leaf_train.append(evaluate.rmse(d["log_tr"], tree.predict(d["X_tr"])))
        leaf_val.append(evaluate.rmse(d["log_val"], tree.predict(d["X_val"])))
        print(f"  max_depth={best_depth}, min_samples_leaf={leaf:>3d}  leaves={tree.get_n_leaves():>5d}"
              f"  train RMSE={leaf_train[-1]:.4f}  val RMSE={leaf_val[-1]:.4f}")
    best_leaf = LEAF_SIZES[int(np.argmin(leaf_val))]
    print(f"Chosen on validation: max_depth={best_depth}, min_samples_leaf={best_leaf}")

    return {"train": train_rmse, "val": val_rmse, "leaf_train": leaf_train,
            "leaf_val": leaf_val, "max_depth": best_depth, "min_samples_leaf": best_leaf}


def study_tree_classification(d):
    section("3b. TASK B - our decision tree (classification): Gini vs entropy")
    curves = {}
    best_f1, best_criterion, best_depth = -1.0, None, None
    for criterion in ["gini", "entropy"]:
        train_f1, val_f1 = [], []
        for depth in DEPTHS:
            tree = DecisionTree(task="classification", criterion=criterion, max_depth=depth)
            tree.fit(d["X_tr"], d["tier_tr"])
            train_f1.append(evaluate.precision_recall_f1(d["tier_tr"], tree.predict(d["X_tr"]))[2])
            val_f1.append(evaluate.precision_recall_f1(d["tier_val"], tree.predict(d["X_val"]))[2])
            print(f"  {criterion:<7s} max_depth={depth:>2d}  leaves={tree.get_n_leaves():>5d}"
                  f"  train F1={train_f1[-1]:.4f}  val F1={val_f1[-1]:.4f}")
            if val_f1[-1] > best_f1:
                best_f1, best_criterion, best_depth = val_f1[-1], criterion, depth
        curves[criterion] = (train_f1, val_f1)

    leaf_train, leaf_val = [], []
    for leaf in LEAF_SIZES:
        tree = DecisionTree(task="classification", criterion=best_criterion,
                            max_depth=best_depth, min_samples_leaf=leaf)
        tree.fit(d["X_tr"], d["tier_tr"])
        leaf_train.append(evaluate.precision_recall_f1(d["tier_tr"], tree.predict(d["X_tr"]))[2])
        leaf_val.append(evaluate.precision_recall_f1(d["tier_val"], tree.predict(d["X_val"]))[2])
        print(f"  {best_criterion}, max_depth={best_depth}, min_samples_leaf={leaf:>3d}"
              f"  train F1={leaf_train[-1]:.4f}  val F1={leaf_val[-1]:.4f}")
    best_leaf = LEAF_SIZES[int(np.argmax(leaf_val))]
    print(f"Chosen on validation: criterion={best_criterion}, max_depth={best_depth},"
          f" min_samples_leaf={best_leaf}")

    return {"curves": curves, "leaf_train": leaf_train, "leaf_val": leaf_val,
            "criterion": best_criterion, "max_depth": best_depth, "min_samples_leaf": best_leaf}


def plot_tree_curves(reg, clf):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(DEPTHS, reg["train"], "o-", color=BLUE, label="train")
    ax.plot(DEPTHS, reg["val"], "o-", color=ORANGE, label="validation")
    ax.axvline(reg["max_depth"], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("RMSE on log(price)")
    ax.set_title("(a) Task A: depth sweep (our tree, MSE)")
    ax.legend()

    ax = axes[1]
    ax.plot(DEPTHS, clf["curves"]["gini"][0], "o--", color=BLUE, label="Gini - train")
    ax.plot(DEPTHS, clf["curves"]["gini"][1], "o-", color=BLUE, label="Gini - validation")
    ax.plot(DEPTHS, clf["curves"]["entropy"][0], "s--", color=ORANGE, label="entropy - train")
    ax.plot(DEPTHS, clf["curves"]["entropy"][1], "s-", color=ORANGE, label="entropy - validation")
    ax.axvline(clf["max_depth"], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("F1 (premium class)")
    ax.set_title("(b) Task B: Gini vs entropy")
    ax.legend()

    ax = axes[2]
    ax.plot(LEAF_SIZES, reg["leaf_train"], "o-", color=BLUE, label="train")
    ax.plot(LEAF_SIZES, reg["leaf_val"], "o-", color=ORANGE, label="validation")
    ax.axvline(reg["min_samples_leaf"], color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("min_samples_leaf (log scale)")
    ax.set_ylabel("RMSE on log(price)")
    ax.set_title(f"(c) Task A: min_samples_leaf (max_depth={reg['max_depth']})")
    ax.legend()

    save(fig, "tree_curves.png")


# =============================================================================
# Bonus (Problem 8): Ridge regression study (validation only)
# =============================================================================
def study_ridge(d):
    section("3c. BONUS - our Ridge regression (normal equations, Task A)")
    S_tr, S_val = data_prep.standardize(d["X_tr"], d["X_val"])
    val_rmse = []
    for lam in RIDGE_LAMBDAS:
        ridge = RidgeRegression(lambda_=lam).fit(S_tr, d["log_tr"])
        train_rmse = evaluate.rmse(d["log_tr"], ridge.predict(S_tr))
        val_rmse.append(evaluate.rmse(d["log_val"], ridge.predict(S_val)))
        print(f"  lambda={lam:<7g} ||w||={np.linalg.norm(ridge.coef_):6.3f}"
              f"  train RMSE={train_rmse:.4f}  val RMSE={val_rmse[-1]:.4f}")
    best_lambda = RIDGE_LAMBDAS[int(np.argmin(val_rmse))]
    print(f"Chosen on validation: lambda={best_lambda}")
    return {"lambda": best_lambda}


# =============================================================================
# 3. SVM studies (validation only)
# =============================================================================
def study_svm(d):
    section("4. TASK B - our Pegasos SVM")
    S_tr, S_val = data_prep.standardize(d["X_tr"], d["X_val"])
    tier_tr, tier_val = d["tier_tr"], d["tier_val"]

    # (a) regularisation sweep, linear SVM
    print("(a) lambda sweep (linear, pegasos schedule, 20 epochs)")
    margins, train_f1, val_f1 = [], [], []
    for lam in LAMBDAS:
        svm = PegasosSVM(lambda_=lam, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                         random_state=SEED).fit(S_tr, tier_tr)
        n_sv = len(svm.support_vector_indices(S_tr, tier_tr))
        margins.append(svm.margin_width())
        train_f1.append(evaluate.precision_recall_f1(tier_tr, svm.predict(S_tr))[2])
        val_f1.append(evaluate.precision_recall_f1(tier_val, svm.predict(S_val))[2])
        print(f"  lambda={lam:<7g} ||w||={svm.weight_norm():7.3f}  margin 2/||w||={margins[-1]:6.3f}"
              f"  support vectors={n_sv:>6d}  final J={svm.objective_history_[-1]:.4f}"
              f"  train F1={train_f1[-1]:.4f}  val F1={val_f1[-1]:.4f}")
    best_lambda = LAMBDAS[int(np.argmax(val_f1))]
    print(f"Chosen lambda (validation): {best_lambda}")

    # (b) epochs and learning-rate schedule at the chosen lambda
    print(f"(b) learning-rate schedule x epochs (lambda={best_lambda}, eta0={ETA0})")
    histories = {}
    for schedule in SCHEDULES:
        for epochs in EPOCH_CHOICES:
            svm = PegasosSVM(lambda_=best_lambda, n_epochs=epochs, batch_size=BATCH_SIZE,
                             lr_schedule=schedule, eta0=ETA0, random_state=SEED).fit(S_tr, tier_tr)
            f1 = evaluate.precision_recall_f1(tier_val, svm.predict(S_val))[2]
            print(f"  {schedule:<9s} epochs={epochs:>2d}  final J={svm.objective_history_[-1]:.4f}"
                  f"  ||w||={svm.weight_norm():7.3f}  margin={svm.margin_width():6.3f}  val F1={f1:.4f}")
            if epochs == N_EPOCHS:
                histories[schedule] = svm.objective_history_

    # (c) non-linear: random Fourier features, bandwidth x lambda
    print(f"(c) random Fourier features (D={N_RFF}): gamma x lambda")
    rff_f1 = {lam: [] for lam in RFF_LAMBDAS}
    best_rff = (-1.0, None, None)
    for gamma in GAMMAS:
        rff = RandomFourierFeatures(N_RFF, gamma, random_state=SEED).fit(S_tr)
        Z_tr, Z_val = rff.transform(S_tr), rff.transform(S_val)
        for lam in RFF_LAMBDAS:
            svm = PegasosSVM(lambda_=lam, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                             random_state=SEED).fit(Z_tr, tier_tr)
            f1 = evaluate.precision_recall_f1(tier_val, svm.predict(Z_val))[2]
            rff_f1[lam].append(f1)
            print(f"  gamma={gamma:<6g} lambda={lam:<7g} margin={svm.margin_width():6.3f}  val F1={f1:.4f}")
            if f1 > best_rff[0]:
                best_rff = (f1, gamma, lam)
    print(f"Chosen RFF config (validation): gamma={best_rff[1]}, lambda={best_rff[2]}")

    return {"margins": margins, "train_f1": train_f1, "val_f1": val_f1, "lambda": best_lambda,
            "histories": histories, "rff_f1": rff_f1, "gamma": best_rff[1],
            "rff_lambda": best_rff[2]}


def plot_svm_curves(s):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    for schedule, color in zip(SCHEDULES, [BLUE, ORANGE, GREEN]):
        ax.plot(range(N_EPOCHS + 1), s["histories"][schedule], "o-", markersize=3,
                color=color, label=schedule)
    ax.set_xlabel("epoch")
    ax.set_ylabel("objective J(w, b)")
    ax.set_title(f"(a) Training objective (lambda={s['lambda']})")
    ax.legend(title="learning-rate schedule")

    ax = axes[0, 1]
    ax.plot(LAMBDAS, s["margins"], "o-", color=BLUE)
    ax.set_xscale("log")
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("margin width 2 / ||w||")
    ax.set_title("(b) Stronger regularisation = wider margin")

    ax = axes[1, 0]
    ax.plot(LAMBDAS, s["train_f1"], "o--", color=BLUE, label="train")
    ax.plot(LAMBDAS, s["val_f1"], "o-", color=ORANGE, label="validation")
    ax.axvline(s["lambda"], color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("F1 (premium class)")
    ax.set_title("(c) Linear SVM: F1 vs lambda")
    ax.legend()

    ax = axes[1, 1]
    for lam, color in zip(RFF_LAMBDAS, [BLUE, ORANGE, GREEN]):
        ax.plot(GAMMAS, s["rff_f1"][lam], "o-", color=color, label=f"lambda={lam:g}")
    ax.set_xscale("log")
    ax.set_xlabel("RBF bandwidth gamma (log scale)")
    ax.set_ylabel("validation F1")
    ax.set_title(f"(d) Random Fourier features (D={N_RFF})")
    ax.legend()

    save(fig, "svm_curves.png")


# =============================================================================
# 4. Final models + scikit-learn baselines
# =============================================================================
def fit_and_predict(model, A_tr, y_tr, A_ev, task):
    """Fit, predict, time both. For Task B also return a score for ROC-AUC."""
    start = time.perf_counter()
    model.fit(A_tr, y_tr)
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(A_ev)
    score = None
    if task == "B":
        if hasattr(model, "decision_function"):      # SVMs: signed margin
            score = model.decision_function(A_ev)
        else:                                        # trees: leaf frequency
            score = model.predict_proba(A_ev)[:, 1]
    predict_s = time.perf_counter() - start
    return {"model": model, "pred": pred, "score": score, "fit_s": fit_s, "predict_s": predict_s}


def run_task_a(cfg, X_tr, log_tr, X_ev):
    """Our regression tree and Ridge, and sklearn's, with the same hyperparameters."""
    # Ridge needs scaled features (train statistics); sklearn's Ridge does the
    # same scaling inside a Pipeline. Trees use the raw features.
    S_tr, S_ev = data_prep.standardize(X_tr, X_ev)
    models = [
        ("Ours: decision tree",
         DecisionTree(task="regression", criterion="mse", max_depth=cfg["reg_depth"],
                      min_samples_leaf=cfg["reg_leaf"]),
         X_tr, X_ev),
        ("sklearn DecisionTreeRegressor",
         DecisionTreeRegressor(criterion="squared_error", max_depth=cfg["reg_depth"],
                               min_samples_leaf=cfg["reg_leaf"], random_state=SEED),
         X_tr, X_ev),
        ("Ours: Ridge (normal equations)",
         RidgeRegression(lambda_=cfg["ridge_lambda"]),
         S_tr, S_ev),
        ("sklearn Ridge",
         Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=cfg["ridge_lambda"]))]),
         X_tr, X_ev),
    ]
    out = {}
    for name, model, A_tr, A_ev in models:
        out[name] = fit_and_predict(model, A_tr, log_tr, A_ev, "A")
    return out


def run_task_b(cfg, X_tr, tier_tr, X_ev):
    """Our tree + SVMs and the sklearn baselines, with matched settings."""
    # our SVMs get features scaled with TRAIN statistics; the sklearn SVMs do
    # the same thing inside a Pipeline (StandardScaler -> model)
    S_tr, S_ev = data_prep.standardize(X_tr, X_ev)
    rff = RandomFourierFeatures(N_RFF, cfg["gamma"], random_state=SEED).fit(S_tr)
    Z_tr, Z_ev = rff.transform(S_tr), rff.transform(S_ev)
    n = len(tier_tr)

    models = [
        ("Ours: decision tree",
         DecisionTree(task="classification", criterion=cfg["clf_criterion"],
                      max_depth=cfg["clf_depth"], min_samples_leaf=cfg["clf_leaf"]),
         X_tr, X_ev),
        ("sklearn DecisionTreeClassifier",
         DecisionTreeClassifier(criterion=cfg["clf_criterion"], max_depth=cfg["clf_depth"],
                                min_samples_leaf=cfg["clf_leaf"], random_state=SEED),
         X_tr, X_ev),
        ("Ours: linear SVM (Pegasos)",
         PegasosSVM(lambda_=cfg["svm_lambda"], n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                    random_state=SEED),
         S_tr, S_ev),
        ("sklearn SGDClassifier (hinge)",
         Pipeline([("scale", StandardScaler()),
                   ("svm", SGDClassifier(loss="hinge", alpha=cfg["svm_lambda"],
                                         max_iter=N_EPOCHS, tol=None, random_state=SEED))]),
         X_tr, X_ev),
        ("Ours: RFF SVM (Pegasos)",
         PegasosSVM(lambda_=cfg["rff_lambda"], n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                    random_state=SEED),
         Z_tr, Z_ev),
        # same RBF bandwidth, C = 1 / (n * lambda) matches our objective
        ("sklearn SVC (RBF)",
         Pipeline([("scale", StandardScaler()),
                   ("svm", SVC(kernel="rbf", gamma=cfg["gamma"], C=1.0 / (n * cfg["rff_lambda"])))]),
         X_tr, X_ev),
    ]
    out = {}
    for name, model, A_tr, A_ev in models:
        out[name] = fit_and_predict(model, A_tr, tier_tr, A_ev, "B")
    return out


def evaluate_on_test(d, cfg):
    section("5. FINAL MODELS on the TEST set (configs chosen on validation)")
    for key, value in cfg.items():
        print(f"  {key:<14s} = {value}")
    task_a = run_task_a(cfg, d["X_tr"], d["log_tr"], d["X_te"])
    task_b = run_task_b(cfg, d["X_tr"], d["tier_tr"], d["X_te"])
    for name, r in task_a.items():
        r["metrics"] = evaluate.regression_report(d["log_te"], r["pred"])
    for name, r in task_b.items():
        r["metrics"] = evaluate.classification_report(d["tier_te"], r["pred"], r["score"])
    return task_a, task_b


def cross_validate(d, cfg):
    """5-fold CV on train+val; threshold and scaling refit inside every fold."""
    section(f"6. {N_FOLDS}-FOLD CROSS-VALIDATION on train+val (uncertainty)")
    X_pool = np.vstack([d["X_tr"], d["X_val"]])
    price_pool = np.concatenate([d["price_tr"], d["price_val"]])
    cv_a, cv_b = {}, {}
    for fold, (tr, ev) in enumerate(evaluate.kfold_indices(len(price_pool), N_FOLDS, SEED)):
        tier_tr, threshold = data_prep.make_tier_label(price_pool[tr])
        tier_ev, _ = data_prep.make_tier_label(price_pool[ev], threshold)
        log_tr, log_ev = np.log(price_pool[tr]), np.log(price_pool[ev])

        for name, r in run_task_a(cfg, X_pool[tr], log_tr, X_pool[ev]).items():
            cv_a.setdefault(name, []).append(evaluate.rmse(log_ev, r["pred"]))
        for name, r in run_task_b(cfg, X_pool[tr], tier_tr, X_pool[ev]).items():
            metrics = evaluate.classification_report(tier_ev, r["pred"], r["score"])
            cv_b.setdefault(name, {"F1": [], "ROC_AUC": []})
            cv_b[name]["F1"].append(metrics["F1"])
            cv_b[name]["ROC_AUC"].append(metrics["ROC_AUC"])
        print(f"  fold {fold + 1}/{N_FOLDS} done")
    return cv_a, cv_b


def results_tables(task_a, task_b, cv_a, cv_b, d):
    section("7. RESULTS TABLES (test set + CV mean +/- std on train+val)")
    print("Task A - regression on log(price)  [metrics in AZN after exp()]")
    table = {}
    for name, r in task_a.items():
        m = r["metrics"]
        table[name] = {"RMSE_log": m["RMSE_log"], "MAE_log": m["MAE_log"], "R2_log": m["R2_log"],
                       "RMSE_AZN": f"{m['RMSE_AZN']:,.0f}", "MAE_AZN": f"{m['MAE_AZN']:,.0f}",
                       "R2_AZN": m["R2_AZN"], "CV RMSE_log": evaluate.mean_std(cv_a[name]),
                       "fit_s": f"{r['fit_s']:.2f}", "predict_s": f"{r['predict_s']:.3f}"}
    evaluate.compare(table)

    print("\nTask B - price tier (premium = 1)")
    table = {}
    for name, r in task_b.items():
        m = r["metrics"]
        table[name] = {"accuracy": m["accuracy"], "precision": m["precision"],
                       "recall": m["recall"], "F1": m["F1"], "ROC_AUC": m["ROC_AUC"],
                       "CV F1": evaluate.mean_std(cv_b[name]["F1"]),
                       "CV ROC_AUC": evaluate.mean_std(cv_b[name]["ROC_AUC"]),
                       "fit_s": f"{r['fit_s']:.2f}", "predict_s": f"{r['predict_s']:.3f}"}
    evaluate.compare(table)

    print("\nConfusion matrices on test [[TN, FP], [FN, TP]]:")
    for name, r in task_b.items():
        cm = evaluate.confusion_matrix(d["tier_te"], r["pred"])
        print(f"  {name:<32s} {cm.tolist()}")


# =============================================================================
# 5. Our models vs sklearn: structure, margin, weights
# =============================================================================
def compare_with_sklearn(task_a, task_b, d):
    section("8. OUR MODELS vs SCIKIT-LEARN (structure and agreement)")
    names = d["names"]
    for task, results in [("A", task_a), ("B", task_b)]:
        ours = results["Ours: decision tree"]
        theirs = results["sklearn DecisionTreeRegressor" if task == "A"
                         else "sklearn DecisionTreeClassifier"]
        o, s = ours["model"], theirs["model"]
        print(f"Task {task} tree:")
        print(f"  leaves: ours {o.get_n_leaves()} vs sklearn {s.get_n_leaves()};"
              f" depth: ours {o.get_depth()} vs sklearn {s.get_depth()}")
        print(f"  root split: ours {names[o.root.feature]} <= {o.root.threshold:.4f}"
              f"  |  sklearn {names[s.tree_.feature[0]]} <= {s.tree_.threshold[0]:.4f}")
        if task == "A":
            diff = np.abs(ours["pred"] - theirs["pred"])
            print(f"  identical predictions: {np.mean(diff < 1e-9):.4f};"
                  f" mean |difference| = {diff.mean():.4f} (log units)")
        else:
            print(f"  identical predictions: {np.mean(ours['pred'] == theirs['pred']):.4f}")

    svm = task_b["Ours: linear SVM (Pegasos)"]["model"]
    sgd = task_b["sklearn SGDClassifier (hinge)"]["model"].named_steps["svm"]
    S_tr = data_prep.standardize(d["X_tr"])[0]
    sv = svm.support_vector_indices(S_tr, d["tier_tr"])
    print("Linear SVM (final, trained on train split):")
    print(f"  ||w|| = {svm.weight_norm():.3f}, margin 2/||w|| = {svm.margin_width():.3f}, b = {svm.b:.3f}")
    print(f"  support vectors (y(w.x+b) <= 1): {len(sv)} of {len(S_tr)} training points"
          f" ({len(sv) / len(S_tr):.1%})")
    print(f"  sklearn SGDClassifier ||w|| = {np.linalg.norm(sgd.coef_):.3f}")
    agree = np.mean(task_b["Ours: linear SVM (Pegasos)"]["pred"]
                    == task_b["sklearn SGDClassifier (hinge)"]["pred"])
    print(f"  identical predictions ours vs SGDClassifier: {agree:.4f}")
    agree = np.mean(task_b["Ours: RFF SVM (Pegasos)"]["pred"]
                    == task_b["sklearn SVC (RBF)"]["pred"])
    print(f"  identical predictions ours RFF vs SVC(RBF): {agree:.4f}")

    ridge = task_a["Ours: Ridge (normal equations)"]["model"]
    sk_ridge = task_a["sklearn Ridge"]["model"].named_steps["ridge"]
    pred_diff = np.abs(task_a["Ours: Ridge (normal equations)"]["pred"]
                       - task_a["sklearn Ridge"]["pred"])
    print("Ridge regression (bonus, Task A):")
    print(f"  max |coefficient difference| ours vs sklearn = {np.abs(ridge.coef_ - sk_ridge.coef_).max():.2e}")
    print(f"  max |prediction difference| = {pred_diff.max():.2e} (log units)")
    print("  largest coefficients (standardized features, log price):")
    for i in np.argsort(np.abs(ridge.coef_))[::-1][:5]:
        print(f"    {names[i]:<28s} {ridge.coef_[i]:+.4f}")


def plot_confusion(task_b, d):
    shown = ["Ours: decision tree", "sklearn DecisionTreeClassifier",
             "Ours: linear SVM (Pegasos)", "Ours: RFF SVM (Pegasos)"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, name in zip(axes, shown):
        cm = evaluate.confusion_matrix(d["tier_te"], task_b[name]["pred"])
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["standard", "premium"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["standard", "premium"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(name, fontsize=10)
    save(fig, "confusion_matrices.png")


def plot_importance_and_weights(task_a, task_b, d):
    section("9. FEATURE IMPORTANCE (tree) and SVM WEIGHTS")
    names = np.array(d["names"])
    ours = task_a["Ours: decision tree"]["model"].feature_importances_
    theirs = task_a["sklearn DecisionTreeRegressor"]["model"].feature_importances_
    ours_b = task_b["Ours: decision tree"]["model"].feature_importances_
    # rank by the sum, so a feature used by only one of the two trees shows up
    top_a = np.argsort(ours + theirs)[::-1][:10]
    print("Task A tree, impurity-decrease importance (ours vs sklearn), top 10:")
    for i in top_a:
        print(f"  {names[i]:<28s} ours {ours[i]:.4f}   sklearn {theirs[i]:.4f}")
    print("Task B tree importance (ours), top 5:")
    for i in np.argsort(ours_b)[::-1][:5]:
        print(f"  {names[i]:<28s} {ours_b[i]:.4f}")

    w = task_b["Ours: linear SVM (Pegasos)"]["model"].w
    print("Linear SVM weights (standardized features), largest |w|, top 10:")
    for i in np.argsort(np.abs(w))[::-1][:10]:
        print(f"  {names[i]:<28s} {w[i]:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    top = top_a[::-1]                      # largest at the top of the plot
    pos = np.arange(len(top))
    ax.barh(pos + 0.2, ours[top], height=0.4, color=BLUE, label="ours")
    ax.barh(pos - 0.2, theirs[top], height=0.4, color=ORANGE, label="sklearn")
    ax.set_yticks(pos)
    ax.set_yticklabels(names[top])
    ax.set_xlabel("share of total impurity decrease")
    ax.set_title("(a) Task A tree: feature importance (top 10)")
    ax.legend()

    ax = axes[1]
    top = np.argsort(np.abs(w))[::-1][:10][::-1]
    ax.barh(np.arange(len(top)), w[top], color=BLUE)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(names[top])
    ax.set_xlabel("weight (positive = pushes towards premium)")
    ax.set_title("(b) Linear SVM: largest weights")
    save(fig, "importance_weights.png")


# =============================================================================
# 6. Error analysis
# =============================================================================
def error_analysis(task_a, task_b, d):
    section("10. ERROR ANALYSIS (test set)")
    names, X = d["names"], d["X_te"]
    price = d["price_te"]
    cat_new = X[:, names.index("cat_new_build")] == 1
    cat_old = X[:, names.index("cat_old_build")] == 1
    cat_house = X[:, names.index("cat_house")] == 1
    loc_cols = [i for i, n in enumerate(names) if n.startswith("loc_")]
    top_district = X[:, loc_cols].sum(axis=1) == 1
    near_threshold = np.abs(price / d["threshold"] - 1) < 0.10
    groups = {
        "new build": cat_new, "old build": cat_old, "house": cat_house,
        "top-20 districts": top_district, "other districts": ~top_district,
        "near threshold": near_threshold,      # price within 10% of the threshold
    }

    # Task A: where does the regression tree fail?
    tree_err = np.abs(task_a["Ours: decision tree"]["pred"] - d["log_te"])
    sk_err = np.abs(task_a["sklearn DecisionTreeRegressor"]["pred"] - d["log_te"])
    print("Task A, our tree: MAE on log(price) by group")
    for group, mask in groups.items():
        if group == "near threshold":
            continue
        print(f"  {group:<18s} n={mask.sum():>5d}  MAE_log={tree_err[mask].mean():.4f}")

    deciles = np.percentile(price, np.arange(10, 100, 10))
    decile_id = np.searchsorted(deciles, price, side="right")
    mae_ours = [tree_err[decile_id == k].mean() for k in range(10)]
    mae_sk = [sk_err[decile_id == k].mean() for k in range(10)]
    print("Task A, MAE_log by true-price decile (1 = cheapest, 10 = most expensive):")
    print("  ours   : " + " ".join(f"{v:.3f}" for v in mae_ours))
    print("  sklearn: " + " ".join(f"{v:.3f}" for v in mae_sk))

    print("Task A, 5 worst predictions of our tree:")
    pred_price = np.exp(task_a["Ours: decision tree"]["pred"])
    for i in np.argsort(tree_err)[::-1][:5]:
        kind = "house" if cat_house[i] else ("new build" if cat_new[i] else "old build")
        print(f"  {kind:<9s} area={X[i, names.index('area')]:>6.0f} m2"
              f" rooms={X[i, names.index('rooms')]:.0f}"
              f"  true={price[i]:>10,.0f}  predicted={pred_price[i]:>10,.0f} AZN")

    # Task B: error rate per group for the tree and both SVMs
    shown = {"Ours: decision tree": "tree",
             "Ours: linear SVM (Pegasos)": "linear SVM",
             "Ours: RFF SVM (Pegasos)": "RFF SVM"}
    print("Task B, error rate by group (near threshold = price within 10% of it):")
    print(f"  {'group':<18s} {'n':>5s}" + "".join(f"{s:>12s}" for s in shown.values()))
    rates = {name: [] for name in shown}
    for group, mask in groups.items():
        row = f"  {group:<18s} {mask.sum():>5d}"
        for name in shown:
            wrong = task_b[name]["pred"][mask] != d["tier_te"][mask]
            rates[name].append(wrong.mean())
            row += f"{wrong.mean():>12.3f}"
        print(row)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(range(1, 11), mae_ours, "o-", color=BLUE, label="ours")
    ax.plot(range(1, 11), mae_sk, "s--", color=ORANGE, label="sklearn")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("true-price decile (1 = cheapest, 10 = most expensive)")
    ax.set_ylabel("MAE on log(price)")
    ax.set_title("(a) Task A: tree errors by price decile")
    ax.legend()

    ax = axes[1]
    pos = np.arange(len(groups))
    for k, (name, color) in enumerate(zip(shown, [BLUE, ORANGE, GREEN])):
        ax.bar(pos + (k - 1) * 0.27, rates[name], width=0.25, color=color, label=shown[name])
    ax.set_xticks(pos)
    ax.set_xticklabels(list(groups), rotation=20)
    ax.set_ylabel("test error rate")
    ax.set_title("(b) Task B: error rate by group (ours)")
    ax.legend()
    save(fig, "error_analysis.png")


def main() -> None:
    start = time.perf_counter()

    # 1. data -------------------------------------------------------------
    d = load_data()
    plot_eda(d)

    # 2. our tree: Task A and Task B, tuned on validation -----------------
    reg = study_tree_regression(d)
    clf = study_tree_classification(d)
    plot_tree_curves(reg, clf)

    # bonus: our Ridge regression, Task A baseline -------------------------
    ridge = study_ridge(d)

    # 3. our SVM: Task B, tuned on validation -----------------------------
    svm = study_svm(d)
    plot_svm_curves(svm)

    # 4. final configs (validation evidence only) -> test + CV -------------
    cfg = {
        "reg_depth": reg["max_depth"], "reg_leaf": reg["min_samples_leaf"],
        "ridge_lambda": ridge["lambda"],
        "clf_criterion": clf["criterion"], "clf_depth": clf["max_depth"],
        "clf_leaf": clf["min_samples_leaf"],
        "svm_lambda": svm["lambda"], "gamma": svm["gamma"], "rff_lambda": svm["rff_lambda"],
    }
    task_a, task_b = evaluate_on_test(d, cfg)
    cv_a, cv_b = cross_validate(d, cfg)

    # 5. tables, comparison, figures, error analysis -----------------------
    results_tables(task_a, task_b, cv_a, cv_b, d)
    compare_with_sklearn(task_a, task_b, d)
    plot_confusion(task_b, d)
    plot_importance_and_weights(task_a, task_b, d)
    error_analysis(task_a, task_b, d)

    print(f"\nTotal run time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
