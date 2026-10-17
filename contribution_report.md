# Contribution Report

Compiled from the git history of `main` (commits `0988543` … `ab16792`,
16–17 Oct 2026). Commit hashes and PR numbers below refer to that history.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Shahbazli, Alp Eren         | Alpachino012  | Repo owner and setup; Pegasos SVM + random Fourier features (`src/svm.py`); experiment pipeline (`src/run_all.py`); top-level `README.md` |
| Isgandarov, Murad      | murad15       | Decision tree from scratch (`src/decision_tree.py`); bonus Ridge regression (`src/ridge.py`) and its integration in `run_all.py`; PR review/merges |
| Huseynzada, Mehdi      | Universoul23  | Data loading, cleaning, features, tier label, split and scaling (`src/data_prep.py`); dataset documentation (`data/README.md`) |
| Najafov, Malik         | maliknajaff   | Metrics and evaluation helpers from scratch (`src/evaluate.py`); sanity tests against scikit-learn (`tests/test_sanity.py`) |

## Per-member summary
**Shahbazli, Alp Eren —** Created the GitHub repository, the project skeleton,
`.gitignore` and the pinned `requirements.txt` (`0ac8374`, `cbd4de1`), and
imported the course starter pack (`ddaa068`). Implemented `PegasosSVM`
(mini-batch Pegasos with the bias folded into `w`, three learning-rate
schedules, per-epoch objective history, margin and support-vector helpers)
and `RandomFourierFeatures` for the RBF kernel approximation in `src/svm.py`
(`bc81b5d`, PR #2). Wrote the experiment orchestrator `src/run_all.py`
(`3444b39`): validation-only hyperparameter grids, EDA figure, tree and SVM
tuning curves, scikit-learn baselines with matched settings, 5-fold CV,
results tables, confusion matrices, importances and error analysis. Wrote
the top-level `README.md` with the repo layout, one-command reproduction and
results table (`905c34f`).

**Isgandarov, Murad —** Implemented the from-scratch `DecisionTree` in
`src/decision_tree.py` over six commits (PR #3): Gini, entropy and MSE
impurity (`c9bf2cb`, `a22a0d7`); a vectorised best-split search using
cumulative sums over each sorted feature (`3e5c3f8`); recursive
`_grow`/`fit` with the stopping rules and impurity-based feature importances
(`aecb2ed`); and `predict`/`predict_proba` plus `get_depth`/`get_n_leaves`
for the scikit-learn comparison (`96a8734`, `6b7295d`). Added the bonus Ridge
regression (`src/ridge.py`: normal equations solved with `np.linalg.solve`,
intercept not penalised) and wired it into `run_all.py`: λ sweep on
validation, Task A results row and a coefficient check against sklearn's
`Ridge` (`bdf838f`, PR #6). Reviewed and merged PRs #3, #5, #6, #7 and #8,
and removed the starter `report/` folder and `PROJECT_STATEMENT.pdf` from
the repo (`0186949`, `5d027af`).

**Huseynzada, Mehdi —** Implemented the whole data pipeline in
`src/data_prep.py` (PR #4). This covers the leakage, identifier and
Azerbaijani→English column definitions and raw loading (`e59a2d3`), and the
cleaning: AZN check, two-stage de-duplication, residential categories only,
parsing of `"145 m²"`, `"1.3 sot"` and `"7 / 9"`, missing-value rules and
fixed outlier limits (`2f3ab31`). It also covers the feature matrix
(one-hot category and top-20 districts), the train-median tier label, a
70/15/15 split stratified by tier in which only training prices set the
threshold, and train-statistics standardisation (`f7764a9`). Documented the
dataset source, version and pitfalls in `data/README.md` (`bad222d`) and
fixed the bugs found when testing the pipeline (`cb3fc9f`).

**Najafov, Malik —** Implemented the metrics module `src/evaluate.py` from
scratch (`49889e2`, PR #5). Task A: RMSE, MAE and R², with a report on both
the log-price and the AZN scale. Task B: confusion matrix,
precision/recall/F1 and a rank-based (Mann–Whitney) ROC-AUC that averages
tied ranks, which tree probabilities need. Also wrote the k-fold index
generator, the mean ± std formatter and the comparison-table printer that
`run_all.py` uses. Wrote the sanity-test suite `tests/test_sanity.py`
(`089f6ce`, `297d994`, `d4442ba`, `8116b25`, PR #7), with nine tests that
compare our code with scikit-learn. On an 8-point toy set where sklearn
builds the same tree for every seed, our tree must reproduce sklearn's
splits exactly for Gini, entropy and MSE. `predict_proba` and each stopping
rule are also checked against sklearn. Pegasos must lower its objective,
agree with `SGDClassifier(hinge)` and report support vectors that match its
margin. The metrics are checked against `sklearn.metrics`, including
ROC-AUC with ties, and a test confirms the tier threshold comes from the
training prices only.

## Git statistics
Counted with `git log --numstat` and `git blame -w` on `main`. Merge
commits are excluded; binary files (`PROJECT_STATEMENT.pdf`) are not counted
as lines. Work done outside this repository, such as the written report and
the slides, is not in the git history and is not counted here.

### Raw numbers (what GitHub shows)
| Member | Commits (excl. merges) | Merges | Lines added | Share of lines added |
|--------|-----------------------:|-------:|------------:|---------------------:|
| Shahbazli, Alp Eren    |  7 | 2 | 1,368 | 49.9% |
| Isgandarov, Murad | 10 | 5 |   477 | 17.2% |
| Huseynzada, Mehdi |  5 | 1 |   467 |  16.8% |
| Najafov, Malik    |  5 | 1 |   430 | 15.2% |
| **Total**         | 27 | 9 | 2,743 | 100% |

### Adjusted for the starter pack
Alp Eren pushed the instructor-provided starter pack (`ddaa068`: function
stubs that only raise `NotImplementedError`, the report template, the
folder READMEs and the template of this file). He also made GitHub's
auto-generated "Initial commit" (`0988543`). Together these are 645 lines
that no team member wrote, about a third of his raw count. Removing them
gives:

| Member |  Lines in final code (`git blame`), excl. starter | Share |
|--------|---------------------------:|------:|-------------------------------------------------:|------:|
| Shahbazli, Alp Eren    | 608 | 34.1% |
| Isgandarov, Murad  | 435 | 24.4% |
| Huseynzada, Mehdi  | 358 | 20.0% |
| Najafov, Malik    | 383 | 21.5% |
| **Total**         | 1,784 | 100% |


## Rough effort split
| Member | Approx. % |
|--------|-----------|
| Shahbazli, Alp Eren    | 31% |
| Isgandarov, Murad | 23% |
| Huseynzada, Mehdi | 23% |
| Najafov, Malik    | 23% |

**Imbalance and why.** Removing the starter pack cuts Alp Eren's share from
50% to about 34% of the final code. Most of the remaining lead comes from
`src/run_all.py` (729 of its 806 lines). Line counts still overstate his
share of the effort. `run_all.py` is mostly orchestration (tuning loops,
printing and plotting), while the from-scratch modules are short but dense.
Each of those four modules (tree, SVM, data pipeline, metrics) had one
clear owner.The other three contributed about equally. Murad wrote
the decision tree, the bonus Ridge regression and its integration, and did
the PR merges. Mehdi wrote the data pipeline and its documentation and
debugged it. Malik wrote the metrics module and the sanity-test suite that
checks everyone's models against scikit-learn.
