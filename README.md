# MLE-AI-201 Machine Learning — Final Project

A **decision tree** and a **soft-margin SVM (Pegasos)** written from scratch
in NumPy, applied to the **bina.az** real-estate sale listings and compared
with scikit-learn.

- **Task A — Regression:** predict the sale price (we model `log(price)`).
- **Task B — Classification:** price tier, premium (1) vs standard (0),
  split at the median price of the **training** set.

Our models: `DecisionTree` (Gini / entropy / MSE) in `src/decision_tree.py`,
`PegasosSVM` + `RandomFourierFeatures` in `src/svm.py`. scikit-learn is used
only for the baselines and the sanity tests.

## Headline results (test set)
57,215 listings after cleaning; train / val / test = 40,050 / 8,582 / 8,583.
Hyperparameters chosen on validation only; CV = 5-fold on train+val.

| Model | Task A RMSE (log price) | Task A R² | Task B F1 | Task B ROC-AUC | CV F1 (mean ± std) |
|-------|------|------|------|------|------|
| **Ours: decision tree** | **0.2422** | **0.851** | **0.925** | 0.930 | 0.922 ± 0.002 |
| sklearn decision tree   | 0.2424 | 0.851 | 0.924 | 0.929 | 0.921 ± 0.001 |
| **Ours: linear SVM (Pegasos)** | — | — | 0.888 | 0.948 | 0.880 ± 0.001 |
| sklearn SGDClassifier (hinge)  | — | — | 0.886 | 0.948 | 0.876 ± 0.003 |
| **Ours: RFF SVM (Pegasos)**    | — | — | 0.892 | 0.964 | 0.878 ± 0.018 |
| sklearn SVC (RBF)              | — | — | 0.906 | 0.972 | 0.903 ± 0.003 |

Final configs: Task A tree `mse, max_depth=12, min_samples_leaf=10`;
Task B tree `entropy, max_depth=20`; linear SVM `lambda=1e-3`;
RFF SVM `gamma=0.03, lambda=1e-5, D=500`.

## Reproduce everything (one command)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# download the dataset into data/ (see data/README.md), then:
python -m src.run_all
```
`run_all` prints every number used in the report and saves every figure to
`report/figures/`. All randomness uses the fixed seed 42
(`data_prep.SEED`). Runtime: about 10–12 minutes on a laptop CPU (most of it
is the exact RBF `SVC` baseline inside the cross-validation).

Sanity tests (our tree vs sklearn on a tiny fully-determined example, SVM
loss decreasing and matching `SGDClassifier`, metrics vs sklearn):
```bash
python -m tests.test_sanity        # or: python -m pytest tests
```

## Repo layout
```
.
├── README.md               # this file
├── requirements.txt        # pinned versions
├── .gitignore
├── contribution_report.md  # who did what
├── data/                   # dataset lives here (not committed — see data/README.md)
├── src/
│   ├── data_prep.py        #   loading, cleaning, split, tier label, scaling
│   ├── decision_tree.py    #   DecisionTree (Gini / entropy / MSE), from scratch
│   ├── svm.py              #   PegasosSVM + RandomFourierFeatures, from scratch
│   ├── evaluate.py         #   metrics (NumPy) + comparison helpers
│   └── run_all.py          #   ONE command reproduces every number/figure
├── tests/
│   └── test_sanity.py      #   our models / metrics vs scikit-learn
├── report/
│   ├── report.tex          #   IEEE-style report
│   └── figures/            #   created by run_all
└── presentation/
    └── slides.tex          #   demo slides (compile to slides.pdf)
```

## Pipeline in short
1. **Cleaning** (`data_prep.clean`): drop the leakage columns `unit_price`,
   `total_price` and all identifiers; translate the Azerbaijani column
   names; keep AZN prices; remove repeated listings; keep flats and houses;
   parse `"145 m²"`, `"1.3 sot"`, `"7 / 9"`; fixed outlier rules.
2. **Features** (`data_prep.make_features`): 10 numeric columns, one-hot
   category (3) and one-hot top-20 districts → 33 features.
3. **Split** (`data_prep.train_val_test_split`): 70 / 15 / 15, the tier
   threshold comes from the training prices only, validation and test are
   stratified on the tier. Scaling uses training statistics only.
4. **Tuning** on the validation split only; test is used once at the end.
5. **Benchmark** vs `DecisionTreeRegressor`, `DecisionTreeClassifier`,
   `SGDClassifier(hinge)` and `SVC(rbf)` with matched settings, plus 5-fold
   cross-validation (mean ± std) on train+val.

## Report and slides
`report/report.tex` and `presentation/slides.tex` compile with `pdflatex`
(or Overleaf) after `run_all` has created the figures.
