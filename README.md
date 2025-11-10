# Phishing Detection — Paper Reproduction

This repository reproduces the results from the paper **"Phishing Website Detection using Machine Learning"** (UCI dataset).

### Dataset
The dataset comes from the UCI Phishing Websites dataset and contains 30 features + 1 label (`-1`, `0`, `1`).

### Approach
We replicate the paper’s pipeline:
- Label mapping: `{-1, 0} → 0 (phish)`, `1 → 1 (legit)`
- Normalization: MinMaxScaler to [0,1]
- Models: SVM (RBF), ANN (MLP), Decision Tree, Random Forest
- Evaluation: 5-fold Stratified CV on training data, metrics on test and unseen val sets

### Files
- `phishing_reproduction.py` — full reproducible script
- `phishing_reproduction.ipynb` — notebook version
- `data/` — contains `train.csv`, `test.csv`, and `val_unseen.csv`
- `outputs/` — stores generated metrics tables

### How to Run

#### Option 1 — Python script
```bash
python phishing_reproduction.py   --train data/train.csv   --test data/test.csv   --val data/val_unseen.csv   --outdir outputs
```

#### Option 2 — Jupyter Notebook
Run all cells in `phishing_reproduction.ipynb`.

### Requirements
See `requirements.txt`.
