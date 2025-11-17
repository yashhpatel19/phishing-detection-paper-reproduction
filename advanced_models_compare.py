#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare RF, XGBoost, LightGBM, PyTorch MLP
using train.csv, test.csv, val_unseen.csv.

- 5-fold Stratified CV on train
- Train on full train
- Evaluate on test and val_unseen
- Save summary to advanced_models_summary_sorted.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

# Optional external libs
# We wrap imports in broad try/except so script keeps running even if something is broken.

try:
    from xgboost import XGBClassifier
except Exception as e:
    print("WARNING: XGBoost is not usable on this system, skipping XGBoost model.")
    print("Reason:", e)
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception as e:
    print("WARNING: LightGBM is not usable on this system, skipping LightGBM model.")
    print("Reason:", e)
    LGBMClassifier = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    print("WARNING: PyTorch is not usable on this system, skipping TorchMLP model.")
    print("Reason:", e)
    torch = None

RANDOM_STATE = 42

LABEL_CANDIDATES = [
    "Result","Label","label","target","Target","Class","class","y","Outcome","outcome"
]

# ---------------------------------------------------------------------
# Data helpers (same mapping as your original script)
# ---------------------------------------------------------------------
def detect_label_column(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    last = df.columns[-1]
    nunq = df[last].nunique(dropna=True)
    if nunq in (2, 3):
        return last
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if nun in (2, 3):
            return c
    raise ValueError(
        "Could not detect label column. Please rename label to one of: "
        + ", ".join(LABEL_CANDIDATES)
    )

def load_dataset(path: Path):
    df = pd.read_csv(path)
    y_col = detect_label_column(df)
    X = df.drop(columns=[y_col])
    y_raw = df[y_col].copy()
    y_raw = pd.to_numeric(y_raw, errors="coerce")
    if y_raw.isna().any():
        raise ValueError(f"Label column '{y_col}' contains non-numeric values.")

    # Map {-1,0} -> 0, {1} -> 1  (same as original code)
    y = y_raw.replace({-1: 0, 0: 0, 1: 1})

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)

    return X.values, y.values, y_col

def tpr_tnr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return tpr, tnr, cm

def evaluate_model_fscore(model, X, y, tag="test"):
    """Evaluate fitted model on (X, y)."""
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    tpr, tnr, cm = tpr_tnr(y, y_pred)
    return {
        "set": tag,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tpr_pos1": float(tpr),
        "tnr_pos1": float(tnr),
        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
    }

# ---------------------------------------------------------------------
# PyTorch MLP classifier
# ---------------------------------------------------------------------
class TorchMLPClassifier:
    def __init__(self, input_dim, hidden_dims=(64, 32), lr=1e-3,
                 max_epochs=40, batch_size=64, random_state=RANDOM_STATE, device=None):
        if torch is None:
            raise ImportError("PyTorch is not installed or usable.")
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(self.random_state)
        self._build_model()

    def _build_model(self):
        layers = []
        in_dim = self.input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Dropout(p=0.2))
        layers.append(nn.Linear(in_dim, 2))  # 2-class logits
        self.model = nn.Sequential(*layers).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(ds)
            # Optional debug:
            # print(f"Epoch {epoch+1}/{self.max_epochs}, loss={epoch_loss:.4f}")
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# ---------------------------------------------------------------------
# CV helper: run CV on train set
# ---------------------------------------------------------------------
def cv_metrics(builder, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    accs, precs, recs, f1s = [], [], [], []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = builder()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        accs.append(accuracy_score(y_va, y_pred))
        precs.append(precision_score(y_va, y_pred, zero_division=0))
        recs.append(recall_score(y_va, y_pred, zero_division=0))
        f1s.append(f1_score(y_va, y_pred, zero_division=0))
    return {
        "cv_accuracy": float(np.mean(accs)),
        "cv_precision": float(np.mean(precs)),
        "cv_recall": float(np.mean(recs)),
        "cv_f1": float(np.mean(f1s)),
    }

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare RF, XGBoost, LightGBM, TorchMLP using train/test/val_unseen."
    )
    parser.add_argument("--train", type=Path, required=True, help="Path to train.csv")
    parser.add_argument("--test", type=Path, required=True, help="Path to test.csv")
    parser.add_argument("--val", type=Path, required=True, help="Path to val_unseen.csv")
    parser.add_argument("--outdir", type=Path, default=Path("./outputs_advanced_all3"))
    parser.add_argument("--cv_splits", type=int, default=5)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    X_train, y_train, ycol_train = load_dataset(args.train)
    X_test, y_test, ycol_test = load_dataset(args.test)
    X_val, y_val, ycol_val = load_dataset(args.val)

    print(f"Train label col: {ycol_train}, shape={X_train.shape}, label counts={np.bincount(y_train)}")
    print(f"Test  label col: {ycol_test}, shape={X_test.shape}, label counts={np.bincount(y_test)}")
    print(f"Val   label col: {ycol_val}, shape={X_val.shape}, label counts={np.bincount(y_val)}")

    # Scale features once and use for all models
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_val_sc = scaler.transform(X_val)
    input_dim = X_train_sc.shape[1]

    # Model builders
    def build_rf():
        return RandomForestClassifier(
            n_estimators=300,
            criterion="gini",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if XGBClassifier is None:
        def build_xgb():
            raise RuntimeError("XGBClassifier unavailable")
    else:
        def build_xgb():
            return XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist"
            )

    if LGBMClassifier is None:
        def build_lgbm():
            raise RuntimeError("LGBMClassifier unavailable")
    else:
        def build_lgbm():
            return LGBMClassifier(
                n_estimators=400,
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

    if torch is None:
        def build_torch():
            raise RuntimeError("TorchMLP unavailable")
    else:
        def build_torch():
            return TorchMLPClassifier(
                input_dim=input_dim,
                hidden_dims=(64, 32),
                lr=1e-3,
                max_epochs=40,
                batch_size=64,
                random_state=RANDOM_STATE
            )

    model_builders = {
        "RandomForest": build_rf,
        "XGBoost": build_xgb,
        "LightGBM": build_lgbm,
        "TorchMLP": build_torch,
    }

    rows = []

    for name, builder in model_builders.items():
        print("=" * 80)
        print(f"Model: {name}")
        print("=" * 80)
        try:
            # CV on train
            cv_summary = cv_metrics(builder, X_train_sc, y_train, n_splits=args.cv_splits)
            print("CV summary (means):", {k: round(v, 4) for k, v in cv_summary.items()})

            # Fit on full train
            model = builder()
            model.fit(X_train_sc, y_train)

            # Evaluate on test and val
            test_res = evaluate_model_fscore(model, X_test_sc, y_test, tag="test")
            val_res = evaluate_model_fscore(model, X_val_sc, y_val, tag="val_unseen")

            print("[TEST] acc={:.4f} prec={:.4f} rec={:.4f} f1={:.4f}".format(
                test_res["accuracy"], test_res["precision"], test_res["recall"], test_res["f1"]
            ))
            print("[VAL ] acc={:.4f} prec={:.4f} rec={:.4f} f1={:.4f}".format(
                val_res["accuracy"], val_res["precision"], val_res["recall"], val_res["f1"]
            ))

            rows.append({
                "model": name,
                **cv_summary,
                "test_accuracy": test_res["accuracy"],
                "test_precision": test_res["precision"],
                "test_recall": test_res["recall"],
                "test_f1": test_res["f1"],
                "val_accuracy": val_res["accuracy"],
                "val_precision": val_res["precision"],
                "val_recall": val_res["recall"],
                "val_f1": val_res["f1"],
            })
        except Exception as e:
            print(f"Skipping model {name} due to error: {e}")

    summary_df = pd.DataFrame(rows)
    summary_path = args.outdir / "advanced_models_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    summary_sorted_path = args.outdir / "advanced_models_summary_sorted.csv"
    summary_df.sort_values("test_f1", ascending=False).to_csv(summary_sorted_path, index=False)

    print("=" * 80)
    print("Saved summary to:", summary_path)
    print("Saved sorted summary to:", summary_sorted_path)
    print("Sorted by test_f1 (descending):")
    print(summary_df.sort_values("test_f1", ascending=False))


if __name__ == "__main__":
    main()
