#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42

LABEL_CANDIDATES = [
    "Result","Label","label","target","Target","Class","class","y","Outcome","outcome"
]

def detect_label_column(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: last column if it looks categorical with 2-3 unique vals
    last = df.columns[-1]
    nunq = df[last].nunique(dropna=True)
    if nunq in (2,3):
        return last
    # try any column with small unique set
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if nun in (2,3):
            return c
    raise ValueError("Could not detect label column. Please rename your label to one of: " + ", ".join(LABEL_CANDIDATES))

def load_dataset(path: Path):
    df = pd.read_csv(path)
    y_col = detect_label_column(df)
    X = df.drop(columns=[y_col])
    y_raw = df[y_col].copy()
    # Ensure numeric
    y_raw = pd.to_numeric(y_raw, errors='coerce')
    if y_raw.isna().any():
        raise ValueError(f"Label column '{y_col}' contains non-numeric values.")
    # Map {-1,0} -> 0 (phish), {1}->1 (legit)
    y = y_raw.replace({-1:0, 0:0, 1:1})
    # Coerce all features to numeric (just in case)
    X = X.apply(pd.to_numeric, errors='coerce')
    # Fill any leftover NaNs with 0 (UCI encoding should prevent this, but safe)
    X = X.fillna(0)
    return X, y, y_col

def tpr_tnr(y_true, y_pred, positive_label=1):
    # Confusion matrix with labels [0,1] order
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    # TPR: sensitivity/recall for positive class
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # TNR: specificity for positive class (i.e., recall for negative class)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return tpr, tnr, cm

def cv_and_fit(model_name, estimator, X, y, n_splits=5):
    """Run 5-fold stratified CV (on train), then fit on full train for later test eval."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', estimator)
    ])
    scoring = {
        'accuracy':'accuracy',
        'precision':'precision',
        'recall':'recall',
        'f1':'f1'
    }
    cv_res = cross_validate(pipe, X, y, scoring=scoring, cv=skf, n_jobs=-1, return_train_score=False)
    # Fit on full training set
    pipe.fit(X, y)
    cv_summary = { f'cv_{k}_mean': float(np.mean(v)) for k, v in cv_res.items() if k.startswith('test_') }
    return pipe, cv_summary

def evaluate(pipe, X, y, tag):
    y_pred = pipe.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    tpr, tnr, cm = tpr_tnr(y, y_pred)
    report = classification_report(y, y_pred, digits=4, zero_division=0)
    res = {
        'set': tag,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'tpr_pos1': float(tpr),
        'tnr_pos1': float(tnr),
        'cm_tn': int(cm[0,0]), 'cm_fp': int(cm[0,1]), 'cm_fn': int(cm[1,0]), 'cm_tp': int(cm[1,1]),
        'classification_report': report
    }
    return res

def build_models():
    models = {
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'ANN_MLP': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=RANDOM_STATE),
        'DecisionTree': DecisionTreeClassifier(criterion='gini', splitter='best', random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=100, criterion='gini', random_state=RANDOM_STATE, n_jobs=-1)
    }
    return models

def main():
    parser = argparse.ArgumentParser(description="Reproduce phishing detection results with classic ML models.")
    parser.add_argument("--train", type=Path, default=Path("/mnt/data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("/mnt/data/test.csv"))
    parser.add_argument("--val", type=Path, default=Path("/mnt/data/val_unseen.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("./outputs"))
    parser.add_argument("--cv_splits", type=int, default=5)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    X_train, y_train, ycol_train = load_dataset(args.train)
    X_test, y_test, ycol_test = load_dataset(args.test)
    X_val, y_val, ycol_val = load_dataset(args.val)

    # Ensure same columns/order across splits
    missing_in_test = [c for c in X_train.columns if c not in X_test.columns]
    missing_in_val = [c for c in X_train.columns if c not in X_val.columns]
    if missing_in_test or missing_in_val:
        raise ValueError(f"Feature mismatch: missing in test: {missing_in_test}, missing in val: {missing_in_val}")
    X_test = X_test[X_train.columns]
    X_val = X_val[X_train.columns]

    models = build_models()

    rows = []
    detailed_reports = []

    for name, est in models.items():
        print("="*80)
        print(f"Model: {name}")
        pipe, cv_summary = cv_and_fit(name, est, X_train, y_train, n_splits=args.cv_splits)
        # Evaluate on test and val
        test_res = evaluate(pipe, X_test, y_test, tag="test")
        val_res = evaluate(pipe, X_val, y_val, tag="val_unseen")

        # Collect row for summary
        row = {
            'model': name,
            **{k.replace('cv_test_','cv_'): v for k, v in cv_summary.items()},  # rename keys to shorter
            'test_accuracy': test_res['accuracy'],
            'test_precision': test_res['precision'],
            'test_recall': test_res['recall'],
            'test_f1': test_res['f1'],
            'val_accuracy': val_res['accuracy'],
            'val_precision': val_res['precision'],
            'val_recall': val_res['recall'],
            'val_f1': val_res['f1'],
        }
        rows.append(row)

        # Print details
        print("CV (5-fold) means:", {k: round(v,4) for k,v in cv_summary.items()})
        print("[TEST] acc={:.4f} prec={:.4f} rec={:.4f} f1={:.4f} TPR={:.4f} TNR={:.4f}".format(
            test_res['accuracy'], test_res['precision'], test_res['recall'], test_res['f1'], test_res['tpr_pos1'], test_res['tnr_pos1']
        ))
        print("Confusion matrix [ [tn, fp], [fn, tp] ]:", [[test_res['cm_tn'], test_res['cm_fp']], [test_res['cm_fn'], test_res['cm_tp']]])
        print("Classification report (TEST):\n", test_res['classification_report'])

        print("[VAL]  acc={:.4f} prec={:.4f} rec={:.4f} f1={:.4f} TPR={:.4f} TNR={:.4f}".format(
            val_res['accuracy'], val_res['precision'], val_res['recall'], val_res['f1'], val_res['tpr_pos1'], val_res['tnr_pos1']
        ))
        print("Confusion matrix [ [tn, fp], [fn, tp] ]:", [[val_res['cm_tn'], val_res['cm_fp']], [val_res['cm_fn'], val_res['cm_tp']]])
        print("Classification report (VAL):\n", val_res['classification_report'])

        detailed_reports.append({
            'model': name,
            'test_report': test_res['classification_report'],
            'val_report': val_res['classification_report']
        })

    # Save summary table
    summary_df = pd.DataFrame(rows)
    summary_path = args.outdir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    # Save a nicely sorted version by test_f1 desc
    summary_sorted_path = args.outdir / "summary_metrics_sorted.csv"
    summary_df.sort_values("test_f1", ascending=False).to_csv(summary_sorted_path, index=False)

    print("="*80)
    print("Saved summary to:", summary_path)
    print("Saved sorted summary to:", summary_sorted_path)

if __name__ == "__main__":
    main()
