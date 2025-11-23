# src/train.py
"""
Train a simple RandomForest baseline for predictive maintenance.

Usage:
    python src\train.py          # uses default data path
    python src\train.py path/to/data.csv

Saves:
    models/baseline.pkl  -- sklearn Pipeline with scaler + model
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# local features helper
from src.features import load_data, make_window_features

def main(data_path: str = "data/dataset_forced_failures.csv"):
    print("Loading data from:", data_path)
    df = load_data(data_path)

    print("Making window features (60s windows, 60s step)...")
    X, y, meta = make_window_features(df, window_sec=60, step_sec=30)
    print("Feature shape:", X.shape, "Labels shape:", y.shape)
    print("Label distribution:\n", y.value_counts().to_dict())

    # keep at least two classes
    if y.nunique() < 2:
        raise RuntimeError("Not enough label variety for training. Need at least two classes.")

    # simple train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # pipeline: scale -> RF
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))
    ])

    print("Training RandomForest...")
    pipe.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # feature importances (from RF inside pipeline)
    try:
        rf = pipe.named_steps['rf']
        importances = rf.feature_importances_
        feat_imp = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)[:15]
        print("\\nTop feature importances:")
        for f, imp in feat_imp:
            print(f"{f}: {imp:.4f}")
    except Exception as e:
        print("Could not compute feature importances:", e)

    # ensure models/ exists
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline.pkl"
    joblib.dump(pipe, out_path)
    print(f"Saved pipeline to: {out_path}")

if __name__ == "__main__":
    datapath = sys.argv[1] if len(sys.argv) > 1 else "data/dataset_forced_failures.csv"
    main(datapath)
