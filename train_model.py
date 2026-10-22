#!/usr/bin/env python3
"""Train CICIDS2017 multi-class XGBoost and Isolation Forest models locally.

Usage: python train_model.py --data-dir ./CICIDS2017 --output-dir artifacts
Training is intended for a local machine with sufficient RAM, not Streamlit Cloud.
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

SEED = 42

def normalize(name):
    return re.sub(r"\\s+", "_", str(name).replace("\\ufeff", "").strip())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--trees", type=int, default=350)
    args = parser.parse_args()
    files = sorted(args.data_dir.rglob("*.csv")) if args.data_dir.is_dir() else []
    if not files:
        parser.error(f"No CSV files found under {args.data_dir}")
    frames = []
    for path in files:
        print(f"Reading {path}", flush=True)
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [normalize(c) for c in frame.columns]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    label_col = next((c for c in data.columns if c.lower() in {"label", "class", "attack"}), None)
    if label_col is None:
        parser.error("CICIDS2017 target column not found; expected Label.")
    y = data[label_col].astype("string").str.strip()
    X = data.drop(columns=[label_col]).copy()
    X.columns = [normalize(c) for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = y.notna() & y.ne("")
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns).replace([np.inf, -np.inf], 0).fillna(0)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(y.astype(str))
    classes = [str(c) for c in encoder.classes_]
    if len(classes) < 2:
        parser.error("Training data must contain at least two classes.")
    if len(classes) != 15:
        print(f"WARNING: dataset has {len(classes)} classes, not 15: {classes}")
    selector = ExtraTreesClassifier(n_estimators=100, class_weight="balanced",
                                    max_features="sqrt", n_jobs=-1, random_state=SEED)
    selector.fit(X, labels)
    ranked = np.argsort(selector.feature_importances_)[::-1]
    features = [str(X.columns[i]) for i in ranked[:min(40, X.shape[1])]]
    X = X[features].astype(np.float32)
    stratify = labels if np.min(np.bincount(labels)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=.2, random_state=SEED, stratify=stratify)
    model = XGBClassifier(objective="multi:softprob", num_class=len(classes),
        n_estimators=args.trees, max_depth=6, learning_rate=.08, subsample=.85,
        colsample_bytree=.85, reg_lambda=2, min_child_weight=2, tree_method="hist",
        eval_metric="mlogloss", n_jobs=max(1, min(4, os.cpu_count() or 1)),
        random_state=SEED)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(X), min(100_000, len(X)), replace=False)
    ood = IsolationForest(n_estimators=150, contamination="auto",
        max_samples=min(2048, len(sample_idx)),
        n_jobs=max(1, min(4, os.cpu_count() or 1)), random_state=SEED)
    ood.fit(X.iloc[sample_idx])
    print(classification_report(y_test, model.predict(X_test), labels=np.arange(len(classes)),
                               target_names=classes, zero_division=0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "xgboost_model.joblib", compress=3)
    joblib.dump(ood, args.output_dir / "isolation_forest.joblib", compress=3)
    joblib.dump(encoder, args.output_dir / "label_encoder.joblib", compress=3)
    joblib.dump(imputer, args.output_dir / "imputer.joblib", compress=3)
    (args.output_dir / "model_metadata.json").write_text(json.dumps({
        "feature_columns": features, "label_column": label_col, "classes": classes,
        "n_classes": len(classes), "selected_feature_count": len(features),
        "dataset": "CICIDS2017", "random_state": SEED
    }, indent=2))
    print(f"Artifacts saved to {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
