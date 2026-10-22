#!/usr/bin/env python3
"""Train the ThreatLens CICIDS2017 multiclass and out-of-distribution models.

Example:
  python train_model.py --data-dir . --output-dir artifacts

Accepts Parquet dataset files. Environment and generated directories are excluded
during recursive discovery.

Accepts CSV and Parquet files. If both formats exist under the input directory,
Parquet files are preferred to avoid training twice on duplicate exports.

Train locally on a machine with enough RAM. The Streamlit app consumes the saved
artifacts and will reject inference datasets whose columns differ from metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
TARGET_NAMES = {"label", "class", "attack"}
NON_FEATURE_NAMES = {
    "flow_id", "src_ip", "source_ip", "source_address", "id_orig_h",
    "dst_ip", "dest_ip", "destination_ip", "destination_address", "id_resp_h",
    "timestamp", "time", "date",
}


def normalize(name: object) -> str:
    """Normalize common CSV header noise consistently for saved metadata."""
    value = str(name).replace("\ufeff", "").strip()
    return re.sub(r"\s+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--trees", type=int, default=350)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data_dir}")

    # Search only for Parquet files. Exclude virtual environments and generated files
    # so package test datasets are never mistaken for CICIDS2017 input.
    excluded_dirs = {
        ".venv", "venv", "node_modules", ".git", "__pycache__",
        ".pytest_cache", "site-packages", "artifacts",
    }

    def is_allowed_dataset_file(path: Path) -> bool:
        try:
            relative_parts = path.relative_to(args.data_dir).parts[:-1]
        except ValueError:
            return False
        return not any(
            part in excluded_dirs or part.startswith(".")
            for part in relative_parts
        )

    files = sorted(
        path
        for path in args.data_dir.rglob("*.parquet")
        if is_allowed_dataset_file(path)
    )
    if not files:
        parser.error(f"No Parquet files found under {args.data_dir}")

    print(f"Found {len(files)} Parquet dataset file(s).", flush=True)

    frames = []
    for file in files:
        print(f"Reading {file}", flush=True)
        try:
            frame = pd.read_parquet(file)
        except (ValueError, OSError, ImportError) as exc:
            parser.error(f"Could not read Parquet file {file}: {exc}")
        frame.columns = [normalize(column) for column in frame.columns]
        frames.append(frame)

    # Preserve the CSV pipeline's combined-data behavior.
    data = pd.concat(frames, ignore_index=True, sort=False)
    label_col = next((c for c in data.columns if c.strip().lower() in TARGET_NAMES), None)
    if label_col is None:
        parser.error("Target column not found; expected Label, Class, or Attack.")

    y = data[label_col].astype("string").str.strip()
    X = data.drop(columns=[label_col]).copy()
    X.columns = [normalize(c) for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    drop_names = [c for c in X.columns if c.strip().lower() in NON_FEATURE_NAMES]
    if drop_names:
        print("Excluding identifier/time columns:", ", ".join(drop_names), flush=True)
        X = X.drop(columns=drop_names)

    # CICIDS exports include string identifiers. Only measured numeric telemetry
    # may become model features; strings are not encoded as invented numeric values.
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    valid = y.notna() & y.ne("")
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    if len(X) < 10:
        parser.error("Too few valid labeled rows to train a meaningful model.")
    if X.shape[1] < 40:
        parser.error(f"Only {X.shape[1]} usable numeric features found; the app requires at least 40.")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(y.astype(str))
    classes = [str(c) for c in encoder.classes_]
    if len(classes) < 2:
        parser.error("Training data must contain at least two classes.")
    if len(classes) != 15:
        print(f"WARNING: dataset has {len(classes)} classes, not 15: {classes}", flush=True)

    counts = np.bincount(labels)
    if not len(counts) or int(counts.min()) < 2:
        parser.error("Each class needs at least two labeled rows for a stratified train/test split.")

    # Split before fitting the imputer or feature selector to avoid hold-out leakage.
    train_indices, test_indices = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=SEED, stratify=labels
    )
    X_train_raw = X.iloc[train_indices].copy()
    X_test_raw = X.iloc[test_indices].copy()
    y_train, y_test = labels[train_indices], labels[test_indices]

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train_raw), columns=X.columns, index=X_train_raw.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test_raw), columns=X.columns, index=X_test_raw.index
    )
    if X_train_imputed.isna().any().any() or X_test_imputed.isna().any().any():
        parser.error("Numeric preprocessing left missing values; inspect the source dataset.")
    X_train_imputed = X_train_imputed.astype(np.float32)
    X_test_imputed = X_test_imputed.astype(np.float32)

    # Feature selection uses training rows and labels only.
    selector = ExtraTreesClassifier(
        n_estimators=100, class_weight="balanced", max_features="sqrt",
        n_jobs=max(1, min(4, os.cpu_count() or 1)), random_state=SEED,
    )
    selector.fit(X_train_imputed, y_train)
    ranked = np.argsort(selector.feature_importances_)[::-1]
    features = [str(X.columns[i]) for i in ranked[:40]]
    X_train = X_train_imputed[features]
    X_test = X_test_imputed[features]

    model = XGBClassifier(
        objective="multi:softprob", num_class=len(classes), n_estimators=args.trees,
        max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=2, min_child_weight=2, tree_method="hist", eval_metric="mlogloss",
        n_jobs=max(1, min(4, os.cpu_count() or 1)), random_state=SEED,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Fit OOD detection on training rows only; hold-out metrics remain separate.
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(X_train), min(100_000, len(X_train)), replace=False)
    ood = IsolationForest(
        n_estimators=150, contamination="auto", max_samples=min(2048, len(sample_idx)),
        n_jobs=max(1, min(4, os.cpu_count() or 1)), random_state=SEED,
    )
    ood.fit(X_train.iloc[sample_idx])
    print(classification_report(
        y_test, model.predict(X_test), labels=np.arange(len(classes)),
        target_names=classes, zero_division=0,
    ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "xgboost_model.joblib", compress=9)
    joblib.dump(ood, args.output_dir / "isolation_forest.joblib", compress=9)
    joblib.dump(encoder, args.output_dir / "label_encoder.joblib", compress=9)
    joblib.dump(imputer, args.output_dir / "imputer.joblib", compress=9)
    metadata = {
        "feature_columns": features,
        "label_column": label_col,
        "classes": classes,
        "n_classes": len(classes),
        "selected_feature_count": len(features),
        "dataset": "CICIDS2017",
        "random_state": SEED,
        "schema_policy": "exact_40_numeric_features_no_zero_fill",
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "feature_selection_fit": "training_partition_only",
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Artifacts saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
