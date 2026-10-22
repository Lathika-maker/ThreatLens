#!/usr/bin/env python3
"""Train ThreatLens' XGBoost, OOD detector, scaler, and rare-class Threat Vault.

Example:
  python train_model.py --data-dir ./CICIDS2017 --output-dir artifacts --trees 100

The output artifact names and the 40-feature schema are designed to match app.py.
Training data is read from Parquet files; generated and environment directories are
excluded from recursive discovery. Train locally with sufficient RAM.
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
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
N_FEATURES = 40
RARE_CLASS_THRESHOLD = 500
TARGET_NAMES = {"label", "class", "attack"}
NON_FEATURE_NAMES = {
    "flow_id", "src_ip", "source_ip", "source_address", "id_orig_h",
    "dst_ip", "dest_ip", "destination_ip", "destination_address", "id_resp_h",
    "timestamp", "time", "date",
}
EXCLUDED_DIRS = {
    ".venv", "venv", "node_modules", ".git", "__pycache__", ".pytest_cache",
    "site-packages", "artifacts",
}


def normalize_column(value: object) -> str:
    value = str(value).replace("\ufeff", "").strip()
    return re.sub(r"\s+", "_", value)


def canonicalize_label(value: object) -> str:
    """Map CICIDS spellings to stable names expected by app.py / MITRE_MAP."""
    raw = str(value).replace("\ufffd", "-").replace("–", "-").replace("—", "-").strip()
    normalized = re.sub(r"\s+", " ", raw).casefold()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    aliases = {
        "benign": "BENIGN",
        "benign traffic": "BENIGN",
        "web attack sql injection": "Web Attack – SQL Injection",
        "web attack sql-injection": "Web Attack – SQL Injection",
        "web attack sql injection ": "Web Attack – SQL Injection",
        "web attack xss": "Web Attack – XSS",
        "web attack brute force": "Web Attack – Brute Force",
        "web attack brute-force": "Web Attack – Brute Force",
        "heartbleed": "Heartbleed",
        "infiltration": "Infiltration",
    }
    return aliases.get(normalized, raw)


def find_parquet_files(data_dir: Path) -> list[Path]:
    results = []
    for path in data_dir.rglob("*.parquet"):
        try:
            parts = path.relative_to(data_dir).parts[:-1]
        except ValueError:
            continue
        if any(part in EXCLUDED_DIRS or part.startswith(".") for part in parts):
            continue
        results.append(path)
    return sorted(results)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--trees", type=int, default=350)
    parser.add_argument("--rare-threshold", type=int, default=RARE_CLASS_THRESHOLD)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.data_dir}")
    if args.trees < 1:
        parser.error("--trees must be >= 1")
    if args.rare_threshold < 2:
        parser.error("--rare-threshold must be >= 2")

    # 1. Create output directory before training or artifact export.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = find_parquet_files(args.data_dir)
    if not files:
        parser.error(f"No Parquet files found under {args.data_dir}")
    print(f"Found {len(files)} Parquet file(s).", flush=True)

    frames: list[pd.DataFrame] = []
    for file in files:
        print(f"Reading {file}", flush=True)
        try:
            frame = pd.read_parquet(file)
        except Exception as exc:
            parser.error(f"Could not read {file}: {type(exc).__name__}: {exc}")
        frame.columns = [normalize_column(c) for c in frame.columns]
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True, sort=False)
    label_col = next((c for c in data.columns if c.strip().casefold() in TARGET_NAMES), None)
    if label_col is None:
        parser.error("Target column not found; expected Label, Class, or Attack.")

    y = data[label_col].astype("string").str.strip()
    X = data.drop(columns=[label_col]).copy()
    X.columns = [normalize_column(c) for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    drop_names = [c for c in X.columns if c.strip().casefold() in NON_FEATURE_NAMES]
    if drop_names:
        print("Excluding identifier/time columns: " + ", ".join(drop_names), flush=True)
        X = X.drop(columns=drop_names)

    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    valid = y.notna() & y.ne("")
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True).map(canonicalize_label)

    if len(X) < 10:
        parser.error("Too few valid labeled rows.")
    if X.shape[1] < N_FEATURES:
        parser.error(f"Only {X.shape[1]} numeric features found; app.py requires {N_FEATURES}.")

    counts_by_label = y.value_counts()
    too_small = counts_by_label[counts_by_label < 2]
    if not too_small.empty:
        parser.error("Every label needs at least two rows for a stratified split: " +
                     ", ".join(f"{label}={count}" for label, count in too_small.items()))

    # Keep label space stable. Rare classes are handled by the Threat Vault, not XGBoost.
    # Force XSS through the Threat Vault path even if its row count exceeds the
    # general rarity threshold. It must not be trained as an XGBoost class.
    forced_vault_labels = {"Web Attack – XSS"}
    present_forced_labels = forced_vault_labels.intersection(set(counts_by_label.index.astype(str)))
    common_labels = sorted(
        label for label, count in counts_by_label.items()
        if count >= args.rare_threshold and label not in forced_vault_labels
    )
    rare_labels = sorted(
        set(counts_by_label[counts_by_label < args.rare_threshold].index.astype(str))
        | present_forced_labels
    )
    missing_forced = sorted(forced_vault_labels - set(counts_by_label.index.astype(str)))
    if missing_forced:
        print("WARNING: forced Threat Vault label(s) not present in dataset: " +
              ", ".join(missing_forced), flush=True)
    if len(common_labels) < 2:
        parser.error(
            f"Need at least two classes with >= {args.rare_threshold} rows for XGBoost; "
            f"found {len(common_labels)}. Adjust --rare-threshold only if appropriate."
        )
    print(f"Rows: {len(X):,}; classes: {len(counts_by_label)}; "
          f"XGBoost classes: {len(common_labels)}; Threat Vault classes: {len(rare_labels)}",
          flush=True)
    print("Canonical labels: " + ", ".join(sorted(counts_by_label.index.astype(str))), flush=True)

    # Split all rows first, ensuring prototype construction uses training rows only.
    all_indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train_raw, X_test_raw = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train_all, y_test_all = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw = X_test_raw.reset_index(drop=True)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_all = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X.columns)
    X_test_all = pd.DataFrame(imputer.transform(X_test_raw), columns=X.columns)
    X_train_all = X_train_all.astype(np.float32)
    X_test_all = X_test_all.astype(np.float32)

    # Feature selection is fitted only on common-label training rows.
    common_mask = y_train_all.isin(common_labels).to_numpy()
    X_common_unselected = X_train_all.loc[common_mask].reset_index(drop=True)
    y_common_train = y_train_all.loc[common_mask].reset_index(drop=True)
    selector = ExtraTreesClassifier(
        n_estimators=100, class_weight="balanced", max_features="sqrt",
        n_jobs=max(1, min(4, os.cpu_count() or 1)), random_state=SEED,
    )
    selector.fit(X_common_unselected, y_common_train)
    ranked = np.argsort(selector.feature_importances_)[::-1]
    feature_columns = [str(X.columns[i]) for i in ranked[:N_FEATURES]]

    X_train_features = X_train_all[feature_columns]
    X_test_features = X_test_all[feature_columns]

    # 3. Fit StandardScaler and export immediately, before model training.
    scaler = StandardScaler()
    common_train_scaled = scaler.fit_transform(X_train_features.loc[common_mask])
    joblib.dump(scaler, args.output_dir / "scaler.joblib", compress=3)
    print(f"Saved scaler.joblib with {len(feature_columns)} features.", flush=True)

    # 4. Threat Vault = per-rare-class centroid computed only from scaled training rows.
    threat_vault: dict[str, list[float]] = {}
    rare_mask = y_train_all.isin(rare_labels).to_numpy()
    if rare_mask.any():
        rare_scaled = scaler.transform(X_train_features.loc[rare_mask])
        rare_y = y_train_all.loc[rare_mask].reset_index(drop=True)
        for label in rare_labels:
            rows = rare_scaled[rare_y.to_numpy() == label]
            if rows.size == 0:
                print(f"WARNING: rare class {label!r} has no training rows; skipped.", flush=True)
                continue
            prototype = np.mean(rows, axis=0)
            if not np.isfinite(prototype).all():
                print(f"WARNING: non-finite prototype for {label!r}; skipped.", flush=True)
                continue
            # Store the vector even if its norm is small; app.py safely skips zero-norm vectors.
            threat_vault[label] = [float(v) for v in prototype]
    (args.output_dir / "threat_vault.json").write_text(
        json.dumps(threat_vault, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Saved threat_vault.json with {len(threat_vault)} rare-class prototype(s).", flush=True)

    # 5. Train XGBoost and Isolation Forest on the same scaled feature space.
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_common_train.astype(str))
    test_common_mask = y_test_all.isin(common_labels).to_numpy()
    X_test_common_scaled = scaler.transform(X_test_features.loc[test_common_mask])
    y_test_common = y_test_all.loc[test_common_mask].astype(str)
    test_encoded = encoder.transform(y_test_common) if len(y_test_common) else np.array([], dtype=int)

    model = XGBClassifier(
        objective="multi:softprob", num_class=len(encoder.classes_), n_estimators=args.trees,
        max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=2, min_child_weight=2, tree_method="hist", eval_metric="mlogloss",
        n_jobs=max(1, min(4, os.cpu_count() or 1)), random_state=SEED,
    )
    model.fit(common_train_scaled, y_encoded, verbose=False)

    # Fit OOD detector only on the common-class, scaled training distribution.
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(
        len(common_train_scaled), min(100_000, len(common_train_scaled)), replace=False
    )
    ood = IsolationForest(
        n_estimators=150, contamination="auto",
        max_samples=min(2048, len(sample_idx)),
        n_jobs=max(1, min(4, os.cpu_count() or 1)), random_state=SEED,
    )
    ood.fit(common_train_scaled[sample_idx])

    joblib.dump(model, args.output_dir / "xgboost_model.joblib", compress=3)
    joblib.dump(ood, args.output_dir / "isolation_forest.joblib", compress=3)
    joblib.dump(encoder, args.output_dir / "label_encoder.joblib", compress=3)
    joblib.dump(imputer, args.output_dir / "imputer.joblib", compress=3)

    if len(test_encoded):
        predictions = model.predict(X_test_common_scaled).astype(int)
        print("\nXGBoost evaluation on held-out common classes:")
        print(classification_report(
            test_encoded, predictions, labels=np.arange(len(encoder.classes_)),
            target_names=[str(c) for c in encoder.classes_], zero_division=0,
        ))
    else:
        print("WARNING: held-out split has no common-class rows; evaluation skipped.")

    metadata = {
        "feature_columns": feature_columns,
        "label_column": label_col,
        "classes": [str(c) for c in encoder.classes_],
        "n_classes": int(len(encoder.classes_)),
        "selected_feature_count": len(feature_columns),
        "dataset": "CICIDS2017",
        "random_state": SEED,
        "schema_policy": "exact_40_numeric_features_no_zero_fill",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "feature_selection_fit": "common-class training partition only",
        "scaler_fit": "common-class training partition only",
        "rare_class_threshold": int(args.rare_threshold),
        "rare_classes": rare_labels,
        "threat_vault_classes": sorted(threat_vault.keys()),
        "xgboost_classes": [str(c) for c in encoder.classes_],
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    expected = {
        "xgboost_model.joblib", "isolation_forest.joblib", "label_encoder.joblib",
        "imputer.joblib", "scaler.joblib", "threat_vault.json", "model_metadata.json",
    }
    missing = sorted(name for name in expected if not (args.output_dir / name).is_file())
    if missing:
        fail("Artifact export incomplete: " + ", ".join(missing))
    print("\nAll seven app.py artifacts exported to:", args.output_dir.resolve(), flush=True)
    for name in sorted(expected):
        path = args.output_dir / name
        print(f"  {name}: {path.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
