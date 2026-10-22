#!/usr/bin/env python3
"""Prepare a raw CICIDS-style CSV for strict ThreatLens inference.

This selects only the 40 trained numeric feature columns recorded in
artifacts/model_metadata.json. It never imputes, zero-fills, or alters values.
Rows containing invalid selected feature values are dropped with an explicit count.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalize(name: object) -> str:
    return re.sub(r"\s+", "_", str(name).replace("\ufeff", "").strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw CICIDS-style CSV")
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/model_metadata.json"))
    parser.add_argument("--output", type=Path, required=True, help="Output CSV with exactly the trained features")
    parser.add_argument("--max-rows", type=int, default=100_000, help="Maximum rows to read (1–100000)")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    if not args.metadata.is_file():
        parser.error(f"Model metadata not found: {args.metadata}; train the model first.")
    if not 1 <= args.max_rows <= 100_000:
        parser.error("--max-rows must be between 1 and 100000.")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    features = metadata.get("feature_columns")
    if not isinstance(features, list) or len(features) != 40:
        parser.error("Metadata must list exactly 40 feature_columns.")

    features = [normalize(feature) for feature in features]
    if len(set(features)) != len(features):
        parser.error("Metadata contains duplicate feature names after normalization.")
    required = set(features)

    try:
        # Select only model features while parsing the CSV to reduce peak memory.
        frame = pd.read_csv(
            args.input,
            usecols=lambda name: normalize(name) in required,
            nrows=args.max_rows,
            low_memory=False,
        )
    except Exception as exc:
        parser.error(f"Could not parse input CSV: {exc}")

    frame.columns = [normalize(name) for name in frame.columns]
    if frame.columns.duplicated().any():
        parser.error("Duplicate selected feature names after header normalization.")
    missing = [name for name in features if name not in frame.columns]
    if missing:
        parser.error("Input CSV lacks selected model features: " + ", ".join(missing))

    frame = frame[features].copy()
    for name in features:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    valid = frame.notna().all(axis=1)
    dropped = int((~valid).sum())
    frame = frame.loc[valid].astype(np.float32)
    if frame.empty:
        parser.error("No valid rows remain after rejecting missing/non-numeric/infinite values.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Prepared {len(frame):,} rows with exactly {len(features)} trained features.")
    print(f"Dropped {dropped:,} rows containing invalid selected feature values; no imputation was applied.")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
