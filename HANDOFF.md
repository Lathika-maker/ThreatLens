# ThreatLens handoff

## Current implementation

The application is a single-file Streamlit dashboard in `app.py`. Its model input is the fixed `FEATURES` list; `Label`, `Flow_ID`, and IPs remain context only. Class labels are `BENIGN`, `PortScan`, `DDoS`, `BruteForce`, and `PrivilegeAnomaly`.

The embedded `THREAT_INTEL` dictionary contains exactly T1046, T1110, T1071, T1499, T1190, and T1078. `CLASS_TO_TECHNIQUE` maps PortScan to T1046, BruteForce to T1110, DDoS to T1499, and PrivilegeAnomaly to T1078. The classifier mapping stays primary for those known classes; FAISS nearest ID and L2 distance are preserved as supporting retrieval output if they disagree.

Synthetic input is produced by `generate_flows(n_per_class=1500, seed=42)`: 2,500 BENIGN flows and 1,500 per attack class, with seeded cross-class feature noise and recomputed packet rates. `train_model()` uses a stratified 75/25 split, random state 42, and a 100-tree Random Forest with max depth 12. Its hold-out accuracy on this run was **93.6471%** (0.936471); this is synthetic-data pipeline validation, not real traffic performance.

## Functions in `app.py`

- `technique_text`, `_sample_range`, `generate_flows`
- `load_embedder`, `build_vector_index`, `_model_matrix`, `_baseline_matrix`, `train_model`
- `normalize_columns`, `prepare_uploaded_frame`, `load_uploaded_csv`
- `predict_flows`, `_transformed_value`, `explain_flow`
- `_behavior_phrase`, `build_query`, `retrieve_technique`
- `generate_report`, `triage_flow`, `reports_to_frame`
- `system_status`, `_style_flows`, `main`

The embedder uses `sentence-transformers/all-MiniLM-L6-v2`; cached files live under `.cache/huggingface`, which is ignored by Git. `HF_HUB_OFFLINE=1` works after the model has been cached. CPU thread pools are limited to keep local encoding stable on Apple Silicon. PyArrow is constrained below 20 to remain compatible with NumPy 1.x.

Sidebar health flags are computed by `system_status()` from the trained model, actual FAISS dimensions/count, and loaded CPU embedder. Triage time starts at `time.perf_counter()` and covers evidence extraction, retrieval, and report creation for the rendered (up to 50 highest-confidence) alerts. It shows `N/A` when no alerts were triaged.

## Verification recorded

- `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q`: **12 passed in 4.97 seconds**.
- Streamlit `AppTest` loaded the app, clicked **Load Synthetic Network Logs**, and rendered four metrics and 50 alert cards without exceptions.
- `HF_HUB_OFFLINE=1 .venv/bin/streamlit run app.py --server.headless true --server.port 8502`: server started; `/_stcore/health` returned `ok`; `/` returned HTTP 200; server was stopped.
- Real MiniLM embeddings produced a 6 × 384 FAISS index. Each knowledge-base text retrieved its own entry at distance 0. The PortScan, DDoS, and BruteForce examples returned T1046, T1499, and T1110.
- CSV ingestion tests confirm missing feature names are reported without raising and invalid numbers/infinities become zero.
- `.venv/` is ignored by Git.

## Known verification limits

No representative CICIDS2017 evaluation or live traffic capture was performed. A first-run model download from an empty cache was not part of the offline checks; the cached model was loaded and encoded successfully with `HF_HUB_OFFLINE=1`.
