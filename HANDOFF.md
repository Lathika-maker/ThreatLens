# ThreatLens handoff

## Current branch change

Branch `feature/chunked-csv-preprocessing` refactors the app to the requested **ThreatLens: Hybrid Quantum-Classical AI Co-Pilot for Explainable Threat Triage** demo. The existing `main` branch was preserved for review. The branch adds an XGBoost binary classifier, SHAP explanations, a PennyLane `default.qubit` edge-case score, local MiniLM + FAISS retrieval, and chunked network CSV processing.

## Pipeline and constraints

- `FEATURES`: `Destination_Port`, `Flow_Duration`, `Total_Fwd_Packets`, `Total_Bwd_Packets`, `Flow_Bytes_s`, `Packet_Length_Mean`, `SYN_Flag_Count`.
- `generate_synthetic_data()` builds a repeatable 500-row binary labeled benchmark; `train_model()` trains XGBoost on this synthetic data only. Uploaded `Label` values are never passed into model inference.
- `preprocess_network_chunk()` normalizes common CICIDS/NetFlow/Zeek aliases, derives byte-rate/mean-packet-length measurements only when appropriate source columns exist, and reports missing features or invalid cells.
- `analyze_csv_in_chunks()` streams parsing and inference in batches and writes complete output predictions to a temporary local CSV. The UI retains a 500-row preview and up to 200 candidate alerts, then generates detailed cards for the top 50. Streamlit upload ceiling is set to 1 GiB by `.streamlit/config.toml`.
- `explain_flow()` ranks the top three absolute SHAP attributions.
- The four-qubit PennyLane circuit runs only when the XGBoost score is in [0.40, 0.65]. Its fixed-parameter output is an exploratory score, not a calibrated probability or demonstrated quantum advantage.
- The local FAISS index contains six curated ATT&CK entries and uses MiniLM 384-dimensional embeddings. Retrieval is supporting context only, not proof of a technique.
- No cloud LLM/inference API is used for uploaded data. Hugging Face weights can be fetched once on first launch; after caching, `HF_HUB_OFFLINE=1` can force offline loading.

## Important interpretation limits

The model schema is fixed. Alias normalization helps with common exports, but arbitrary CSV columns do not become semantically compatible automatically. Missing numeric model features are zero-filled so preprocessing can continue, with warnings shown to users; predictions with missing or differently defined features may be unreliable. Flow-duration-based derivations assume microseconds as in CICIDS-style datasets.

The synthetic benchmark is for pipeline demonstration, not real-world performance. Validate with representative labeled network data before making detection claims. The quantum circuit is a fixed-weight simulator demo, and not evidence of a practical quantum advantage. Triage cards assist a human analyst; ThreatLens does not auto-block traffic.

## Verification status

The tests in `tests/test_pipeline.py` have been updated to cover schema normalization, preprocessing diagnostics and derivations, synthetic binary data, XGBoost predictions, SHAP feature attribution, quantum score range, and chunked output. **These updated tests still need to be run in the project environment after installing the updated `requirements.txt`; do not represent them as passing until that command succeeds.**

Run locally:

```bash
python -m pip install -r requirements.txt
pytest -q tests/test_pipeline.py
streamlit run app.py
```
