# ThreatLens | Hybrid SOC Co-Pilot

ThreatLens is a local Streamlit proof-of-concept for explainable triage of network-flow CSVs. It combines XGBoost binary classification, SHAP feature attribution, a PennyLane four-qubit circuit for ambiguous model scores, and local MiniLM + FAISS retrieval over six curated MITRE ATT&CK entries.

## Architecture

```text
CSV upload → chunked normalization / feature derivation → XGBoost
                                         ├── SHAP top-three evidence
                                         ├── (0.40–0.65 score only) PennyLane default.qubit
                                         └── MiniLM embeddings → local FAISS ATT&CK retrieval
                                                ↓
                              deterministic Streamlit SOC triage cards + local CSV output
```

No uploaded flow data is sent to a cloud LLM or remote inference API. The Sentence Transformers model weights may be downloaded from Hugging Face on first launch; cache them before running with `HF_HUB_OFFLINE=1` for offline use.

## Run locally

Python 3.11 is recommended, especially for scientific and quantum-simulation wheels.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app provides **Load Synthetic Attack Benchmark** (500 reproducible rows) and **Preprocess & Analyze CSV**. CSV rows are read in configurable chunks and predictions stream to a local temporary CSV. The UI keeps a 500-row preview and up to 200 high-scoring candidate alerts in memory, then computes SHAP/retrieval details for the top 50 alerts. It executes the four-qubit simulation only for scores in the 0.40–0.65 interval, prioritizing borderline scores and limiting each upload to 1,000 simulations to protect throughput; diagnostics disclose eligible cases skipped after that cap. The sidebar upload limit is configured to 1 GiB in `.streamlit/config.toml`; increase it there for larger local deployments if sufficient disk/RAM are available.

## CSV preprocessing and schema compatibility

ThreatLens normalizes common CICIDS-style, NetFlow-style, and some Zeek-style aliases for these seven model features:

- `Destination_Port`
- `Flow_Duration`
- `Total_Fwd_Packets`
- `Total_Bwd_Packets`
- `Flow_Bytes_s`
- `Packet_Length_Mean`
- `SYN_Flag_Count`

It can derive mean packet length from directional means and approximate byte rate from directional byte counts plus flow duration. These derivations assume CICIDS-style flow duration measured in microseconds. Unavailable fields are filled with zero so files can be inspected, but the app displays a warning for each missing feature family and marks the row's preprocessing warning. **Arbitrary network CSVs are not semantically interchangeable**: predictions from a file with missing or differently defined fields may be unreliable. Review diagnostics and normalize source units before interpreting detections. Upload `Label` columns are retained as context only and never passed to the model.

Chunked parsing and inference reduce peak working memory for data rows and write predictions to disk rather than holding every prediction in a single dataframe. Streamlit still receives the uploaded file and a local temporary output file is used; the UI does not promise unlimited data sizes. Monitor local storage and available memory for very large uploads.

## Detection, explainability, and retrieval

- **XGBoost:** a binary model trained on an internally generated 500-row synthetic benchmark. It predicts benign versus malicious; it is not trained from upload labels.
- **SHAP:** ranks three features by absolute local SHAP value for a flagged flow. Values explain model-output attribution, not causation or attack probability.
- **PennyLane:** a four-qubit `default.qubit` circuit receives four normalized features only when the XGBoost score is between 0.40 and 0.65 inclusive. The mapped score is an exploratory simulator signal, not a calibrated risk probability and does not override the classifier.
- **MiniLM + FAISS:** encodes a six-entry curated ATT&CK knowledge base locally, retrieves a semantically close entry, and presents its tactics and three mitigation steps. A retrieval match is contextual guidance, not proof that the technique occurred.
- **Streamlit:** shows dataset metrics, colored preview rows, preprocessing warnings, top alert cards, and a downloadable full processed CSV.

## Evaluation limits

The 500-row benchmark demonstrates the workflow, not real-world accuracy. Validate on representative datasets such as CICIDS2017/UNSW-NB15, check class-wise precision/recall, calibrate probability thresholds, and audit false positives before operational use. The quantum circuit is a fixed-parameter heuristic demonstration, not evidence of quantum advantage or a trained quantum classifier. ThreatLens supports human investigation; it does not autonomously confirm incidents or block traffic.

## Tests

```bash
pytest -q tests/test_pipeline.py
```
