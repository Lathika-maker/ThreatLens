# ThreatLens | SOC Co-Pilot

A local Streamlit dashboard for network-flow triage. A Random Forest classifies flows, a deterministic evidence score explains flagged records, and local MiniLM embeddings with FAISS retrieve a MITRE ATT&CK technique. Report text and response steps come from fixed templates and the embedded knowledge base; no flow data is sent to a cloud model.

## Architecture

```text
CSV / seeded demo → Pandas ingestion → Random Forest → per-flow evidence → MiniLM + FAISS → Streamlit triage report
```

The five layers are ingestion, detection, evidence extraction, retrieval and mapping, and dashboard. MiniLM downloads once into `.cache/huggingface`; `HF_HUB_OFFLINE=1` uses the cached copy without contacting Hugging Face.

## Run locally

Python 3.11 is recommended, especially for FAISS and PyTorch wheels.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Use **Load Synthetic Network Logs** for the reproducible demo or upload a CSV. CSV parsing is configurable in chunks (10k–250k rows) and normalizes common CICIDS/NetFlow/Zeek-style field aliases. Where source fields exist, preprocessing can derive packet rate from packet counts and flow duration, or approximate byte rate from packet counts, mean packet size, and duration. Invalid, missing, and infinite required numeric cells are converted to zero and reported in the preprocessing summary. The cleaned data is retained in memory for the current dashboard, so chunked parsing reduces parser memory pressure but does not make the entire app out-of-core for arbitrarily large datasets.

## Data schema

Required model features after safe alias mapping/derivation: `Destination_Port`, `Flow_Duration`, `Total_Fwd_Packets`, `Total_Bwd_Packets`, `Flow_Bytes_s`, `Flow_Packets_s`, `Packet_Length_Mean`, and `SYN_Flag_Count`. Arbitrary CSVs are not automatically compatible: if these values cannot be mapped or derived, the app reports the missing fields rather than inventing values. The duration-based rate derivation assumes source flow duration is in microseconds (as in CICIDS-style flows). `Label`, addresses, and IDs are never model inputs. Upload labels are retained as source context; predictions are generated independently. Missing or malformed numeric cells are converted to zero after coercion.

## Model and triage

The seeded synthetic five-class dataset is split with stratification and `random_state=42`. The displayed accuracy and class report come from its hold-out split. Top-three evidence combines Random Forest feature importance with each flow's deviation from the benign training baseline in log space; displayed shares sum to 100%. Confidence is the model's ensemble vote share, not the probability that a flow is malicious. High priority means confidence of at least 90% for DDoS, BruteForce, or PrivilegeAnomaly; all other alerts are Medium.

For known attack classes, the classifier's class-to-technique mapping is the primary report technique, while the FAISS nearest technique and distance remain available for comparison. This keeps retrieval grounded without allowing a semantic mismatch to overrule the classifier.

## Evaluation honesty and limits

Synthetic data checks that the full workflow operates; its scores do not estimate real-world detection performance. Real CICIDS2017 evaluation is a separate, optional research step. The detector knows only its trained classes and does not capture packets, identify unknown attack families, or replace analyst validation. MITRE retrieval is based on six curated entries, not the full ATT&CK matrix.

## Future work

Evaluate on a representative CICIDS2017 sample, expand the technique knowledge base, add novelty detection, and support Zeek or NetFlow ingestion. Live packet capture, cloud LLM summaries, authentication, and a persistent database are outside this MVP.

## Verification

```bash
pytest -q tests/test_pipeline.py
HF_HUB_OFFLINE=1 streamlit run app.py --server.headless true
```

The synthetic data and model tests run without a model download. The real embedding model is downloaded on the first app run unless `HF_HUB_OFFLINE=1` is set; offline mode requires that cached copy to exist.

See [HANDOFF.md](HANDOFF.md) for the verified function list, measured synthetic hold-out score, test output, and remaining evaluation limits.
