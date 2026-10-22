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

Use **Load Synthetic Network Logs** for the reproducible demo or upload a CSV. The app retains the selected dataset across Streamlit reruns. Uploads replace common CICIDS column spacing and punctuation with underscores, validate all required features, coerce invalid numeric values, and replace infinities or missing measurements with zero.

## Data schema

Required numeric columns: `Destination_Port`, `Flow_Duration`, `Total_Fwd_Packets`, `Total_Bwd_Packets`, `Flow_Bytes_s`, `Flow_Packets_s`, `Packet_Length_Mean`, and `SYN_Flag_Count`. `Label`, addresses, and IDs are never model inputs. Upload labels are retained as source context; predictions are generated independently. Missing or malformed numeric cells are converted to zero after coercion.

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
