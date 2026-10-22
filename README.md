# ThreatLens | Evidence-First SOC Co-Pilot

ThreatLens is a Streamlit prototype for explainable triage of CICIDS2017 network-flow records. Its inference pipeline uses locally trained multiclass XGBoost and Isolation Forest artifacts, per-alert SHAP feature attribution, a compact retrieval corpus for ATT&CK-informed investigation guidance, and optional evidence-grounded narrative generation.

## Current inference contract

- **Exactly 40 numeric input features.** The feature list comes from `artifacts/model_metadata.json`. Uploaded files containing missing, extra, duplicate, non-numeric, NaN, or infinite feature values are rejected before inference.
- **OOD gate first.** Isolation Forest is applied to each row first. Rows flagged out-of-distribution are labeled `Unknown Novel Anomaly` and bypass XGBoost, SHAP, and ATT&CK retrieval.
- **Explainability is evidence, not causation.** SHAP values describe feature contributions to a model output; they do not prove an attack or explain its real-world cause.
- **ATT&CK mapping is contextual guidance.** Semantic similarity to the small curated reference corpus is not an attack probability or proof of attribution.
- **Optional AI narrative.** The narrative model is loaded only on request and a RAM guard provides deterministic SHAP evidence as fallback.
- **Privacy-aware incident ledger.** Supabase receives only allowlisted incident metadata. Raw CSV rows and full feature vectors are not sent to Supabase.

## 1. Install

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Train locally

Download CICIDS2017 from its official/public dataset source, extract its CSV files into a directory, and run:

```bash
python train_model.py --data-dir ./CICIDS2017 --output-dir artifacts
```

Training prints a classification report and writes these files:

- `artifacts/xgboost_model.joblib`
- `artifacts/isolation_forest.joblib`
- `artifacts/label_encoder.joblib`
- `artifacts/imputer.joblib`
- `artifacts/model_metadata.json`

The trainer excludes recognizable identifier/time columns, drops fields that contain no numeric measurements, selects the 40 most informative numeric features, and stores their exact names in metadata. Training is intended to happen locally, not while Streamlit is starting up.

## 3. Prepare an inference CSV

The app intentionally rejects the original unfiltered CICIDS2017 table if it contains label, IP, timestamp, or other columns beyond the 40 selected features. Create an inference CSV with **only** the 40 columns named in `artifacts/model_metadata.json`; preserve those names and numeric values exactly. Do not include the target label. The app does not fill absent fields with zeros.

Prepare a compatible file from a raw CICIDS CSV using the supplied helper. It selects only the trained feature columns and explicitly drops rows with invalid selected values; it never imputes or zero-fills them:

```bash
python prepare_inference_csv.py --input ./CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv --output ./prepared_40_features.csv --max-rows 100000
```

Replace the example filename with a CSV that exists in your downloaded dataset. To inspect the expected schema:

```bash
python -c 'import json; print("\\n".join(json.load(open("artifacts/model_metadata.json"))["feature_columns"]))'
```

## 4. Run the dashboard

```bash
streamlit run app.py
```

The app requires the trained `artifacts/` directory to be available in the runtime. For Streamlit Community Cloud, provide these artifacts securely as part of the deployment process; the app does not train models or fabricate model files when artifacts are missing. Do not commit the original CICIDS2017 dataset or private security telemetry to a public repository.

## 5. Optional Supabase incident ledger

Add the following values to Streamlit secrets (locally, `.streamlit/secrets.toml`; on Community Cloud, the app's Secrets settings):

```toml
SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"
SUPABASE_KEY = "YOUR-SUPABASE-KEY"
```

Create an `incident_ledger` table with columns matching `timestamp`, `alert_id`, `threat_class`, `top_shap_features` (JSON/JSONB), and `mitre_technique`. Use least-privilege credentials and avoid a privileged service-role key in client-facing configurations.

## Limits and evaluation notes

This is a prototype, not a production-certified SOC product. Model performance depends on the dataset, label quality, train/test split, and class distribution. Reported class confidence is the model's maximum class score, not necessarily a calibrated probability. OOD decisions depend on the fitted Isolation Forest. Validate precision, recall, false-positive rate, inference latency, and analyst effort on held-out and operationally representative data before using it for response decisions.
