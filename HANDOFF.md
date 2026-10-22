# ThreatLens handoff

## Current main-branch implementation

ThreatLens is now an artifact-driven, strict-schema SOC triage prototype for CICIDS2017-style network-flow data. It no longer trains the inference classifier on synthetic data inside the Streamlit app.

## Pipeline

1. `train_model.py` trains a multiclass XGBoost model and Isolation Forest locally, selects 40 numeric telemetry features, and writes model artifacts plus `model_metadata.json`.
2. `prepare_inference_csv.py` selects only those trained feature columns from a raw CICIDS-style CSV and explicitly drops rows with invalid selected feature values. It does not zero-fill or impute.
3. `app.py` rejects uploads unless they contain exactly the trained 40 columns, with no missing/extra/duplicate columns or invalid numeric values.
4. Isolation Forest runs first. Rows flagged OOD become `Unknown Novel Anomaly` and bypass XGBoost, SHAP, and ATT&CK retrieval.
5. In-distribution rows are classified by XGBoost. Per-alert SHAP evidence is shown with a clear non-causality disclaimer. The compact ATT&CK corpus supplies investigation context and suggested defensive validation, not proof of attribution.
6. The optional narrative model loads only when requested and checks available RAM first. Supabase, when configured, receives only a defined allowlist of incident metadata, never raw feature vectors or CSV rows.

## Local execution

```bash
python -m pip install -r requirements.txt
python train_model.py --data-dir ./CICIDS2017 --output-dir artifacts
python prepare_inference_csv.py --input ./CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv --output ./prepared_40_features.csv --max-rows 100000
streamlit run app.py
```

Replace the example dataset filename with a file that exists locally. Upload `prepared_40_features.csv` in the running app.

## Deployment prerequisites

- The trained `artifacts/` directory is intentionally git-ignored because model binaries and source datasets should not be committed accidentally. The directory must nevertheless be provisioned securely in the deployed runtime; the dashboard shows a clear error when artifacts are absent.
- Optional Supabase settings are `SUPABASE_URL` and `SUPABASE_KEY` in Streamlit secrets. Use least-privilege access.
- Do not upload private security telemetry or commit CICIDS2017 source files, secrets, or model binaries into a public repository.

## Validation boundaries

The Python syntax workflow compiles `app.py`, `train_model.py`, and `prepare_inference_csv.py`. The focused pytest suite covers exact schema acceptance/rejection, numeric validation, similarity-score bounds, SHAP output handling, and evidence formatting. These checks do not replace training on the actual dataset, live model inference, Streamlit memory testing, or validation on held-out operational data. Reported class confidence is not guaranteed to be calibrated. No automatic traffic blocking is performed.
