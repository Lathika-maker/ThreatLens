"""ThreatLens: evidence-first SOC triage for CICIDS2017 network flows.

Train model artifacts locally with train_model.py before deploying this app.
Uploaded CSVs must contain exactly the 40 numeric feature columns in
artifacts/model_metadata.json. No missing-feature filling or extra-column guessing.
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
import psutil
import streamlit as st

st.set_page_config(page_title="ThreatLens | SOC Co-Pilot", page_icon="🛡️", layout="wide")

ARTIFACT_DIR = Path(os.getenv("THREATLENS_ARTIFACT_DIR", "artifacts"))
MAX_FEATURES = 40
MAX_UPLOAD_MB = 80
MAX_ROWS = 100_000
MAX_ALERTS = 100
MIN_RAM_FOR_SLM = 600 * 1024 * 1024
SLM_MODEL_ID = "MBZUAI/LaMini-Flan-T5-77M"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

MITRE_MAP: dict[str, dict[str, str]] = {
    "BENIGN": {"technique": "No technique assigned", "description": "Classified as benign by the model; this is not proof of safety.", "mitigation": "Compare with local baselines and surrounding telemetry."},
    "Bot": {"technique": "T1071 — Application Layer Protocol (context-dependent)", "description": "The dataset class is Bot; flow features alone cannot prove botnet attribution.", "mitigation": "Review DNS, destination reputation, endpoint telemetry, and connection periodicity."},
    "DDoS": {"technique": "T1498 — Network Denial of Service", "description": "Traffic may be consistent with network denial-of-service behavior.", "mitigation": "Correlate traffic volume, source distribution, service health, and upstream telemetry."},
    "DoS Hulk": {"technique": "T1498 — Network Denial of Service", "description": "Potential service exhaustion behavior; confirm impact in service logs.", "mitigation": "Validate request rates and service impact before applying rate limits."},
    "DoS GoldenEye": {"technique": "T1498 — Network Denial of Service", "description": "Potential denial-of-service behavior.", "mitigation": "Review service health, traffic baselines, and source distribution."},
    "DoS slowloris": {"technique": "T1499 — Endpoint Denial of Service", "description": "Potential connection exhaustion against an endpoint.", "mitigation": "Review concurrent connections, request completion times, and web-server limits."},
    "DoS Slowhttptest": {"technique": "T1499 — Endpoint Denial of Service", "description": "Potential endpoint resource exhaustion.", "mitigation": "Inspect HTTP sessions and server resource metrics."},
    "FTP-Patator": {"technique": "T1110 — Brute Force", "description": "Repeated FTP authentication attempts may indicate credential guessing.", "mitigation": "Review authentication logs, account lockouts, source reputation, and MFA."},
    "SSH-Patator": {"technique": "T1110 — Brute Force", "description": "Repeated SSH attempts may indicate password guessing.", "mitigation": "Review SSH logs, restrict access, and prefer key-based authentication."},
    "PortScan": {"technique": "T1046 — Network Service Discovery", "description": "Connection patterns may indicate probing for exposed services.", "mitigation": "Review source scope, destination ports, asset ownership, and firewall telemetry."},
    "Infiltration": {"technique": "T1041 — Exfiltration Over C2 Channel (context-dependent)", "description": "The dataset label suggests suspicious activity but does not prove exfiltration.", "mitigation": "Review endpoint/proxy logs and unusual egress; preserve evidence."},
    "Web Attack – Brute Force": {"technique": "T1110 — Brute Force", "description": "Potential repeated login attempts against a web application.", "mitigation": "Correlate web and identity logs and check whether accounts were accessed."},
    "Web Attack – XSS": {"technique": "Technique requires validation", "description": "XSS is a vulnerability class; this label alone does not establish an ATT&CK technique.", "mitigation": "Inspect application requests and input handling in an authorized environment."},
    "Web Attack – Sql Injection": {"technique": "T1190 — Exploit Public-Facing Application", "description": "Potential injection attempts against an exposed application.", "mitigation": "Review application/WAF logs, validate input handling, and patch vulnerable components."},
    "Heartbleed": {"technique": "T1190 — Exploit Public-Facing Application (context-dependent)", "description": "The dataset class suggests Heartbleed-like traffic; confirm vulnerable software and exploit evidence.", "mitigation": "Patch affected TLS implementations and rotate secrets if exposure is confirmed."},
}
DEFAULT_ATTACK = {"technique": "Technique unassigned", "description": "No direct mapping is configured for this class. A model label is not proof of attribution.", "mitigation": "Validate with independent endpoint, identity, network, and application telemetry."}

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background:#111c2c;border:1px solid #24364d;padding:14px 16px;border-radius:12px;}
[data-testid="stMetricLabel"] {color:#9bb4ca;}
[data-testid="stMetricValue"] {color:#47f5c5;}
</style>
""", unsafe_allow_html=True)


def normalize_column(value: object) -> str:
    value = str(value).replace("\ufeff", "").strip()
    return re.sub(r"\s+", "_", value)


def safe_text(value: Any, limit: int = 160) -> str:
    return "".join(ch for ch in str(value) if ch.isprintable())[:limit]


def validate_input_frame(raw: pd.DataFrame, features: list[str], max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Return a validated float32 matrix or raise ValueError without imputing data."""
    frame = raw.copy()
    frame.columns = [normalize_column(c) for c in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError("STRICT SCHEMA REJECTION: duplicate normalized columns: " + ", ".join(duplicates))
    missing = [name for name in features if name not in frame.columns]
    extra = [name for name in frame.columns if name not in features]
    if missing or extra or len(frame.columns) != len(features):
        detail = []
        if missing:
            detail.append("missing required features: " + ", ".join(missing))
        if extra:
            detail.append("unexpected columns: " + ", ".join(extra))
        raise ValueError("STRICT SCHEMA REJECTION — no inference performed. " + " | ".join(detail))
    if frame.empty or len(frame) > max_rows:
        raise ValueError(f"Upload rejected: expected 1–{max_rows:,} data rows; got {len(frame):,}.")
    matrix = frame[features].copy()
    for name in features:
        matrix[name] = pd.to_numeric(matrix[name], errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    invalid = matrix.isna().sum()
    invalid = invalid[invalid > 0]
    if not invalid.empty:
        detail = ", ".join(f"{name}={int(count)}" for name, count in invalid.items())
        raise ValueError("STRICT DATA REJECTION — non-numeric, missing, or infinite values: " + detail)
    return matrix.astype(np.float32)


# Replace each placeholder with the share ID from a Google Drive file link.
# The Drive files must be accessible to the deployment service account/public-link policy.
ARTIFACT_DRIVE_IDS = {
    "artifacts/xgboost_model.joblib": "REPLACE_WITH_GDRIVE_ID",
    "artifacts/isolation_forest.joblib": "REPLACE_WITH_GDRIVE_ID",
    "artifacts/label_encoder.joblib": "REPLACE_WITH_GDRIVE_ID",
    "artifacts/imputer.joblib": "REPLACE_WITH_GDRIVE_ID",
    "artifacts/model_metadata.json": "REPLACE_WITH_GDRIVE_ID",
}
ARTIFACT_FILENAMES = tuple(Path(name).name for name in ARTIFACT_DRIVE_IDS)


def ensure_artifacts_present() -> None:
    """Download missing artifacts from Drive, or fail with an actionable error."""
    missing = [
        (relative_name, ARTIFACT_DIR / Path(relative_name).name, drive_id)
        for relative_name, drive_id in ARTIFACT_DRIVE_IDS.items()
        if not (ARTIFACT_DIR / Path(relative_name).name).is_file()
    ]
    if not missing:
        return

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with st.spinner("Downloading heavy enterprise model artifacts (>100MB) from secure cloud storage..."):
        invalid_ids = [
            name for name, _path, drive_id in missing
            if not drive_id or drive_id.startswith("REPLACE_") or drive_id.strip() in {"", "YOUR_FILE_ID"}
        ]
        if invalid_ids:
            raise RuntimeError(
                "Google Drive IDs are still placeholders for: " + ", ".join(invalid_ids)
            )
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "gdown is not installed. Install dependencies from requirements.txt and redeploy."
            ) from exc

        failed = []
        for relative_name, destination, drive_id in missing:
            try:
                result = gdown.download(
                    id=drive_id,
                    output=str(destination),
                    quiet=False,
                    fuzzy=True,
                )
                if not result or not destination.is_file() or destination.stat().st_size == 0:
                    failed.append(relative_name)
            except Exception:
                failed.append(relative_name)
        if failed:
            for name in failed:
                (ARTIFACT_DIR / Path(name).name).unlink(missing_ok=True)
            raise RuntimeError("Google Drive download failed or returned empty files: " + ", ".join(failed))


@st.cache_resource(max_entries=1, show_spinner=False)
def load_artifacts() -> dict[str, Any]:
    ensure_artifacts_present()
    paths = {
        "model": ARTIFACT_DIR / "xgboost_model.joblib",
        "ood": ARTIFACT_DIR / "isolation_forest.joblib",
        "encoder": ARTIFACT_DIR / "label_encoder.joblib",
        "imputer": ARTIFACT_DIR / "imputer.joblib",
        "metadata": ARTIFACT_DIR / "model_metadata.json",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing trained artifacts after cloud retrieval: " + ", ".join(missing))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    features = metadata.get("feature_columns")
    if not isinstance(features, list) or len(features) != MAX_FEATURES:
        raise ValueError(f"model_metadata.json must define exactly {MAX_FEATURES} feature_columns.")
    features = [normalize_column(x) for x in features]
    if len(set(features)) != len(features):
        raise ValueError("Feature metadata contains duplicate names after normalization.")
    model, ood = joblib.load(paths["model"]), joblib.load(paths["ood"])
    encoder, imputer = joblib.load(paths["encoder"]), joblib.load(paths["imputer"])
    if hasattr(model, "n_features_in_") and int(model.n_features_in_) != len(features):
        raise ValueError("XGBoost feature count differs from model metadata.")
    if hasattr(ood, "n_features_in_") and int(ood.n_features_in_) != len(features):
        raise ValueError("Isolation Forest feature count differs from model metadata.")
    return {"model": model, "ood": ood, "encoder": encoder, "imputer": imputer,
            "metadata": metadata, "features": features}


@st.cache_resource(max_entries=1, show_spinner=False)
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_ID, device="cpu")


@st.cache_resource(max_entries=1, show_spinner=False)
def load_rag_resources():
    import faiss
    records = []
    for attack, details in MITRE_MAP.items():
        if attack == "BENIGN":
            continue
        records.append({"attack": attack, **details,
            "text": f"{attack}. {details['description']} Mitigation: {details['mitigation']}"})
    vectors = load_embedder().encode([r["text"] for r in records],
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, records


def contextual_match_confidence(squared_distance: float) -> float:
    # Unit-normalized embeddings have squared L2 distance in [0, 4].
    d = max(0.0, min(4.0, float(squared_distance)))
    return 100.0 * (1.0 - d / 4.0)


def retrieve_mitre(attack: str) -> tuple[dict[str, str], float]:
    index, records = load_rag_resources()
    query = f"{attack} network flow attack behavior investigation and defensive mitigation"
    vector = load_embedder().encode([query], convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    distances, indices = index.search(vector, 1)
    del vector
    pos = int(indices[0, 0])
    if pos < 0 or pos >= len(records):
        raise RuntimeError("Semantic retrieval returned no valid record.")
    return records[pos], contextual_match_confidence(float(distances[0, 0]))


def shap_values_for_class(explainer, row: pd.DataFrame, class_index: int) -> np.ndarray:
    values = explainer.shap_values(row)
    if isinstance(values, list):
        return np.asarray(values[min(class_index, len(values) - 1)]).reshape(row.shape[0], -1)[0]
    values = np.asarray(values)
    if values.ndim == 3:
        # SHAP layouts encountered: samples x features x classes.
        return values[0, :, min(class_index, values.shape[2] - 1)]
    if values.ndim == 2:
        return values[0]
    return values.reshape(-1)


def make_evidence(row: pd.Series, impacts: np.ndarray, features: list[str]):
    impacts = np.asarray(impacts).reshape(-1)
    order = np.argsort(np.abs(impacts))[::-1][:min(5, len(impacts), len(features))]
    items = [{"feature": features[int(i)], "value": float(row.iloc[int(i)]),
              "impact": float(impacts[int(i)])} for i in order]
    if not items:
        return "SHAP evidence is unavailable; no feature-impact claim is made.", []
    parts = []
    for item in items:
        direction = "increased" if item["impact"] > 0 else "decreased" if item["impact"] < 0 else "had near-zero"
        parts.append(f"{item['feature']}={item['value']:.5g} {direction} the model score for this class (SHAP {item['impact']:+.5g})")
    return "Top model contributions: " + "; ".join(parts) + ". SHAP describes model influence, not causation.", items


@st.cache_resource(max_entries=1, show_spinner=False)
def load_narrative_pipeline():
    from transformers import pipeline
    return pipeline("text2text-generation", model=SLM_MODEL_ID, device=-1)


def generate_narrative(attack: str, evidence: str) -> str:
    if psutil.virtual_memory().available < MIN_RAM_FOR_SLM:
        st.warning("Available RAM is below the safety threshold; using deterministic SHAP explanation.")
        return evidence
    try:
        gc.collect()
        generator = load_narrative_pipeline()
        prompt = ("Write a short SOC analyst note using ONLY the evidence supplied. "
            "Do not invent causes or facts. State uncertainty and one defensive validation step. "
            f"Predicted class: {attack}. Evidence: {evidence}")
        answer = generator(prompt, max_new_tokens=80, do_sample=False, truncation=True)
        return str(answer[0].get("generated_text", "")).strip() or evidence
    except Exception as exc:
        st.warning(f"AI narrative unavailable ({type(exc).__name__}); using deterministic evidence.")
        return evidence
    finally:
        gc.collect()


def get_supabase_client():
    try:
        from supabase import create_client
        url, key = st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]
        return create_client(url, key) if url and key else None
    except Exception:
        return None


def main() -> None:
    st.title("🛡️ ThreatLens")
    st.caption("Dynamic Explainable Threat Triage · Evidence-first SOC co-pilot")

    try:
        bundle = load_artifacts()
        artifact_error = None
    except Exception as exc:
        bundle, artifact_error = None, str(exc)

    if bundle is None:
        st.error("Model artifacts not found. Please run train_model.py locally and commit the exported models to the repository.")
        if artifact_error:
            st.caption(f"Artifact recovery detail: {safe_text(artifact_error, 500)}")
        st.stop()

    supabase = get_supabase_client()
    with st.sidebar:
        st.header("Data ingestion")
        upload = st.file_uploader("Upload CSV with exactly the 40 trained features", type=["csv"])
        st.caption(f"Upload limit {MAX_UPLOAD_MB} MB · {MAX_ROWS:,} rows")
        st.divider()
        st.subheader("System status")
        st.write(("🟢" if bundle else "🔴") + " Trained model artifacts")
        st.write("🟡 Isolation Forest checks rows before classification")
        st.write(("🟢" if supabase else "⚪") + " Supabase incident ledger")
        if artifact_error:
            st.error(artifact_error)

    if bundle is None:
        st.info("Place locally trained artifacts in artifacts/ before running analysis.")
        st.code("python train_model.py --data-dir ./CICIDS2017 --output-dir artifacts")
        st.stop()

    features = bundle["features"]
    if upload is None:
        cards = st.columns(4)
        for col, label in zip(cards, ["Total Flows", "Threats", "OOD Anomalies", "Mean Triage Time"]):
            col.metric(label, "—")
        st.info("Upload a prepared CSV containing exactly the 40 feature columns from model_metadata.json. Do not include Label, IP, timestamp, or unrelated columns.")
        st.stop()
    if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Upload rejected: file exceeds {MAX_UPLOAD_MB} MB.")
        st.stop()
    try:
        raw = pd.read_csv(upload, low_memory=False)
    except Exception as exc:
        st.error(f"CSV parsing failed: {safe_text(exc)}")
        st.stop()
    try:
        X = validate_input_frame(raw, features)
    except ValueError as exc:
        st.error(str(exc))
        st.caption("Select exactly the 40 features in model_metadata.json. The app never zero-fills or guesses missing measurements.")
        st.stop()

    started = time.perf_counter()
    ood = np.asarray(bundle["ood"].predict(X)) == -1
    pred = np.full(len(X), -1, dtype=np.int32)
    confidence = np.full(len(X), np.nan, dtype=np.float32)
    trusted = np.flatnonzero(~ood)
    if len(trusted):
        probabilities = bundle["model"].predict_proba(X.iloc[trusted])
        pred[trusted] = np.argmax(probabilities, axis=1)
        confidence[trusted] = np.max(probabilities, axis=1)
        del probabilities
    elapsed_ms = (time.perf_counter() - started) * 1000
    class_names = [str(v) for v in bundle["encoder"].classes_]
    labels = ["Unknown Novel Anomaly" if ood[i] else class_names[int(pred[i])] for i in range(len(X))]
    benign_names = {"BENIGN", "BENIGN TRAFFIC", "NORMAL", "0"}
    malicious = np.array([not ood[i] and labels[i].strip().upper() not in benign_names for i in range(len(X))])
    severity = ["Unknown" if ood[i] else ("Critical" if malicious[i] and confidence[i] >= .9 else "High" if malicious[i] else "Low") for i in range(len(X))]
    alert_ids = [f"TL-{uuid.uuid4().hex[:10].upper()}" for _ in range(len(X))]
    results = pd.DataFrame({
        "Alert ID": alert_ids, "Threat Class": labels, "Class Confidence": confidence,
        "Distribution Status": np.where(ood, "Unknown Novel Anomaly", "In-distribution"),
        "Severity": severity,
    })

    metrics = st.columns(4)
    metrics[0].metric("Total Flows", f"{len(X):,}")
    metrics[1].metric("Predicted Threats", f"{int(malicious.sum()):,}")
    metrics[2].metric("OOD Anomalies", f"{int(ood.sum()):,}")
    metrics[3].metric("Mean Inference Time", f"{elapsed_ms / max(1, len(X)):.3f} ms/flow")
    st.subheader("Triage Overview")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button("Download triage results", data=results.to_csv(index=False).encode(),
        file_name="threatlens_triage_results.csv", mime="text/csv")

    alert_tab, ood_tab, ledger_tab = st.tabs(["Threat investigation", "Unknown/OOD anomalies", "SOC incident ledger"])
    with alert_tab:
        positions = np.flatnonzero(malicious)[:MAX_ALERTS]
        if not len(positions):
            st.success("No in-distribution malicious predictions were produced.")
        else:
            explainer = None
            for pos in positions:
                attack = labels[pos]
                with st.expander(f"{severity[pos]} · {attack} · {alert_ids[pos]}"):
                    xrow = X.iloc[[pos]]
                    evidence, items = "SHAP unavailable; no feature-impact claim is made.", []
                    try:
                        import shap
                        if explainer is None:
                            explainer = shap.TreeExplainer(bundle["model"])
                        impacts = shap_values_for_class(explainer, xrow, int(pred[pos]))
                        evidence, items = make_evidence(xrow.iloc[0], impacts, features)
                    except Exception as exc:
                        st.warning(f"SHAP explanation unavailable ({type(exc).__name__}).")
                    try:
                        mitre, match = retrieve_mitre(attack)
                    except Exception as exc:
                        mitre, match = MITRE_MAP.get(attack, DEFAULT_ATTACK), None
                        st.warning(f"Semantic retrieval unavailable ({type(exc).__name__}); showing static guidance.")
                    left, right = st.columns([1, 1.4])
                    with left:
                        st.markdown("**Prediction evidence**")
                        st.write(f"Class: **{attack}**")
                        st.write(f"XGBoost class confidence: **{confidence[pos] * 100:.1f}%**")
                        st.write(evidence)
                        if items:
                            st.bar_chart(pd.DataFrame(items).set_index("feature")[["impact"]])
                            st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
                        st.markdown("**Contextual Match Confidence**")
                        st.write(f"{match:.1f}%" if match is not None else "Unavailable")
                        st.caption("Semantic proximity in the curated reference corpus, not a calibrated attack probability.")
                    with right:
                        st.markdown("**Why this matters**")
                        st.write(f"{attack} was assigned by the trained classifier. Review the feature evidence and corroborate it with independent telemetry before containment.")
                        narrative_key = f"narrative-{alert_ids[pos]}"
                        if st.button("Generate AI Narrative", key=f"generate-{alert_ids[pos]}"):
                            st.session_state[narrative_key] = generate_narrative(attack, evidence)
                        st.write(st.session_state.get(narrative_key, evidence))
                        st.markdown(f"**MITRE ATT&CK context:** {mitre.get('technique', DEFAULT_ATTACK['technique'])}")
                        st.write(mitre.get("description", DEFAULT_ATTACK["description"]))
                        st.markdown("**Suggested defensive validation / mitigation**")
                        st.write(mitre.get("mitigation", DEFAULT_ATTACK["mitigation"]))
                        if st.button("Save metadata to SOC Ledger", key=f"save-{alert_ids[pos]}"):
                            if supabase is None:
                                st.error("Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY in Streamlit secrets.")
                            else:
                                safe_row = {
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "alert_id": safe_text(alert_ids[pos], 40),
                                    "threat_class": safe_text(attack, 80),
                                    "top_shap_features": [{"feature": safe_text(x["feature"], 80), "impact": round(float(x["impact"]), 6)} for x in items[:3]],
                                    "mitre_technique": safe_text(mitre.get("technique", "Unassigned"), 120),
                                }
                                try:
                                    supabase.table("incident_ledger").insert(safe_row).execute()
                                    st.success("Only allowlisted incident metadata was saved; raw network rows were not sent.")
                                except Exception as exc:
                                    st.error(f"Ledger write failed ({type(exc).__name__}); check table schema/permissions.")
                    del xrow
                    gc.collect()

    with ood_tab:
        st.warning("Isolation Forest flagged these rows as outside the model's learned distribution. They bypassed XGBoost classification, SHAP, and MITRE retrieval.")
        view = results.loc[ood].copy()
        if view.empty:
            st.success("No rows were flagged as out-of-distribution.")
        else:
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.caption("No class is assigned to OOD rows. Investigate using independent telemetry.")
    with ledger_tab:
        if supabase is None:
            st.info("Configure SUPABASE_URL and SUPABASE_KEY in Streamlit secrets to enable the ledger.")
        else:
            try:
                records = supabase.table("incident_ledger").select(
                    "timestamp,alert_id,threat_class,top_shap_features,mitre_technique"
                ).order("timestamp", desc=True).limit(500).execute().data or []
                st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True) if records else st.info("The incident ledger is empty.")
            except Exception as exc:
                st.error(f"Ledger read failed ({type(exc).__name__}); check table permissions.")


if __name__ == "__main__":
    main()
