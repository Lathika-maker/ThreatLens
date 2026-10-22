"""ThreatLens: evidence-first SOC triage for CICIDS2017 network flows.

Train model artifacts locally with train_model.py before deploying this app.
Uploaded CSVs are normalized through a Universal Ingest Layer that aligns them
to the 40 model features, drops unexpected columns, and zero-fills missing values.
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
import streamlit as st
from scipy.spatial.distance import cosine

st.set_page_config(page_title="ThreatLens | SOC Co-Pilot", page_icon="🛡️", layout="wide")

ARTIFACT_DIR = Path(os.getenv("THREATLENS_ARTIFACT_DIR", "artifacts"))
MAX_FEATURES = 40
MAX_UPLOAD_MB = 80
MAX_ROWS = 2_000_000
DEFAULT_ANALYSIS_ROWS = 500_000
MAX_ALERTS = 100
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
    "Web Attack – XSS": {"tactic": "Initial Access (context-dependent)", "technique": "T1190 — Exploit Public-Facing Application (context-dependent)", "description": "The dataset class suggests cross-site scripting-like web attack traffic. This label alone does not prove successful exploitation.", "mitigation": "Review web and WAF logs, validate output encoding and input handling, and patch affected application components."},
    "Web Attack – Sql Injection": {"technique": "T1190 — Exploit Public-Facing Application", "description": "Potential injection attempts against an exposed application.", "mitigation": "Review application/WAF logs, validate input handling, and patch vulnerable components."},
    "Heartbleed": {"tactic": "Initial Access (context-dependent)", "technique": "T1190 — Exploit Public-Facing Application (context-dependent)", "description": "The dataset class suggests Heartbleed-like traffic; confirm vulnerable software and exploit evidence.", "mitigation": "Patch affected TLS implementations and rotate secrets if exposure is confirmed."},
    "Web Attack – SQL Injection": {"tactic": "Initial Access (context-dependent)", "technique": "T1190 — Exploit Public-Facing Application", "description": "Potential SQL injection attempts against an exposed application. The dataset label is not proof of successful exploitation.", "mitigation": "Review application and WAF logs, validate parameterized queries and input handling, and patch vulnerable components."},
}
DEFAULT_ATTACK = {"tactic": "Requires validation", "technique": "Technique unassigned", "description": "No direct mapping is configured for this class. A model label is not proof of attribution.", "mitigation": "Validate with independent endpoint, identity, network, and application telemetry."}


def canonical_attack_label(value: str) -> str:
    """Normalize dataset spelling/case/punctuation to stable ThreatLens keys."""
    normalized = " ".join(str(value).replace("\ufffd", "-").replace("–", "-").replace("—", "-").split()).casefold()
    normalized = normalized.replace("sql injection", "sql injection")
    aliases = {
        "web attack - sql injection": "Web Attack – SQL Injection",
        "web attack - sql injection ": "Web Attack – SQL Injection",
        "web attack - sql injection": "Web Attack – SQL Injection",
        "web attack - xss": "Web Attack – XSS",
        "web attack xss": "Web Attack – XSS",
        "web attack - brute force": "Web Attack – Brute Force",
    }
    return aliases.get(normalized, str(value).strip())


def get_mitre_details(attack: str) -> dict[str, str]:
    """Prefer exact canonical class mapping; use semantic retrieval only as fallback."""
    canonical = canonical_attack_label(attack)
    if canonical in MITRE_MAP:
        return MITRE_MAP[canonical]
    folded = canonical.casefold()
    for key, details in MITRE_MAP.items():
        if key.casefold() == folded:
            return details
    return DEFAULT_ATTACK

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


def align_network_features(
    uploaded_df: pd.DataFrame,
    expected_cols: list[str],
    max_rows: int = MAX_ROWS,
) -> pd.DataFrame:
    """Align a network CSV to the model schema and return finite float32 features.

    This is schema alignment, not semantic feature translation: zero-padding a
    missing CICIDS feature does not make an unrelated dataset (e.g. UNSW-NB15)
    statistically equivalent to CICIDS2017. Use only with a documented mapping
    or treat results from foreign schemas as exploratory.
    """
    frame = uploaded_df.copy()
    frame.columns = [normalize_column(c) for c in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError("Duplicate columns after normalization: " + ", ".join(duplicates))
    if frame.empty or len(frame) > max_rows:
        raise ValueError(f"Upload rejected: expected 1–{max_rows:,} rows; got {len(frame):,}.")
    # Discard labels, identifiers, and foreign columns; preserve only exact expected names.
    aligned = frame.reindex(columns=expected_cols, fill_value=0)
    for name in expected_cols:
        aligned[name] = pd.to_numeric(aligned[name], errors="coerce")
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    # Invalid values in present features become zero so a malformed cell cannot crash scaler.
    aligned = aligned.fillna(0)
    return aligned.astype(np.float32)


# Strict local-only artifact loading. No network/cloud download fallback.
ARTIFACT_DIR = Path("./artifacts")
ARTIFACT_FILENAMES = (
    "xgboost_model.joblib", "isolation_forest.joblib", "label_encoder.joblib",
    "imputer.joblib", "scaler.joblib", "threat_vault.json", "model_metadata.json",
)


@st.cache_resource(max_entries=1, show_spinner=False)
def load_artifacts() -> dict[str, Any]:
    """Load artifacts only from ./artifacts; never fetch missing files remotely."""
    paths = {name: ARTIFACT_DIR / name for name in ARTIFACT_FILENAMES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing local artifact(s) in ./artifacts/: " + ", ".join(missing)
            + ". Run `python train_model.py --data-dir ./CICIDS2017 --output-dir ./artifacts` "
              "to generate the local artifacts. No remote retrieval was attempted."
        )

    # Explicit local paths make artifact provenance auditable.
    xgboost_model = joblib.load("./artifacts/xgboost_model.joblib")
    isolation_forest = joblib.load("./artifacts/isolation_forest.joblib")
    label_encoder = joblib.load("./artifacts/label_encoder.joblib")
    imputer = joblib.load("./artifacts/imputer.joblib")
    scaler = joblib.load("./artifacts/scaler.joblib")
    threat_vault = json.loads(paths["threat_vault.json"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["model_metadata.json"].read_text(encoding="utf-8"))

    features = metadata.get("feature_columns")
    if not isinstance(features, list) or len(features) != MAX_FEATURES:
        raise ValueError(f"model_metadata.json must define exactly {MAX_FEATURES} feature_columns.")
    features = [normalize_column(x) for x in features]
    if len(set(features)) != len(features):
        raise ValueError("Feature metadata contains duplicate names after normalization.")
    for class_name, vector in threat_vault.items():
        values = np.asarray(vector, dtype=float)
        if values.shape != (MAX_FEATURES,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid Threat Vault prototype for {class_name!r}: expected {MAX_FEATURES} finite values.")
    if hasattr(xgboost_model, "n_features_in_") and int(xgboost_model.n_features_in_) != len(features):
        raise ValueError("XGBoost feature count differs from model metadata.")
    if hasattr(isolation_forest, "n_features_in_") and int(isolation_forest.n_features_in_) != len(features):
        raise ValueError("Isolation Forest feature count differs from model metadata.")
    return {
        "model": xgboost_model, "ood": isolation_forest, "encoder": label_encoder,
        "imputer": imputer, "scaler": scaler, "vault": threat_vault,
        "metadata": metadata, "features": features,
    }

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


def match_threat_vault(scaled_row: np.ndarray, vault: dict[str, list[float]], threshold: float = 0.95) -> tuple[str | None, float]:
    """Return the highest-similarity rare-class prototype above threshold."""
    row = np.asarray(scaled_row, dtype=float).reshape(-1)
    if row.shape != (MAX_FEATURES,) or not np.isfinite(row).all() or np.linalg.norm(row) <= 1e-12:
        return None, 0.0
    best_label, best_similarity = None, -1.0
    for label, values in vault.items():
        prototype = np.asarray(values, dtype=float).reshape(-1)
        if prototype.shape != row.shape or not np.isfinite(prototype).all() or np.linalg.norm(prototype) <= 1e-12:
            continue
        similarity = 1.0 - float(cosine(row, prototype))
        if np.isfinite(similarity) and similarity > best_similarity:
            best_label, best_similarity = label, similarity
    if best_label is not None and best_similarity > threshold:
        return canonical_attack_label(best_label), best_similarity
    return None, max(0.0, best_similarity)


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
    """Generate an optional evidence-bounded note without system RAM monitoring."""
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
        st.error("Required local model artifacts are missing or invalid in ./artifacts/. Run train_model.py to generate the local artifacts. No remote retrieval was attempted.")
        if artifact_error:
            st.caption(f"Local artifact loading detail: {safe_text(artifact_error, 500)}")
        st.stop()

    supabase = get_supabase_client()
    with st.sidebar:
        st.header("Data ingestion")
        upload = st.file_uploader("Upload network-flow CSV", type=["csv"])
        analysis_row_limit = st.slider(
            "Rows to analyze",
            min_value=10_000,
            max_value=MAX_ROWS,
            value=DEFAULT_ANALYSIS_ROWS,
            step=10_000,
            help="Choose how many rows from the beginning of the uploaded CSV to analyze. Extra columns are automatically removed and missing model features are zero-filled.",
        )
        st.caption("Columns are aligned automatically to the model's 40 features.")
        simulate_sqli = st.button("Simulate SQLi Attack", help="Generate a deterministic test row from the SQL Injection vault prototype.")
        simulate_xss = st.button("Simulate XSS Attack", help="Generate a deterministic test row from the XSS vault prototype.")
        test_mitre_pipeline = st.button("Test MITRE Pipeline", help="Run the end-to-end MITRE mapping smoke test using the SQL Injection prototype.")
        st.caption(f"File limit {MAX_UPLOAD_MB} MB · up to {analysis_row_limit:,} rows per analysis")
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
    simulate_mode = bool(simulate_sqli or simulate_xss or test_mitre_pipeline)
    expected_simulation_label = (
        "Web Attack – XSS" if simulate_xss else "Web Attack – SQL Injection"
    )
    if simulate_mode:
        vault_key = next(
            (key for key in bundle["vault"]
             if canonical_attack_label(key) == expected_simulation_label),
            None,
        )
        if vault_key is None:
            st.error(
                f"{expected_simulation_label} prototype is not present in "
                "artifacts/threat_vault.json. Retrain with the updated train_model.py "
                "and confirm the dataset label spelling."
            )
            st.stop()
        prototype = np.asarray(bundle["vault"][vault_key], dtype=float).reshape(1, -1)
        # Reconstruct raw values so the regular scaler/inference path recreates this prototype.
        mock_values = bundle["scaler"].inverse_transform(prototype)
        raw = pd.DataFrame(mock_values, columns=features)
        X = align_network_features(raw, features, max_rows=1)
        st.info(
            f"Simulation mode: one synthetic row reconstructed from the "
            f"{expected_simulation_label} Threat Vault prototype. This verifies "
            "pipeline wiring, not real attack detection accuracy."
        )
    else:
        if upload is None:
            cards = st.columns(4)
            for col, label in zip(cards, ["Total Flows", "Threats", "OOD Anomalies", "Mean Triage Time"]):
                col.metric(label, "—")
            st.info("Upload a network-flow CSV. The Universal Ingest Layer aligns columns to the model's 40 expected features, drops unknown columns, and zero-fills missing or invalid feature values. This does not semantically translate foreign dataset features.")
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
            # Respect the user-selected row limit without rejecting larger CSVs.
            # Analyze the first N rows; all columns still pass through universal alignment.
            if len(raw) > analysis_row_limit:
                st.warning(
                    f"CSV contains {len(raw):,} rows. Analyzing the first "
                    f"{analysis_row_limit:,} rows based on your slider setting."
                )
                raw = raw.head(analysis_row_limit).copy()

            # Universal Ingest Layer: drop unexpected columns and zero-pad missing features.
            expected_features = features
            X = align_network_features(raw, expected_features, max_rows=analysis_row_limit)
        except ValueError as exc:
            st.error(f"CSV ingestion failed: {safe_text(exc)}")
            st.caption("Check the CSV encoding, duplicate normalized columns, and row count.")
            st.stop()

    started = time.perf_counter()
    X_scaled = pd.DataFrame(bundle["scaler"].transform(X), columns=features)
    vault_labels, vault_similarities = [], np.zeros(len(X), dtype=np.float32)
    for i in range(len(X_scaled)):
        match_label, similarity = match_threat_vault(X_scaled.iloc[i].to_numpy(), bundle["vault"])
        vault_labels.append(match_label)
        vault_similarities[i] = similarity
    vault_hit = np.array([value is not None for value in vault_labels], dtype=bool)
    if simulate_mode:
        # The reconstructed prototype should match itself; fail clearly on mismatched artifacts.
        if not vault_hit[0] or canonical_attack_label(vault_labels[0]) != expected_simulation_label:
            st.error(
                f"{expected_simulation_label} simulation did not trigger the intended "
                f"vault override (match={vault_labels[0]!r}, "
                f"similarity={vault_similarities[0]:.4f}). Check that scaler.joblib "
                "and threat_vault.json were produced by the same training run."
            )
            st.stop()
        st.success(
            f"Threat Vault override verified for {expected_simulation_label} "
            f"(cosine similarity {vault_similarities[0]:.4f})."
        )
    ood = np.zeros(len(X), dtype=bool)
    remaining = np.flatnonzero(~vault_hit)
    if len(remaining):
        ood[remaining] = np.asarray(bundle["ood"].predict(X_scaled.iloc[remaining])) == -1
    pred = np.full(len(X), -1, dtype=np.int32)
    confidence = np.full(len(X), np.nan, dtype=np.float32)
    confidence[vault_hit] = vault_similarities[vault_hit]
    trusted = np.flatnonzero(~vault_hit & ~ood)
    if len(trusted):
        probabilities = bundle["model"].predict_proba(X_scaled.iloc[trusted])
        pred[trusted] = np.argmax(probabilities, axis=1)
        confidence[trusted] = np.max(probabilities, axis=1)
        del probabilities
    elapsed_ms = (time.perf_counter() - started) * 1000
    class_names = [str(v) for v in bundle["encoder"].classes_]
    labels = [vault_labels[i] if vault_hit[i] else ("Unknown Novel Anomaly" if ood[i] else class_names[int(pred[i])]) for i in range(len(X))]
    benign_names = {"BENIGN", "BENIGN TRAFFIC", "NORMAL", "0"}
    malicious = np.array([not ood[i] and labels[i].strip().upper() not in benign_names for i in range(len(X))])
    severity = ["Unknown" if ood[i] else ("Critical" if malicious[i] and confidence[i] >= .9 else "High" if malicious[i] else "Low") for i in range(len(X))]
    alert_ids = [f"TL-{uuid.uuid4().hex[:10].upper()}" for _ in range(len(X))]
    results = pd.DataFrame({
        "Alert ID": alert_ids, "Threat Class": labels, "Class Confidence": confidence,
        "Distribution Status": ["Threat Vault Override" if vault_hit[i] else "Unknown Novel Anomaly" if ood[i] else "In-distribution" for i in range(len(X))],
        "Vault Cosine Similarity": [float(vault_similarities[i]) if vault_hit[i] else np.nan for i in range(len(X))],
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
                    xrow = X_scaled.iloc[[pos]]
                    evidence, items = "SHAP unavailable; no feature-impact claim is made.", []
                    if vault_hit[pos]:
                        evidence = (f"Threat Vault override matched {attack!r} with cosine similarity {vault_similarities[pos]:.4f}. "
                                    "This is prototype similarity, not a calibrated attack probability; validate with independent telemetry.")
                    else:
                        try:
                            import shap
                            if explainer is None:
                                explainer = shap.TreeExplainer(bundle["model"])
                            impacts = shap_values_for_class(explainer, xrow, int(pred[pos]))
                            evidence, items = make_evidence(xrow.iloc[0], impacts, features)
                        except Exception as exc:
                            st.warning(f"SHAP explanation unavailable ({type(exc).__name__}).")
                    canonical_attack = canonical_attack_label(attack)
                    mitre = get_mitre_details(canonical_attack)
                    match = None
                    try:
                        if canonical_attack not in MITRE_MAP:
                            mitre, match = retrieve_mitre(attack)
                    except Exception as exc:
                        st.warning(f"Semantic retrieval unavailable ({type(exc).__name__}); showing static guidance.")
                    left, right = st.columns([1, 1.4])
                    with left:
                        st.markdown("**Prediction evidence**")
                        st.write(f"Class: **{attack}**")
                        if vault_hit[pos]:
                            confidence_pct = float(np.clip(vault_similarities[pos], 0.0, 1.0))
                            st.write(f"Threat Vault cosine similarity: **{confidence_pct * 100:.2f}%**")
                        else:
                            confidence_pct = float(np.clip(confidence[pos], 0.0, 1.0)) if np.isfinite(confidence[pos]) else 0.0
                            st.write(f"XGBoost class confidence: **{confidence_pct * 100:.1f}%**")
                        st.progress(confidence_pct, text="Similarity / model score (not calibrated probability)")
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
                        st.markdown(f"**MITRE ATT&CK tactic:** {mitre.get('tactic', DEFAULT_ATTACK['tactic'])}")
                        st.markdown(f"**MITRE ATT&CK technique:** {mitre.get('technique', DEFAULT_ATTACK['technique'])}")
                        st.write(mitre.get("description", DEFAULT_ATTACK["description"]))
                        with st.expander("Suggested defensive validation / mitigation", expanded=True):
                            st.write(mitre.get("mitigation", DEFAULT_ATTACK["mitigation"]))
                            st.caption("Validate these steps against your environment and independent telemetry before taking containment action.")
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
