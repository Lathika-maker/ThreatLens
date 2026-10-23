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
MAX_UPLOAD_MB = 100
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
THREAT_TAXONOMY: dict[str, str] = {
    "BENIGN": "Benign",
    "Bot": "Malicious - Command and Control",
    "DDoS": "Malicious - Denial of Service",
    "DoS Hulk": "Malicious - Denial of Service",
    "DoS GoldenEye": "Malicious - Denial of Service",
    "DoS slowloris": "Malicious - Denial of Service",
    "DoS Slowhttptest": "Malicious - Denial of Service",
    "PortScan": "Malicious - Reconnaissance",
    "FTP-Patator": "Malicious - Brute Force",
    "SSH-Patator": "Malicious - Brute Force",
    "Web Attack – Brute Force": "Malicious - Brute Force",
    "Web Attack – SQL Injection": "Malicious - Web Attack",
    "Web Attack – XSS": "Malicious - Web Attack",
    "Web Attack – Sql Injection": "Malicious - Web Attack",
    "Infiltration": "Malicious - Privilege/Infiltration",
    "Heartbleed": "Malicious - Exploitation",
}


def threat_taxonomy_category(label: str) -> str:
    """Resolve common label spelling variants into the CICIDS threat taxonomy."""
    canonical = canonical_attack_label(label)
    if canonical in THREAT_TAXONOMY:
        return THREAT_TAXONOMY[canonical]
    folded = str(canonical).casefold()
    if folded in {"benign", "benign traffic", "normal", "0"}:
        return "Benign"
    if "dos" in folded or "ddos" in folded:
        return "Malicious - Denial of Service"
    if "patator" in folded or "brute force" in folded:
        return "Malicious - Brute Force"
    if "sql injection" in folded or "xss" in folded or "web attack" in folded:
        return "Malicious - Web Attack"
    if "infiltration" in folded:
        return "Malicious - Privilege/Infiltration"
    if "heartbleed" in folded:
        return "Malicious - Exploitation"
    return "Unknown / Requires validation"


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


def car_response_plan(attack: str, severity_value: str, is_ood: bool) -> dict[str, list[str]]:
    """Evidence-led Contain, Analyze, Remediate response suggestions; never auto-executes actions."""
    if is_ood:
        return {
            "Contain": ["Do not auto-block from an anomaly score alone; preserve the flow and consider temporary containment only if corroborating telemetry supports active harm."],
            "Analyze": ["Correlate source/destination, endpoint, DNS, identity, firewall and application logs; compare with a known-good baseline."],
            "Remediate": ["Address only the confirmed root cause; tune detections after review and document the evidence."],
        }
    attack_l = attack.casefold()
    if "ddos" in attack_l or "dos " in attack_l:
        contain = ["Validate service impact and traffic surge; apply scoped rate limits or upstream DDoS controls if confirmed."]
        remediate = ["Harden exposed services, tune rate limiting and capacity protections, and verify service recovery."]
    elif "brute force" in attack_l or "patator" in attack_l:
        contain = ["Review authentication telemetry; temporarily block or rate-limit confirmed abusive sources and protect affected accounts."]
        remediate = ["Reset potentially exposed credentials, enforce MFA, tune lockout controls, and remove exposed services where unnecessary."]
    elif "portscan" in attack_l:
        contain = ["Confirm the scanner and asset scope; apply narrowly scoped filtering only if unauthorized scanning is confirmed."]
        remediate = ["Reduce unnecessary exposure, review firewall rules, and tune scan detection thresholds."]
    elif "sql injection" in attack_l or "xss" in attack_l or "web attack" in attack_l:
        contain = ["Correlate WAF, web-server and application logs; block a confirmed malicious request pattern with a scoped rule."]
        remediate = ["Use parameterized queries and context-aware output encoding, patch affected components, and add regression tests."]
    elif "heartbleed" in attack_l:
        contain = ["Identify affected TLS endpoints and restrict exposure while verifying vulnerable software versions."]
        remediate = ["Patch or replace vulnerable TLS libraries and rotate secrets if exposure is confirmed."]
    else:
        contain = ["Validate the alert with independent telemetry before blocking a host, account or connection."]
        remediate = ["Remediate the confirmed root cause, patch affected assets where applicable, and verify the fix."]
    analyze = ["Review the top contributing features and model score; correlate with independent endpoint, network and identity evidence. A model score is not proof of compromise."]
    return {"Contain": contain, "Analyze": analyze, "Remediate": remediate}


def get_soc_ledger_path() -> Path:
    """Use a stable path beside app.py, regardless of Streamlit's working directory."""
    return Path(__file__).resolve().parent / "soc_ledger.csv"


def append_soc_ledger(alert_id: str, threat_class: str, severity_value: str) -> Path:
    """Append minimal incident metadata locally; raw telemetry is never written."""
    ledger_path = get_soc_ledger_path()
    columns = ["Alert ID", "Threat Class", "Severity", "Timestamp"]
    row = pd.DataFrame([{
        "Alert ID": safe_text(alert_id, 40),
        "Threat Class": safe_text(threat_class, 80),
        "Severity": safe_text(severity_value, 40),
        "Timestamp": datetime.now(timezone.utc).isoformat(),
    }], columns=columns)

    # Create the ledger with a header on first use, then append subsequent records.
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not ledger_path.exists() or ledger_path.stat().st_size == 0
    row.to_csv(
        ledger_path,
        mode="a",
        header=needs_header,
        index=False,
        encoding="utf-8",
        columns=columns,
    )
    return ledger_path


def get_model_feature_evidence(model: Any, feature_names: list[str], top_n: int = 3) -> pd.DataFrame:
    """Map global XGBoost feature importance to the exact feature names."""
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return pd.DataFrame(columns=["Feature", "Importance"])
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) != len(feature_names):
        return pd.DataFrame(columns=["Feature", "Importance"])
    selected = np.argsort(np.abs(values))[::-1][:top_n]
    return pd.DataFrame({"Feature": [feature_names[int(i)] for i in selected],
                         "Importance": [float(values[int(i)]) for i in selected]})

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

    with st.sidebar:
        st.header("Data ingestion")
        upload = st.file_uploader(
            "Upload network-flow CSV",
            type=["csv"],
            max_upload_size=MAX_UPLOAD_MB,
            help="CSV files up to 100 MB. Analysis is limited by the row slider and available memory.",
        )
        analysis_row_limit = st.slider(
            "Rows to analyze",
            min_value=10_000,
            max_value=MAX_ROWS,
            value=DEFAULT_ANALYSIS_ROWS,
            step=10_000,
            help="Choose how many rows from the beginning of the uploaded CSV to analyze. Extra columns are automatically removed and missing model features are zero-filled.",
        )
        st.caption("Columns are aligned automatically to the model's 40 features.")
        suppression_engine = st.sidebar.toggle("🛡️ Enable Anti-Fatigue Suppression", value=True, help="Hide alerts below 90% score and consolidate repeated source-IP / threat-class pairs.")
        if "active_mode" not in st.session_state:
            st.session_state.active_mode = "csv"
        if "current_simulated_row" not in st.session_state:
            st.session_state.current_simulated_row = None
        if "current_alert_context" not in st.session_state:
            st.session_state.current_alert_context = None

        def _set_simulation(mode: str) -> None:
            st.session_state.active_mode = mode
            st.session_state.current_simulated_row = None
            st.session_state.current_alert_context = {"mode": mode, "selected_at": datetime.now(timezone.utc).isoformat()}

        st.button("Simulate SQLi Attack", key="simulate_sqli_btn",
                  on_click=_set_simulation, args=("sqli",),
                  help="Generate a deterministic test row from the SQL Injection vault prototype.")
        st.button("Simulate XSS Attack", key="simulate_xss_btn",
                  on_click=_set_simulation, args=("xss",),
                  help="Generate a deterministic test row from the XSS vault prototype.")
        st.button("Test MITRE Pipeline", key="test_mitre_btn",
                  on_click=_set_simulation, args=("mitre",),
                  help="Run the end-to-end MITRE mapping smoke test using the SQL Injection prototype.")
        st.button("Analyze uploaded CSV", key="analyze_uploaded_csv_btn",
                  on_click=_set_simulation, args=("csv",),
                  help="Switch from simulation mode back to the uploaded CSV.")
        st.caption(f"File limit {MAX_UPLOAD_MB} MB · up to {analysis_row_limit:,} rows per analysis")
        st.divider()
        st.subheader("System status")
        st.write(f"{'🟢' if bundle is not None and bundle.get('model') is not None else '🔴'} Local ML Engine: {'Online' if bundle is not None and bundle.get('model') is not None else 'Unavailable'}")
        faiss_ready = bool(bundle is not None and bundle.get("index") is not None)
        st.write(f"{'🟢' if faiss_ready else '🔴'} FAISS: {'Loaded' if faiss_ready else 'Not initialized'}")
        qpu_ready = bool(bundle is not None and bundle.get("qpu_simulator") is not None)
        st.write(f"{'🟢' if qpu_ready else '🔴'} QPU Simulator: {'Configured' if qpu_ready else 'Not configured (not used by inference)'}")
        local_storage_ready = bool(ARTIFACT_DIR.exists() and os.access(ARTIFACT_DIR, os.R_OK))
        st.write(f"{'🟢' if local_storage_ready else '🔴'} Local data access: {'Available' if local_storage_ready else 'Check artifact-directory access'}")
        st.caption("Badges reflect runtime checks. This build does not route inference through a QPU unless a simulator is explicitly configured.")
        ledger_path = get_soc_ledger_path()
        ledger_writable = ledger_path.parent.exists() and os.access(ledger_path.parent, os.W_OK)
        st.write(f"{'🟢' if ledger_writable else '🔴'} Local SOC ledger: {'Writable' if ledger_writable else 'Not writable'}")
        if artifact_error:
            st.error(artifact_error)

    if bundle is None:
        st.info("Place locally trained artifacts in artifacts/ before running analysis.")
        st.code("python train_model.py --data-dir ./CICIDS2017 --output-dir artifacts")
        st.stop()

    features = bundle["features"]
    active_mode = st.session_state.get("active_mode", "csv")
    simulate_sqli = active_mode == "sqli"
    simulate_xss = active_mode == "xss"
    test_mitre_pipeline = active_mode == "mitre"
    simulate_mode = active_mode in {"sqli", "xss", "mitre"}
    expected_simulation_label = "Web Attack – XSS" if simulate_xss else "Web Attack – SQL Injection"
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
        # Simulation mode also feeds the preview table, so initialize it on every path.
        preview_source = raw.head(10).copy()
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
        # Preserve the full uploaded dataframe for the scrollable preview; inference can still be row-limited.
        preview_source = raw.copy()
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
    class_names = [str(v) for v in bundle["encoder"].classes_]
    pred = np.full(len(X), -1, dtype=np.int32)
    confidence = np.full(len(X), np.nan, dtype=np.float32)
    probability_details: list[list[dict[str, float | str]] | None] = [None] * len(X)
    confidence[vault_hit] = vault_similarities[vault_hit]
    trusted = np.flatnonzero(~vault_hit & ~ood)
    if len(trusted):
        probabilities = bundle["model"].predict_proba(X_scaled.iloc[trusted])
        pred[trusted] = np.argmax(probabilities, axis=1)
        confidence[trusted] = np.max(probabilities, axis=1)
        for local_i, global_i in enumerate(trusted):
            order = np.argsort(probabilities[local_i])[::-1][:min(5, probabilities.shape[1])]
            probability_details[int(global_i)] = [
                {"class": class_names[int(class_idx)], "score": float(probabilities[local_i, class_idx]) * 100.0}
                for class_idx in order
            ]
        del probabilities
    elapsed_ms = (time.perf_counter() - started) * 1000
    labels = [vault_labels[i] if vault_hit[i] else ("Unknown Novel Anomaly" if ood[i] else class_names[int(pred[i])]) for i in range(len(X))]
    benign_names = {"BENIGN", "BENIGN TRAFFIC", "NORMAL", "0"}
    malicious = np.array([not ood[i] and labels[i].strip().upper() not in benign_names for i in range(len(X))])
    severity = ["Unknown" if ood[i] else ("Critical" if malicious[i] and confidence[i] >= .9 else "High" if malicious[i] else "Low") for i in range(len(X))]
    priority = [
        "P1 — Immediate review" if ood[i] or (malicious[i] and confidence[i] >= 0.90)
        else "P2 — High" if malicious[i] and confidence[i] >= 0.70
        else "P3 — Normal" if malicious[i]
        else "P4 — Monitor"
        for i in range(len(X))
    ]
    # Stable IDs keep per-alert UI state attached to the same flow across Streamlit reruns.
    import hashlib
    alert_ids = [
        "TL-" + hashlib.sha256(
            (str(i) + "|" + ",".join(f"{float(v):.12g}" for v in X.iloc[i].to_numpy(dtype=float))).encode("utf-8")
        ).hexdigest()[:10].upper()
        for i in range(len(X))
    ]
    results = pd.DataFrame({
        "Alert ID": alert_ids, "Threat Class": labels, "Severity": severity, "Priority": priority,
        "Class Confidence": confidence,
        "Distribution Status": ["Threat Vault Override" if vault_hit[i] else "Unknown Novel Anomaly" if ood[i] else "In-distribution" for i in range(len(X))],
        "Vault Cosine Similarity": [float(vault_similarities[i]) if vault_hit[i] else np.nan for i in range(len(X))],
        "Severity": severity,
    })

    metrics = st.columns(4)
    metrics[0].metric("Total Flows", f"{len(X):,}")
    metrics[1].metric("Predicted Threats", f"{int(malicious.sum()):,}")
    metrics[2].metric("OOD Anomalies", f"{int(ood.sum()):,}")
    mean_ms = elapsed_ms / max(1, len(X))
    metrics[3].metric("Mean Inference Time", f"{mean_ms:.3f} ms/flow")
    with st.expander("How to read these operational metrics", expanded=False):
        st.markdown(
            f"- **Total Flows ({len(X):,}):** rows actually analyzed; rows beyond the selected analysis limit are excluded.\n"
            f"- **Predicted Threats ({int(malicious.sum()):,}):** rows routed as non-benign by the active pipeline, including Threat Vault overrides where applicable. These are review candidates, not confirmed incidents.\n"
            f"- **OOD Anomalies ({int(ood.sum()):,}):** rows that differ from the training distribution; this is a novelty warning, not an attack label.\n"
            f"- **Mean Inference Time ({mean_ms:.3f} ms/flow):** elapsed analysis time divided by analyzed rows; it measures responsiveness, not detection quality."
        )
    st.subheader("Preprocessed network flow preview")
    preview_df = preview_source.copy().reset_index(drop=True)
    preview_count = min(len(preview_df), len(labels))
    # Prediction labels are only assigned to rows actually analyzed by the model.
    preview_df["Predicted Class"] = pd.Series(labels[:preview_count], dtype="object").reindex(range(len(preview_df)))
    preview_df["Threat Category"] = preview_df["Predicted Class"].map(
        lambda value: threat_taxonomy_category(str(value)) if pd.notna(value) else "Unknown / Requires validation"
    )
    preview_df["Analysis Status"] = np.where(
        np.arange(len(preview_df)) < preview_count, "Analyzed", "Not analyzed (row limit)"
    )
    preview_df["Review Flag"] = np.where(
        preview_df["Threat Category"].eq("Benign"), "🟢 Benign",
        np.where(preview_df["Predicted Class"].notna(), "🔴 Review threat", "🟠 Not analyzed")
    )

    if not preview_df.empty:
        # Keep one unified review table. Pandas Styler creates per-cell CSS and can
        # exceed Streamlit's rendering limit for large uploads, so style a bounded
        # head and use Streamlit's native virtualized dataframe for the full dataset.
        STYLE_PREVIEW_ROWS = 1000

        def _flow_row_style(row):
            category = str(row.get("Threat Category", "Unknown / Requires validation"))
            status = str(row.get("Analysis Status", ""))
            if status != "Analyzed":
                background, foreground = "#332b16", "#f6d365"
            elif category == "Benign":
                background, foreground = "#163d2b", "#d8ffe8"
            else:
                background, foreground = "#5b2028", "#ffe4e6"
            return [f"background-color: {background}; color: {foreground};" for _ in row]

        st.markdown("**Threat coloring:** 🟩 Benign · 🟥 Malicious / requires validation · 🟨 Not analyzed")
        display_df = preview_df.copy()
        styled_head = display_df.head(STYLE_PREVIEW_ROWS).style.apply(_flow_row_style, axis=1)
        # Fit small tables to their actual row count to avoid empty-looking filler rows.
        table_height = min(400, max(95, 38 * (min(len(display_df), 10) + 1)))
        if len(display_df) <= STYLE_PREVIEW_ROWS:
            st.dataframe(styled_head, use_container_width=True, height=table_height, hide_index=True)
        else:
            st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)
            st.caption(
                f"All {len(display_df):,} rows are in this single scrollable table. For performance, "
                f"cell-level coloring is limited to datasets with {STYLE_PREVIEW_ROWS:,} rows or fewer; "
                "larger tables retain prediction labels and review flags for every row."
            )
        st.caption(
            "Why a row is red: the active detection pipeline predicted a non-benign class or marked it for review. "
            "Green means predicted benign; amber means it was outside the selected analysis limit. "
            "The model/Threat Vault routing chooses the class; color is only a display aid. "
            "Predictions are investigation candidates, not proof of compromise."
        )

    st.subheader("Triage Overview")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button("Download triage results", data=results.to_csv(index=False).encode(),
        file_name="threatlens_triage_results.csv", mime="text/csv")

    alert_tab, ood_tab, ledger_tab = st.tabs(["Threat investigation", "Unknown/OOD anomalies", "SOC incident ledger"])
    with alert_tab:
        candidate_positions = np.flatnonzero(malicious)
        # Keep a stable, score-descending copy of all malicious candidates before suppression.
        score_for_triage = np.where(vault_hit, vault_similarities, confidence).astype(float)
        score_for_triage = np.nan_to_num(score_for_triage, nan=-1.0, posinf=1.0, neginf=-1.0)
        original_alerts = candidate_positions[
            np.argsort(score_for_triage[candidate_positions])[::-1]
        ]
        positions = original_alerts.copy()
        alerts_suppressed = 0

        if suppression_engine and len(original_alerts):
            # Use the uploaded source-IP column when available. Otherwise, keep rows
            # distinct rather than accidentally suppressing unrelated events.
            source_ip_column = next(
                (column for column in preview_source.columns
                 if normalize_column(str(column)).lower() in {"source_ip", "src_ip", "sourceip", "srcip"}),
                None,
            )
            seen_alert_keys = set()
            kept_positions = []
            for pos in original_alerts:
                score = float(score_for_triage[pos])
                if score <= 0.90:
                    continue
                if source_ip_column is not None and pos < len(preview_source):
                    source_ip = str(preview_source.iloc[int(pos)][source_ip_column]).strip()
                    if not source_ip or source_ip.lower() in {"nan", "none"}:
                        source_ip = f"row:{int(pos)}"
                else:
                    source_ip = f"row:{int(pos)}"
                threat_class = str(labels[pos])
                dedupe_key = (source_ip, threat_class)
                # Already sorted by score: first occurrence is the highest-scoring alert.
                if dedupe_key in seen_alert_keys:
                    continue
                seen_alert_keys.add(dedupe_key)
                kept_positions.append(int(pos))

            positions = np.asarray(kept_positions, dtype=int)
            alerts_suppressed = len(original_alerts) - len(positions)
            st.success(
                f"Anti-Fatigue Engine Active: Auto-suppressed {alerts_suppressed} redundant or low-confidence alerts."
            )
        else:
            # Suppression disabled: preserve the existing raw top-alert list.
            positions = original_alerts[:MAX_ALERTS]

        positions = positions[:MAX_ALERTS]
        if not len(positions):
            st.success("No alerts remain in the triage list under the current suppression settings.")
        else:
            for pos in positions:
                attack = labels[pos]
                with st.expander(f"{priority[pos]} · {severity[pos]} · {attack} · {alert_ids[pos]}"):
                    xrow = X_scaled.iloc[[pos]]
                    evidence, items = "Row-level SHAP evidence is unavailable.", []
                    if vault_hit[pos]:
                        evidence = (f"Threat Vault override matched {attack!r} with cosine similarity {vault_similarities[pos]:.4f}. "
                                    "This is prototype similarity, not a calibrated attack probability; validate with independent telemetry.")
                    else:
                        try:
                            import shap
                            shap_explainer = shap.TreeExplainer(bundle["model"])
                            impacts = shap_values_for_class(shap_explainer, xrow, int(pred[pos]))
                            impact_values = np.asarray(impacts, dtype=float).reshape(-1)
                            if len(impact_values) == len(features) and np.isfinite(impact_values).all():
                                top_indices = np.argsort(np.abs(impact_values))[::-1][:3]
                                items = [{"feature": features[int(i)], "shap_value": float(impact_values[int(i)]),
                                          "absolute_shap": float(abs(impact_values[int(i)]))} for i in top_indices]
                                evidence = "Row-specific SHAP values explain feature contributions to this model output; they are not attack probabilities."
                        except Exception as exc:
                            evidence = f"SHAP could not be calculated for this row ({type(exc).__name__}). The prediction pipeline is unchanged."
                    canonical_attack = canonical_attack_label(attack)
                    mitre = get_mitre_details(canonical_attack)
                    match = None
                    try:
                        retrieved_mitre, match = retrieve_mitre(attack)
                        if canonical_attack not in MITRE_MAP:
                            mitre = retrieved_mitre
                    except Exception:
                        # Keep static MITRE guidance, but do not fabricate a contextual score.
                        match = None
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
                        st.markdown("### SHAP mathematical evidence — top 3 features")
                        if items and "shap_value" in items[0]:
                            shap_df = pd.DataFrame(items, columns=["feature", "shap_value", "absolute_shap"])
                            st.bar_chart(shap_df.set_index("feature")["shap_value"], horizontal=True)
                            st.dataframe(shap_df, use_container_width=True, hide_index=True)
                        elif vault_hit[pos]:
                            st.info("This row was routed by Threat Vault similarity; XGBoost SHAP values are not applicable to the override.")
                        else:
                            st.info(evidence)
                        st.caption("Absolute SHAP values rank this row's feature effects on the model output; they are not attack probabilities or causal proof.")
                        if probability_details[pos]:
                            st.markdown("**Model class-score comparison**")
                            prob_df = pd.DataFrame(probability_details[pos])
                            st.bar_chart(prob_df.set_index("class")[["score"]])
                            st.caption("Top XGBoost class scores for this flow. These scores are not calibrated probabilities and do not prove an attack occurred.")
                        elif vault_hit[pos]:
                            st.info("This alert was assigned by Threat Vault prototype similarity, not XGBoost class probabilities.")
                        st.markdown("**Contextual Match Confidence**")
                        st.write(f"{match:.1f} / 100" if match is not None else "Unavailable")
                        st.caption(
                            "Similarity-derived relevance to the nearest local reference entry, not a calibrated attack probability. "
                            "Unavailable means local embedding/index retrieval did not complete."
                        )
                    with right:
                        st.markdown("### FAISS retrieval and SOC playbook")
                        st.markdown(f"**MITRE ATT&CK Tactic:** {mitre.get('tactic', DEFAULT_ATTACK['tactic'])}")
                        st.markdown(f"**FAISS contextual match confidence:** {match:.1f}%" if match is not None else "**FAISS retrieval distance:** Unavailable")
                        st.markdown("**Quantum edge-case resolver:** Not eligible")
                        st.markdown("**SOC mitigation playbook**")
                        _car_plan = car_response_plan(attack, severity[pos], bool(ood[pos]))
                        st.markdown(f"- **Contain:** {_car_plan.get('Contain', ['Validate before containment.'])[0]}")
                        st.markdown(f"- **Investigate:** {_car_plan.get('Analyze', ['Correlate independent telemetry.'])[0]}")
                        st.markdown(f"- **Remediate:** {_car_plan.get('Remediate', ['Fix the confirmed root cause and verify the fix.'])[0]}")
                        st.markdown("**Why this matters**")
                        st.write(f"{attack} was assigned by the model pipeline. Review the evidence and corroborate it with independent telemetry before containment.")
                        st.markdown(f"**Severity:** {severity[pos]} · **Priority:** {priority[pos]}")
                        narrative_key = f"narrative-{alert_ids[pos]}"
                        if st.button("Generate AI Narrative", key=f"generate-{alert_ids[pos]}"):
                            top_features = ", ".join(item["feature"] for item in items[:3]) or "feature importance unavailable"
                            score_kind = "Threat Vault cosine similarity" if vault_hit[pos] else "XGBoost class score"
                            score_pct = float(np.clip(confidence[pos], 0.0, 1.0)) * 100.0 if np.isfinite(confidence[pos]) else 0.0
                            st.session_state[narrative_key] = (
                                f"AI Analyst Note — {alert_ids[pos]}: telemetry was classified as {attack}. "
                                f"The displayed {score_kind} is {score_pct:.1f}%; it is not a calibrated probability. "
                                f"Top model-level features: {top_features}. Correlate with independent network, endpoint, identity, "
                                f"and service telemetry before containment. Suggested validation/remediation: "
                                f"{mitre.get('mitigation', DEFAULT_ATTACK['mitigation'])}"
                            )
                        st.markdown("**Generative AI analyst narrative**")
                        narrative_text = st.session_state.get(narrative_key)
                        if narrative_text:
                            st.info(narrative_text)
                        else:
                            st.caption("Select Generate AI Narrative to create a contextual forensic note for this alert.")
                        st.markdown(f"**MITRE ATT&CK tactic:** {mitre.get('tactic', DEFAULT_ATTACK['tactic'])}")
                        st.markdown(f"**MITRE ATT&CK technique:** {mitre.get('technique', DEFAULT_ATTACK['technique'])}")
                        st.write(mitre.get("description", DEFAULT_ATTACK["description"]))
                        st.markdown("**CAR response recommendations — Contain, Analyze, Remediate**")
                        response_plan = car_response_plan(attack, severity[pos], bool(ood[pos]))
                        for phase, actions in response_plan.items():
                            st.markdown(f"**{phase}**")
                            for action in actions:
                                st.markdown(f"- {action}")
                        with st.expander("MITRE mitigation reference", expanded=False):
                            st.write(mitre.get("mitigation", DEFAULT_ATTACK["mitigation"]))
                            st.caption("Recommendations are advisory. Validate evidence and authorization before taking containment or remediation action.")

                        st.markdown("**Cryptographic Chain-of-Custody**")
                        try:
                            import hashlib
                            raw_values = np.asarray(X.iloc[pos], dtype=np.float64)
                            canonical_bytes = json.dumps(
                                [None if not np.isfinite(v) else float(v) for v in raw_values],
                                separators=(",", ":"), ensure_ascii=False,
                            ).encode("utf-8")
                            flow_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
                            shap_report_features = [
                                {"feature": item.get("feature"), "shap_value": item.get("shap_value"),
                                 "absolute_shap": item.get("absolute_shap")}
                                for item in items[:3]
                            ]
                            forensic_report = {
                                "report_type": "ThreatLens Incident Response Report",
                                "schema_version": "1.0",
                                "alert_id": alert_ids[pos],
                                "threat_class": attack,
                                "severity": severity[pos],
                                "priority": priority[pos],
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "flow_feature_sha256": flow_sha256,
                                "hash_description": "SHA-256 of the ordered, aligned numeric feature values used for inference; it verifies this representation, not source-file authenticity.",
                                "shap_top_features": shap_report_features,
                                "mitre_attack": {"tactic": mitre.get("tactic", DEFAULT_ATTACK["tactic"]),
                                                 "technique": mitre.get("technique", DEFAULT_ATTACK["technique"]),
                                                 "contextual_match_confidence_percent": match},
                                "faiss_mitigation_steps": _car_plan,
                                "mitigation_reference": mitre.get("mitigation", DEFAULT_ATTACK["mitigation"]),
                                "limitations": [
                                    "A hash alone does not establish chain of custody or prove source authenticity.",
                                    "Threat Vault similarity and model scores are not calibrated attack probabilities.",
                                    "Recommendations require analyst validation before action.",
                                ],
                            }
                            st.caption(f"Flow feature SHA-256: {flow_sha256}")
                            st.download_button(
                                "Generate Forensic IR Report",
                                data=json.dumps(forensic_report, indent=2, ensure_ascii=False).encode("utf-8"),
                                file_name=f"threatlens_forensic_{alert_ids[pos]}.json",
                                mime="application/json",
                                key=f"forensic-report-{alert_ids[pos]}",
                            )
                        except Exception as exc:
                            st.warning(f"Forensic report could not be prepared ({type(exc).__name__}).")

                        if st.button("Save metadata to SOC Ledger", key=f"save-{alert_ids[pos]}"):
                            try:
                                saved_path = append_soc_ledger(alert_ids[pos], attack, severity[pos])
                                st.success(f"Incident metadata saved to {saved_path.resolve()}. Raw telemetry was not saved.")
                            except OSError as exc:
                                st.error(f"Local ledger save failed ({type(exc).__name__}); check folder permissions.")
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
        ledger_path = get_soc_ledger_path()
        st.caption(f"Local ledger file: `{ledger_path}`")
        if ledger_path.exists() and ledger_path.stat().st_size > 0:
            try:
                ledger_df = pd.read_csv(ledger_path)
                if "Timestamp" in ledger_df.columns:
                    ledger_df = ledger_df.sort_values("Timestamp", ascending=False)
                st.dataframe(ledger_df, use_container_width=True, hide_index=True)
                st.download_button("Download SOC ledger", data=ledger_df.to_csv(index=False).encode("utf-8"),
                    file_name="soc_ledger.csv", mime="text/csv")
            except Exception as exc:
                st.error(f"Could not read local SOC ledger ({type(exc).__name__}).")
        else:
            st.info("No incidents saved yet. Open an alert and select Save metadata to SOC Ledger.")
        st.caption("The local ledger stores Alert ID, Threat Class, Severity, and Timestamp only.")


if __name__ == "__main__":
    main()
