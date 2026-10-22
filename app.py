"""ThreatLens: local hybrid classical/quantum network-flow triage MVP.

All traffic analysis runs locally. Hugging Face model weights may be downloaded once
on first launch; set HF_HUB_OFFLINE=1 after caching the encoder to force offline use.
"""
from __future__ import annotations

import hashlib
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / ".cache" / "huggingface"))

import faiss
import numpy as np
import pandas as pd
import pennylane as qml
import shap
import streamlit as st
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

APP_TITLE = "ThreatLens | Hybrid SOC Co-Pilot"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FEATURES = [
    "Destination_Port", "Flow_Duration", "Total_Fwd_Packets",
    "Total_Bwd_Packets", "Flow_Bytes_s", "Packet_Length_Mean", "SYN_Flag_Count",
]
LABEL_COLUMN = "Label"
EDGE_LOW, EDGE_HIGH = 0.40, 0.65
DEFAULT_CHUNK_SIZE = 50_000
MAX_ALERT_CARDS = 50
PREVIEW_ROWS = 500
MAX_QUANTUM_EVALUATIONS = 1_000

# A small, curated local knowledge base. It is an illustrative subset of ATT&CK,
# not a complete or automatically updated ATT&CK corpus.
MITRE: dict[str, dict[str, Any]] = {
    "T1046": {
        "id": "T1046", "name": "Network Service Discovery", "tactic": "Discovery",
        "description": "An actor probes hosts and network services to identify reachable ports and services.",
        "mitigations": [
            "Contain: rate-limit or block the confirmed scanning source at the network boundary.",
            "Investigate: correlate destination ports, targeted hosts, and follow-on connections.",
            "Remediate: close unnecessary services and tune network-scan monitoring.",
        ],
    },
    "T1110": {
        "id": "T1110", "name": "Brute Force", "tactic": "Credential Access",
        "description": "An actor repeatedly attempts to guess passwords or otherwise obtain valid credentials.",
        "mitigations": [
            "Contain: rate-limit or temporarily block the source against the affected service.",
            "Investigate: review identity-provider and host authentication logs for successful logins.",
            "Remediate: enforce MFA, strong credentials, and appropriate lockout controls.",
        ],
    },
    "T1071": {
        "id": "T1071", "name": "Application Layer Protocol", "tactic": "Command and Control",
        "description": "An actor uses application protocols to communicate with systems under their control.",
        "mitigations": [
            "Contain: block confirmed malicious destinations using approved network controls.",
            "Investigate: correlate DNS, proxy, firewall, and endpoint telemetry.",
            "Remediate: restrict outbound access to approved services and destinations.",
        ],
    },
    "T1499": {
        "id": "T1499", "name": "Endpoint Denial of Service", "tactic": "Impact",
        "description": "An actor attempts to exhaust endpoint or service resources so legitimate users are disrupted.",
        "mitigations": [
            "Contain: apply validated rate limits or upstream filtering to the affected service.",
            "Investigate: review packet rates, service health, and related system logs.",
            "Remediate: tune capacity, filtering, and denial-of-service protections.",
        ],
    },
    "T1190": {
        "id": "T1190", "name": "Exploit Public-Facing Application", "tactic": "Initial Access",
        "description": "An actor may exploit a vulnerability in an internet-facing application to gain access.",
        "mitigations": [
            "Contain: restrict suspicious sources where operationally safe.",
            "Investigate: inspect web, application, and host logs for exploit indicators.",
            "Remediate: patch the exposed application and reduce unnecessary public exposure.",
        ],
    },
    "T1078": {
        "id": "T1078", "name": "Valid Accounts", "tactic": "Defense Evasion / Persistence / Privilege Escalation / Initial Access",
        "description": "An actor uses legitimate account credentials to access systems and blend in with normal activity.",
        "mitigations": [
            "Contain: restrict or disable an account when misuse is substantiated.",
            "Investigate: audit sign-ins, privilege changes, and affected systems.",
            "Remediate: rotate credentials and require MFA for privileged accounts.",
        ],
    },
}

# Normalize popular CICIDS2017, common NetFlow, and Zeek field names.
ALIASES = {
    "destinationport": "Destination_Port", "dstport": "Destination_Port",
    "dst_port": "Destination_Port", "dest_port": "Destination_Port",
    "flowduration": "Flow_Duration", "duration": "Flow_Duration",
    "totalfwdpackets": "Total_Fwd_Packets", "tot_fwd_pkts": "Total_Fwd_Packets",
    "fwd_packets": "Total_Fwd_Packets", "forward_packets": "Total_Fwd_Packets",
    "totalbackwardpackets": "Total_Bwd_Packets", "total_bwd_packets": "Total_Bwd_Packets", "total_bwd_pkts": "Total_Bwd_Packets",
    "tot_bwd_pkts": "Total_Bwd_Packets", "bwd_packets": "Total_Bwd_Packets",
    "flowbytes_s": "Flow_Bytes_s", "flow_bytes_s": "Flow_Bytes_s", "flow_bytes_sec": "Flow_Bytes_s",
    "flow_bytes_per_s": "Flow_Bytes_s", "bytes_per_second": "Flow_Bytes_s",
    "flow_packets_s": "Flow_Packets_s", "flow_packets_sec": "Flow_Packets_s",
    "avg_packet_length": "Packet_Length_Mean", "average_packet_size": "Packet_Length_Mean",
    "packet_length_mean": "Packet_Length_Mean", "avg_pkt_size": "Packet_Length_Mean",
    "syn_flag_count": "SYN_Flag_Count", "syn_count": "SYN_Flag_Count",
    "src_ip": "Source_IP", "source_ip": "Source_IP", "id_orig_h": "Source_IP",
    "dst_ip": "Dest_IP", "dest_ip": "Dest_IP", "destination_ip": "Dest_IP", "id_resp_h": "Dest_IP",
    "flow_id": "Flow_ID", "label": "Label", "class": "Label", "attack": "Label",
    "total_len_fwd_pkts": "Total_Length_Fwd_Packets",
    "total_len_bwd_pkts": "Total_Length_Bwd_Packets",
    "totlen_fwd_pkts": "Total_Length_Fwd_Packets",
    "totlen_bwd_pkts": "Total_Length_Bwd_Packets",
    "total_length_of_fwd_packets": "Total_Length_Fwd_Packets",
    "total_length_of_bwd_packets": "Total_Length_Bwd_Packets",
    "fwd_pkt_len_mean": "Fwd_Packet_Length_Mean",
    "bwd_pkt_len_mean": "Bwd_Packet_Length_Mean",
}


def _normal_name(name: Any) -> str:
    import re
    value = str(name).strip().replace("\ufeff", "")
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[/\s.\-]+", "_", value).strip("_").lower()
    compact = value.replace("_", "")
    if compact in ALIASES:
        return ALIASES[compact]
    if value in ALIASES:
        return ALIASES[value]
    return ALIASES.get(value, str(name).strip().replace(" ", "_"))


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map recognizable field aliases to the canonical names consumed by the model."""
    out = frame.copy()
    out.columns = [_normal_name(column) for column in out.columns]
    return out.loc[:, ~out.columns.duplicated()]


def preprocess_network_chunk(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize a chunk, derive supported rates, and disclose missing/imputed features.

    Flow duration derivation assumes microseconds, as used by CICIDS-style exports.
    A dataset with other units should be normalized by the user before analysis.
    """
    out = normalize_columns(frame)
    derived: dict[str, int] = {}

    # Derive mean packet size from directional means if supplied.
    if "Packet_Length_Mean" not in out and {
        "Fwd_Packet_Length_Mean", "Bwd_Packet_Length_Mean"
    }.issubset(out.columns):
        out["Packet_Length_Mean"] = (
            pd.to_numeric(out["Fwd_Packet_Length_Mean"], errors="coerce")
            + pd.to_numeric(out["Bwd_Packet_Length_Mean"], errors="coerce")
        ) / 2.0
        derived["Packet_Length_Mean"] = int(out["Packet_Length_Mean"].notna().sum())

    # A total byte rate can be derived from directional byte counts and duration.
    if "Flow_Bytes_s" not in out and {
        "Total_Length_Fwd_Packets", "Total_Length_Bwd_Packets", "Flow_Duration"
    }.issubset(out.columns):
        seconds = pd.to_numeric(out["Flow_Duration"], errors="coerce") / 1_000_000.0
        total_bytes = (
            pd.to_numeric(out["Total_Length_Fwd_Packets"], errors="coerce")
            + pd.to_numeric(out["Total_Length_Bwd_Packets"], errors="coerce")
        )
        out["Flow_Bytes_s"] = total_bytes / seconds.where(seconds > 0)
        derived["Flow_Bytes_s"] = int(out["Flow_Bytes_s"].notna().sum())

    # Preserve what was unavailable before imputation, rather than hiding it.
    missing_features = [feature for feature in FEATURES if feature not in out.columns]
    invalid_cell_count = 0
    for feature in FEATURES:
        if feature not in out.columns:
            out[feature] = 0.0
            continue
        values = pd.to_numeric(out[feature], errors="coerce")
        invalid_cell_count += int((~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))).sum())
        out[feature] = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "Flow_ID" not in out:
        out.insert(0, "Flow_ID", [f"CSV-{i + 1:08d}" for i in range(len(out))])
    if "Source_IP" not in out:
        out["Source_IP"] = "Unknown"
    if "Dest_IP" not in out:
        out["Dest_IP"] = "Unknown"

    warnings = []
    if missing_features:
        warnings.append("Missing feature columns filled with zero: " + ", ".join(missing_features))
    if invalid_cell_count:
        warnings.append(f"{invalid_cell_count} non-numeric/non-finite values replaced with zero")
    out["Preprocessing_Warning"] = "; ".join(warnings) if warnings else "None"
    report = {
        "rows": int(len(out)),
        "missing_features": missing_features,
        "invalid_numeric_cells_replaced_with_zero": invalid_cell_count,
        "derived_features": derived,
        "warning": "; ".join(warnings) if warnings else "None",
    }
    return out, report


def generate_synthetic_data(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic 500-row demonstration dataset with binary labels."""
    if n_rows < 2:
        raise ValueError("n_rows must be at least 2 to include both classes.")
    rng = np.random.default_rng(seed)
    labels = np.zeros(n_rows, dtype=np.int32)
    labels[rng.choice(n_rows, n_rows // 2, replace=False)] = 1
    benign_n = int((labels == 0).sum())
    malicious_n = n_rows - benign_n
    benign = pd.DataFrame({
        "Destination_Port": rng.choice([53, 80, 443, 8080, 123], benign_n),
        "Flow_Duration": np.exp(rng.uniform(np.log(100_000), np.log(10_000_000), benign_n)),
        "Total_Fwd_Packets": rng.integers(5, 80, benign_n),
        "Total_Bwd_Packets": rng.integers(3, 100, benign_n),
        "Flow_Bytes_s": np.exp(rng.uniform(np.log(1_000), np.log(2_000_000), benign_n)),
        "Packet_Length_Mean": rng.uniform(180, 900, benign_n),
        "SYN_Flag_Count": rng.integers(0, 5, benign_n),
        "Label": 0,
    })
    attack_kind = rng.integers(0, 4, malicious_n)
    malicious = pd.DataFrame({
        "Destination_Port": np.where(attack_kind == 0, rng.integers(1, 65536, malicious_n),
                             np.where(attack_kind == 1, rng.choice([80, 443], malicious_n),
                             np.where(attack_kind == 2, rng.choice([21, 22, 3389], malicious_n),
                                      rng.choice([389, 445, 3389], malicious_n)))),
        "Flow_Duration": np.where(attack_kind == 0, np.exp(rng.uniform(np.log(1), np.log(5_000), malicious_n)),
                          np.where(attack_kind == 1, np.exp(rng.uniform(np.log(10), np.log(50_000), malicious_n)),
                          np.exp(rng.uniform(np.log(30_000), np.log(2_000_000), malicious_n)))),
        "Total_Fwd_Packets": np.where(attack_kind == 1, rng.integers(100, 5_000, malicious_n),
                                      rng.integers(1, 50, malicious_n)),
        "Total_Bwd_Packets": np.where(attack_kind == 1, rng.integers(0, 3, malicious_n),
                                      rng.integers(0, 40, malicious_n)),
        "Flow_Bytes_s": np.where(attack_kind == 1, np.exp(rng.uniform(np.log(1e6), np.log(1e9), malicious_n)),
                                 np.exp(rng.uniform(np.log(10), np.log(1e6), malicious_n))),
        "Packet_Length_Mean": np.where(attack_kind == 1, rng.uniform(40, 150, malicious_n),
                                       rng.uniform(30, 500, malicious_n)),
        "SYN_Flag_Count": np.where(attack_kind == 1, rng.integers(30, 600, malicious_n),
                                   rng.integers(0, 15, malicious_n)),
        "Label": 1,
    })
    data = pd.concat([benign, malicious], ignore_index=True)
    data = data.iloc[rng.permutation(len(data))].reset_index(drop=True)
    data.insert(0, "Flow_ID", [f"SYN-{i + 1:05d}" for i in range(len(data))])
    data.insert(1, "Source_IP", [f"10.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}" for _ in range(len(data))])
    data.insert(2, "Dest_IP", [f"192.168.{rng.integers(0, 256)}.{rng.integers(1, 255)}" for _ in range(len(data))])
    return data


@st.cache_resource(show_spinner=False)
def train_model() -> dict[str, Any]:
    """Train XGBoost on the fixed 500-row generated benchmark; never on upload labels."""
    data = generate_synthetic_data()
    x = data[FEATURES].astype(float)
    y = data["Label"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=2, verbosity=0,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_rows": int(len(y_test)),
    }
    return {"model": model, "metrics": metrics, "train_min": x_train.min(), "train_max": x_train.max()}


@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    """Load MiniLM on CPU; use local cache automatically when it exists."""
    hub = Path(os.environ["HF_HOME"]) / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
    snapshots = hub / "snapshots"
    is_cached = snapshots.is_dir() and any((path / "modules.json").exists() for path in snapshots.iterdir())
    if is_cached:
        return SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_resource(show_spinner=False)
def build_faiss_index() -> tuple[faiss.IndexFlatL2, list[str]]:
    """Embed the six local ATT&CK entries and store 384-D vectors in FAISS."""
    ids = list(MITRE)
    texts = [
        f"{entry['id']} {entry['name']} {entry['tactic']} {entry['description']} "
        + " ".join(entry["mitigations"])
        for entry in MITRE.values()
    ]
    vectors = load_embedder().encode(texts, convert_to_numpy=True).astype(np.float32)
    if vectors.shape != (len(ids), 384):
        raise ValueError(f"Expected embeddings shaped (6, 384), received {vectors.shape}.")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, ids


@st.cache_resource(show_spinner=False)
def build_shap_explainer() -> shap.TreeExplainer:
    """Cache TreeExplainer over the trained local XGBoost model."""
    return shap.TreeExplainer(train_model()["model"])


def predict_chunk(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Classify one preprocessed chunk; Label is never passed to the model."""
    out = frame.copy()
    x = out[FEATURES].astype(np.float32)
    probabilities = bundle["model"].predict_proba(x)[:, 1]
    out["Threat_Probability"] = probabilities
    out["Prediction"] = (probabilities >= 0.5).astype(np.int8)
    out["Prediction_Label"] = np.where(out["Prediction"] == 1, "MALICIOUS", "BENIGN")
    out["Priority"] = np.where((probabilities >= 0.90) & (out["Prediction"] == 1), "HIGH",
                               np.where(out["Prediction"] == 1, "MEDIUM", "—"))
    out["Quantum_Eligible"] = (probabilities >= EDGE_LOW) & (probabilities <= EDGE_HIGH)
    out["Quantum_Routed"] = False
    out["Quantum_Risk_Score"] = np.nan
    out["Quantum_Route_Status"] = np.where(
        out["Quantum_Eligible"], "Eligible; awaiting quantum evaluation", "Not eligible (outside score interval)"
    )
    return out


def _shap_vector(row: pd.Series, bundle: dict[str, Any]) -> np.ndarray:
    """Return a positive-class SHAP vector robust to supported SHAP output shapes."""
    values = build_shap_explainer().shap_values(row[FEATURES].astype(float).to_frame().T)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[0, :, min(1, array.shape[-1] - 1)]
    elif array.ndim == 2:
        array = array[0]
    return np.asarray(array, dtype=float).reshape(-1)


def explain_flow(row: pd.Series, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Return top-three SHAP feature evidence for a single analyzed network flow."""
    values = _shap_vector(row, bundle)
    positions = np.argsort(np.abs(values))[::-1][:3]
    return [
        {"feature": FEATURES[int(i)], "shap_value": float(values[int(i)]),
         "absolute_shap": float(abs(values[int(i)])), "feature_value": float(row[FEATURES[int(i)]])}
        for i in positions
    ]


# PennyLane VQC: a simulation-based edge-case signal, not a calibrated probability.
_QUANTUM_DEVICE = qml.device("default.qubit", wires=4)


@qml.qnode(_QUANTUM_DEVICE)
def _quantum_circuit(inputs: Any, weights: Any) -> Any:
    qml.AngleEmbedding(inputs, wires=range(4), rotation="Y")
    for wire in range(4):
        qml.RY(weights[wire, 0], wires=wire)
        qml.RZ(weights[wire, 1], wires=wire)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])
    return qml.expval(qml.PauliZ(0))


def quantum_risk_score(row: pd.Series, bundle: dict[str, Any]) -> float:
    """Run the four-qubit circuit for ambiguous classifier scores only."""
    raw = row[FEATURES].astype(float).to_numpy()
    low = bundle["train_min"].to_numpy(dtype=float)
    high = bundle["train_max"].to_numpy(dtype=float)
    scaled = np.clip((raw - low) / np.maximum(high - low, 1e-9), 0.0, 1.0) * np.pi
    angles = scaled[:4]
    weights = np.array([[0.31, 0.18], [0.23, 0.27], [0.14, 0.37], [0.41, 0.12]], dtype=float)
    expectation = float(_quantum_circuit(angles, weights))
    return float(np.clip((expectation + 1.0) / 2.0, 0.0, 1.0))


def retrieve_mitre(row: pd.Series, evidence: list[dict[str, Any]],
                    index: faiss.IndexFlatL2, ids: list[str],
                    embedder: SentenceTransformer) -> dict[str, Any]:
    """Find relevant local ATT&CK text using a flow-and-evidence query."""
    features_text = ", ".join(
        f"{item['feature']}={item['feature_value']:.4g} (SHAP={item['shap_value']:+.4g})"
        for item in evidence
    )
    query = (
        f"Network security flow destination port {int(row['Destination_Port'])}; "
        f"duration {row['Flow_Duration']:.4g}; forward packets {row['Total_Fwd_Packets']:.4g}; "
        f"backward packets {row['Total_Bwd_Packets']:.4g}; bytes per second {row['Flow_Bytes_s']:.4g}; "
        f"mean packet length {row['Packet_Length_Mean']:.4g}; SYN flags {row['SYN_Flag_Count']:.4g}. "
        f"Classifier marked {row['Prediction_Label']} with score {row['Threat_Probability']:.3f}. "
        f"Important model features: {features_text}."
    )
    vector = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, positions = index.search(vector, 1)
    technique_id = ids[int(positions[0, 0])]
    return {"query": query, "technique": MITRE[technique_id],
            "retrieved_id": technique_id, "distance": float(distances[0, 0])}


def analyze_csv_in_chunks(source: Any, bundle: dict[str, Any],
                          chunk_size: int = DEFAULT_CHUNK_SIZE,
                          progress: Any | None = None) -> dict[str, Any]:
    """Analyze large CSVs chunk-by-chunk and stream full results to a local temp CSV.

    Only a 500-row preview and the highest-scoring 200 candidate alerts remain in
    memory. The uploaded CSV is never sent to a remote service. The output file is
    local to this Python process and must be treated as sensitive telemetry.
    """
    if hasattr(source, "seek"):
        source.seek(0)
    output = tempfile.NamedTemporaryFile(prefix="threatlens_", suffix=".csv", delete=False)
    output_path = output.name
    output.close()
    preview_parts: list[pd.DataFrame] = []
    alert_candidates = pd.DataFrame()
    missing_features_seen: set[str] = set()
    derived_features_total: dict[str, int] = {}
    invalid_numeric_cells_total = 0
    total_rows = malicious_rows = high_rows = routed_rows = eligible_rows = processed = 0
    quantum_budget_remaining = MAX_QUANTUM_EVALUATIONS
    source_name = getattr(source, "name", "network.csv")
    digest = hashlib.sha256()
    # Rewind after each chunk reader; Pandas manages streaming CSV parsing.
    try:
        chunks = pd.read_csv(source, chunksize=max(1, int(chunk_size)), low_memory=False)
        first_write = True
        with open(output_path, "w", encoding="utf-8", newline="") as result_file:
            for chunk_index, chunk in enumerate(chunks, start=1):
                cleaned, prep = preprocess_network_chunk(chunk)
                predicted = predict_chunk(cleaned, bundle)
                missing_features_seen.update(prep.get("missing_features", []))
                invalid_numeric_cells_total += int(prep.get("invalid_numeric_cells_replaced_with_zero", 0))
                for feature, count in prep.get("derived_features", {}).items():
                    derived_features_total[feature] = derived_features_total.get(feature, 0) + int(count)
                total_rows += len(predicted)
                malicious_rows += int(predicted["Prediction"].sum())
                high_rows += int((predicted["Priority"] == "HIGH").sum())
                eligible_indices = predicted.index[predicted["Quantum_Eligible"]].tolist()
                eligible_rows += len(eligible_indices)
                # Spend the per-upload quantum budget on the scores closest to the
                # classification boundary first. The circuit is never called outside
                # the configured probability interval.
                route_count = min(len(eligible_indices), quantum_budget_remaining)
                if route_count:
                    eligible_scores = predicted.loc[eligible_indices, "Threat_Probability"]
                    selected_indices = (eligible_scores.sub(0.525).abs().nsmallest(route_count).index.tolist())
                    for route_index in selected_indices:
                        quantum_value = quantum_risk_score(predicted.loc[route_index], bundle)
                        predicted.at[route_index, "Quantum_Risk_Score"] = quantum_value
                        predicted.at[route_index, "Quantum_Routed"] = True
                        predicted.at[route_index, "Quantum_Route_Status"] = "Evaluated by PennyLane default.qubit"
                    quantum_budget_remaining -= route_count
                    routed_rows += route_count
                for route_index in eligible_indices:
                    if not bool(predicted.at[route_index, "Quantum_Routed"]):
                        predicted.at[route_index, "Quantum_Route_Status"] = (
                            "Eligible but skipped: per-upload quantum simulation cap reached"
                        )
                if sum(len(part) for part in preview_parts) < PREVIEW_ROWS:
                    remaining = PREVIEW_ROWS - sum(len(part) for part in preview_parts)
                    preview_parts.append(predicted.head(remaining).copy())
                candidates = predicted.loc[predicted["Prediction"] == 1].nlargest(
                    200, "Threat_Probability"
                )
                if not candidates.empty:
                    alert_candidates = pd.concat([alert_candidates, candidates], ignore_index=True)
                    alert_candidates = alert_candidates.nlargest(200, "Threat_Probability").reset_index(drop=True)
                predicted.to_csv(result_file, index=False, header=first_write)
                first_write = False
                processed += len(predicted)
                if progress is not None:
                    progress(min(0.99, processed / (processed + max(1, chunk_size))),
                             text=f"Preprocessed and classified {processed:,} rows…")
        preview = pd.concat(preview_parts, ignore_index=True) if preview_parts else pd.DataFrame()
        return {
            "output_path": output_path, "preview": preview, "alerts": alert_candidates,
            "total_rows": total_rows, "malicious_rows": malicious_rows, "high_rows": high_rows,
            "routed_rows": routed_rows, "eligible_rows": eligible_rows,
            "quantum_skipped_rows": max(0, eligible_rows - routed_rows),
            "source_name": source_name,
            "preprocessing": {
                "missing_features": sorted(missing_features_seen),
                "invalid_numeric_cells_replaced_with_zero": invalid_numeric_cells_total,
                "derived_features": derived_features_total,
            },
        }
    except Exception:
        try:
            os.unlink(output_path)
        except OSError:
            pass
        raise


def analyze_synthetic(bundle: dict[str, Any]) -> dict[str, Any]:
    """Create and analyze the requested 500-row demonstration dataset."""
    data = generate_synthetic_data(500)
    return analyze_csv_in_chunks(io.StringIO(data.to_csv(index=False)), bundle, chunk_size=100)


def clear_previous_analysis() -> None:
    """Release the prior disk-backed result when starting a replacement analysis."""
    previous = st.session_state.pop("threatlens_analysis", None)
    st.session_state.pop("threatlens_reports", None)
    if isinstance(previous, dict):
        path = previous.get("output_path")
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def build_alert_report(row: pd.Series, bundle: dict[str, Any],
                       index: faiss.IndexFlatL2, ids: list[str],
                       embedder: SentenceTransformer) -> dict[str, Any]:
    """Compute SHAP, optional quantum score, and retrieved ATT&CK playbook."""
    evidence = explain_flow(row, bundle)
    retrieval = retrieve_mitre(row, evidence, index, ids, embedder)
    probability = float(row["Threat_Probability"])
    is_routed = bool(row.get("Quantum_Routed", False))
    raw_quantum_score = row.get("Quantum_Risk_Score", np.nan)
    quantum_score = float(raw_quantum_score) if pd.notna(raw_quantum_score) else None
    route_status = str(row.get("Quantum_Route_Status", "Not routed"))
    return {
        "flow_id": str(row.get("Flow_ID", "Unknown")),
        "source_ip": str(row.get("Source_IP", "Unknown")),
        "dest_ip": str(row.get("Dest_IP", "Unknown")),
        "port": int(float(row["Destination_Port"])),
        "probability": probability, "priority": str(row["Priority"]),
        "evidence": evidence, "retrieval": retrieval,
        "quantum_routed": is_routed, "quantum_score": quantum_score, "quantum_route_status": route_status,
        "warning": str(row.get("Preprocessing_Warning", "None")),
    }


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="wide")
    st.title("🛡️ ThreatLens")
    st.caption("Hybrid Quantum-Classical AI Co-Pilot for Explainable Threat Triage")
    st.markdown(
        "Local network-flow preprocessing, XGBoost detection, SHAP explanations, "
        "a PennyLane quantum simulator for ambiguous scores, and local ATT&CK retrieval."
    )

    with st.sidebar:
        st.header("Data ingestion")
        uploaded = st.file_uploader("Upload network data in CSV format", type=["csv"])
        chunk_size = st.select_slider(
            "Rows per processing batch", options=[5_000, 10_000, 25_000, 50_000, 100_000, 250_000],
            value=DEFAULT_CHUNK_SIZE, format_func=lambda value: f"{value:,}",
            help="Larger batches can be faster but use more RAM.",
        )
        run_upload = st.button("Preprocess & Analyze CSV", type="primary", use_container_width=True)
        run_synth = st.button("Load Synthetic Attack Benchmark", use_container_width=True)
        st.divider()
        st.subheader("System status")
        with st.spinner("Loading local ML, embedding, and quantum resources…"):
            try:
                bundle = train_model()
                index, technique_ids = build_faiss_index()
                embedder = load_embedder()
                ready = True
            except Exception as exc:
                ready = False
                st.error(f"Local resource initialization failed: {exc}")
        st.write(f"{'🟢' if ready else '🔴'} Local ML Engine: {'Online' if ready else 'Unavailable'}")
        faiss_ready = ready and index.d == 384 and index.ntotal == len(MITRE)
        st.write(f"{'🟢' if faiss_ready else '🔴'} FAISS: {'Loaded' if faiss_ready else 'Unavailable'}")
        st.write(f"{'🟢' if ready else '🔴'} QPU Simulator: {'Ready (default.qubit)' if ready else 'Unavailable'}")
        st.write("🟢 Traffic analysis: local CPU processing")
        if ready:
            with st.expander("Synthetic hold-out model metrics"):
                st.write({key: round(value, 4) if isinstance(value, float) else value
                          for key, value in bundle["metrics"].items()})
                st.caption("Metrics are from synthetic data only; they do not establish real-world detection accuracy.")

    if not ready:
        st.stop()

    if run_synth:
        clear_previous_analysis()
        with st.spinner("Creating and analyzing 500 synthetic network flows…"):
            state = analyze_synthetic(bundle)
            st.session_state["threatlens_analysis"] = state
            st.session_state["threatlens_reports"] = None
    elif uploaded is not None and run_upload:
        clear_previous_analysis()
        progress = st.progress(0, text="Starting CSV preprocessing…")
        try:
            with st.spinner("Streaming, preprocessing, and classifying CSV chunks locally…"):
                state = analyze_csv_in_chunks(uploaded, bundle, chunk_size, progress.progress)
            st.session_state["threatlens_analysis"] = state
            st.session_state["threatlens_reports"] = None
        except Exception as exc:
            st.error(f"CSV preprocessing/analysis failed: {exc}")
        finally:
            progress.empty()
    elif uploaded is not None:
        st.info("Choose **Preprocess & Analyze CSV** to process the uploaded file in batches.")

    state = st.session_state.get("threatlens_analysis")
    if not state:
        st.info("Upload a network-flow CSV or load the 500-row synthetic benchmark to begin.")
        st.markdown(
            "**CSV compatibility:** ThreatLens normalizes recognized CICIDS/NetFlow/Zeek-style field names and derives some rates when source measurements allow it. "
            "When required measurements cannot be found, it reports that and uses zero for the absent model input; inspect preprocessing warnings because such predictions may be unreliable."
        )
        return

    m = st.columns(4)
    m[0].metric("Total Flows Analyzed", f"{state['total_rows']:,}")
    m[1].metric("High-Confidence Threats", f"{state['high_rows']:,}")
    m[2].metric("Quantum-Routed Edge Cases", f"{state['routed_rows']:,}")
    reports = st.session_state.get("threatlens_reports")
    mean_triage = reports.get("mean_ms") if isinstance(reports, dict) else None
    m[3].metric("Mean Triage Time", f"{mean_triage:.2f} ms / alert" if mean_triage is not None else "Not computed yet")

    st.subheader("Preprocessed network flow preview")
    preview = state["preview"]
    if not preview.empty:
        def color_predicted_rows(row: pd.Series) -> list[str]:
            color = "#4a2028" if int(row.get("Prediction", 0)) == 1 else "#18352f"
            return [f"background-color: {color}; color: #e6edf3"] * len(row)
        st.dataframe(preview.head(PREVIEW_ROWS).style.apply(color_predicted_rows, axis=1),
                     use_container_width=True, height=420)
    st.caption("Large-file mode writes the complete preprocessed predictions to a local temporary CSV; only the preview and top 200 candidate alerts are held in memory.")

    with st.expander("Preprocessing diagnostics", expanded=True):
        prep_summary = state.get("preprocessing", {})
        missing_features = prep_summary.get("missing_features", [])
        invalid_cells = int(prep_summary.get("invalid_numeric_cells_replaced_with_zero", 0))
        if missing_features:
            st.warning("Missing feature columns filled with zero: " + ", ".join(missing_features))
        else:
            st.success("All seven model feature columns were mapped or derived.")
        st.metric("Invalid numeric cells replaced with zero (whole file)", f"{invalid_cells:,}")
        st.write(
            f"Quantum score interval: {EDGE_LOW:.2f}–{EDGE_HIGH:.2f}. "
            f"Eligible rows: {state.get('eligible_rows', 0):,}; circuit evaluations: {state.get('routed_rows', 0):,}; "
            f"eligible rows skipped due to the {MAX_QUANTUM_EVALUATIONS:,}-evaluation cap: {state.get('quantum_skipped_rows', 0):,}."
        )
        derived_features = prep_summary.get("derived_features", {})
        if derived_features:
            st.write("Derived feature values counted across chunks:", derived_features)
        preview_warnings = preview.get("Preprocessing_Warning", pd.Series(dtype=str))
        other_warnings = sorted({str(item) for item in preview_warnings if str(item) not in {"None", ""}})
        for warning in other_warnings:
            st.caption(warning)
        st.write(f"Processed file: **{Path(state['output_path']).name}** (local temporary file)")
        st.caption("The model uses seven fixed numerical features. Different schemas are not semantically interchangeable; zero-filled features should be reviewed before trusting scores.")

    output_path = state["output_path"]
    if os.path.isfile(output_path):
        st.download_button(
            "Download full preprocessed predictions (CSV)",
            data=lambda path=output_path: Path(path).read_bytes(),
            file_name="threatlens_preprocessed_predictions.csv",
            mime="text/csv",
            help="The full output is read from disk only when the download is requested.",
        )
    else:
        st.error("Preprocessed output is no longer available. Please run the analysis again.")

    candidates = state["alerts"]
    if candidates.empty:
        st.success("The classifier did not flag malicious flows in this dataset.")
        return
    st.subheader("Triage alerts")
    if len(candidates) > MAX_ALERT_CARDS:
        st.caption(f"Showing the top {MAX_ALERT_CARDS} malicious candidates by classifier score.")
    candidates = candidates.head(MAX_ALERT_CARDS)
    if not isinstance(reports, dict) or reports.get("key") != state["output_path"]:
        all_reports = []
        started = time.perf_counter()
        with st.spinner("Calculating SHAP evidence, optional quantum scores, and ATT&CK retrieval for top alerts…"):
            for _, row in candidates.iterrows():
                all_reports.append(build_alert_report(row, bundle, index, technique_ids, embedder))
        elapsed = (time.perf_counter() - started) * 1000
        report_state = {"key": state["output_path"], "items": all_reports,
                        "mean_ms": elapsed / max(1, len(all_reports))}
        st.session_state["threatlens_reports"] = report_state
        reports = report_state
        m[3].metric("Mean Triage Time", f"{reports['mean_ms']:.2f} ms / alert")

    for report in reports["items"]:
        title = f"{report['priority']} · {report['probability']:.1%} threat score · {report['flow_id']}"
        with st.expander(title):
            left, right = st.columns([1, 1])
            with left:
                st.markdown("**SHAP mathematical evidence — top 3 features**")
                chart = pd.DataFrame(
                    {"Absolute SHAP": [item["absolute_shap"] for item in report["evidence"]]},
                    index=[item["feature"] for item in report["evidence"]],
                )
                st.bar_chart(chart, horizontal=True)
                st.dataframe(pd.DataFrame(report["evidence"]), hide_index=True, use_container_width=True)
                st.caption("Absolute SHAP values rank this row's feature effects on the model output; they are not attack probabilities or causal proof.")
            with right:
                technique = report["retrieval"]["technique"]
                st.markdown(f"**MITRE ATT&CK: {technique['id']} — {technique['name']}**")
                st.write(f"**Tactic:** {technique['tactic']}")
                st.write(f"**FAISS retrieval distance:** {report['retrieval']['distance']:.4f} (lower means closer in this embedding space)")
                if report["quantum_routed"]:
                    st.metric("Quantum circuit score (exploratory)", f"{report['quantum_score']:.3f}")
                    st.caption("Only run for XGBoost scores between 0.40 and 0.65 inclusive. This fixed-parameter simulator score is not a calibrated probability and does not override XGBoost.")
                else:
                    st.write(f"**Quantum edge-case resolver:** {report['quantum_route_status']}.")
                st.markdown("**SOC mitigation playbook**")
                for step in technique["mitigations"]:
                    st.markdown(f"- {step}")
                if report["warning"] != "None":
                    st.warning(report["warning"])
                st.caption(f"Context is a retrieval match from a six-entry local reference set, not confirmation that the technique occurred.")


if __name__ == "__main__":
    main()
