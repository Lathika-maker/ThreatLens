"""ThreatLens: evidence-grounded, local network threat triage."""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any

# Keep the small CPU encoder stable on Apple Silicon and avoid oversized BLAS
# thread pools in Streamlit's runner process.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / ".cache" / "huggingface"))

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Constants -----------------------------------------------------------------
FEATURES = [
    "Destination_Port", "Flow_Duration", "Total_Fwd_Packets", "Total_Bwd_Packets",
    "Flow_Bytes_s", "Flow_Packets_s", "Packet_Length_Mean", "SYN_Flag_Count",
]
CLASSES = ["BENIGN", "PortScan", "DDoS", "BruteForce", "PrivilegeAnomaly"]
LOG_FEATURES = {
    "Flow_Duration", "Flow_Bytes_s", "Flow_Packets_s", "Total_Fwd_Packets",
    "Total_Bwd_Packets", "SYN_Flag_Count",
}
HIGH_PRIORITY_CLASSES = {"DDoS", "BruteForce", "PrivilegeAnomaly"}
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 2. Threat intelligence knowledge base ---------------------------------------
THREAT_INTEL: dict[str, dict[str, Any]] = {
    "T1046": {
        "id": "T1046", "name": "Network Service Discovery", "tactic": "Discovery",
        "class": "PortScan",
        "description": "An actor probes hosts and services to identify reachable ports. Repeated short flows with few replies can reveal systematic service discovery.",
        "impact": "Scanning exposes reachable services and helps an actor plan follow-on activity.",
        "signature": "many destination ports, tiny flows, one to three forward packets, few backward response packets, low byte rate",
        "actions": [
            "Contain: block or rate-limit the scanning source at the network edge.",
            "Investigate: review probed hosts, exposed services, and follow-on connections.",
            "Remediate: close unnecessary ports and tune scan detection thresholds.",
        ],
    },
    "T1110": {
        "id": "T1110", "name": "Brute Force", "tactic": "Credential Access",
        "class": "BruteForce",
        "description": "An actor repeatedly attempts to guess or reuse credentials against a service. Repetitive traffic to remote access ports can indicate password guessing.",
        "impact": "Successful guessing can expose accounts and provide an initial foothold.",
        "signature": "repeated login attempts, remote access service, destination port 22, 21, or 3389, recurring small flows",
        "actions": [
            "Contain: rate-limit or temporarily block the source on the targeted service.",
            "Investigate: review authentication logs and targeted accounts for successful logins.",
            "Remediate: enforce MFA, strong credentials, and account lockout controls.",
        ],
    },
    "T1071": {
        "id": "T1071", "name": "Application Layer Protocol", "tactic": "Command and Control",
        "class": "DDoS",
        "description": "An actor can use common application protocols to communicate with external systems. Repeated network behavior may blend into ordinary service traffic.",
        "impact": "Covert communication can enable remote control or data movement.",
        "signature": "repeated application traffic, outbound communication, regular network flows, recurring destination service",
        "actions": [
            "Contain: block confirmed malicious destinations at the network boundary.",
            "Investigate: inspect DNS, proxy, and endpoint logs for matching communication.",
            "Remediate: restrict outbound traffic to approved services and destinations.",
        ],
    },
    "T1499": {
        "id": "T1499", "name": "Endpoint Denial of Service", "tactic": "Impact",
        "class": "DDoS",
        "description": "An actor exhausts an endpoint or service through excessive requests or resource use. A flood of short, high-rate flows with few replies is a common signal.",
        "impact": "The target service may become slow or unavailable to legitimate users.",
        "signature": "high packet rate, many SYN flags, very short flows, no backward response packets, flood of small packets to a single port, denial of service",
        "actions": [
            "Contain: apply upstream rate limits or filtering to the targeted service.",
            "Investigate: check service health and preserve traffic and system evidence.",
            "Remediate: tune capacity, firewall rules, and DDoS protections for the endpoint.",
        ],
    },
    "T1190": {
        "id": "T1190", "name": "Exploit Public-Facing Application", "tactic": "Initial Access",
        "class": "PortScan",
        "description": "An actor may exploit a weakness in an application exposed to the internet. Suspicious probing of web services can precede exploitation attempts.",
        "impact": "Successful exploitation may provide access to a host or its data.",
        "signature": "web service probing, repeated requests, public-facing application, destination ports 80 or 443, unusual response pattern",
        "actions": [
            "Contain: temporarily restrict suspicious sources from the exposed service.",
            "Investigate: review web, application, and host logs for exploit indicators.",
            "Remediate: patch the application and remove unnecessary public exposure.",
        ],
    },
    "T1078": {
        "id": "T1078", "name": "Valid Accounts",
        "tactic": "Defense Evasion, Persistence, Privilege Escalation, Initial Access",
        "class": "PrivilegeAnomaly",
        "description": "An actor may use legitimate credentials to access systems and evade simple controls. Unusual traffic to administrative services can add supporting context.",
        "impact": "Misused accounts can enable persistent access and privileged actions.",
        "signature": "unusual administrative access, destination ports 445, 3389, or 389, longer two-way flows, valid account activity",
        "actions": [
            "Contain: disable or restrict the account when misuse is suspected.",
            "Investigate: audit sign-ins, privileges, and affected hosts for unauthorized actions.",
            "Remediate: rotate credentials and require MFA on privileged accounts.",
        ],
    },
}
CLASS_TO_TECHNIQUE = {
    "PortScan": "T1046", "BruteForce": "T1110", "DDoS": "T1499",
    "PrivilegeAnomaly": "T1078",
}


def technique_text(technique: dict[str, Any]) -> str:
    """Return the stable text used to embed a knowledge-base entry."""
    return " ".join(technique[key] for key in ("name", "tactic", "description", "signature"))


# 3. Synthetic data and cached resources --------------------------------------
def _sample_range(rng: np.random.Generator, bounds: tuple[float, float], size: int,
                  log_uniform: bool = False) -> np.ndarray:
    if log_uniform:
        return np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]), size))
    return rng.uniform(bounds[0], bounds[1], size)


def generate_flows(n_per_class: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate repeatable CICIDS-style records with five percent profile noise."""
    rng = np.random.default_rng(seed)
    specs: dict[str, dict[str, Any]] = {
        "BENIGN": {"n": max(2500, n_per_class), "ports": [80, 443, 53, 8080], "duration": (1e5, 1e7), "fwd": (5, 60), "bwd": (5, 80), "bytes": (1e3, 1e6), "length": (200, 900), "syn": (1, 2)},
        "PortScan": {"n": n_per_class, "ports": None, "duration": (1, 5e3), "fwd": (1, 3), "bwd": (0, 1), "bytes": (1, 5e3), "length": (0, 60), "syn": (1, 1)},
        "DDoS": {"n": n_per_class, "ports": [80, 443], "duration": (10, 5e4), "fwd": (100, 5000), "bwd": (0, 2), "bytes": (1e7, 1e9), "length": (40, 120), "syn": (50, 500)},
        "BruteForce": {"n": n_per_class, "ports": [21, 22, 3389], "duration": (1e5, 1e6), "fwd": (10, 30), "bwd": (10, 30), "bytes": (1e3, 5e5), "length": (60, 200), "syn": (1, 3)},
        "PrivilegeAnomaly": {"n": n_per_class, "ports": [389, 445, 3389], "duration": (1e5, 5e6), "fwd": (20, 100), "bwd": (20, 100), "bytes": (1e3, 8e5), "length": (300, 1200), "syn": (1, 2)},
    }
    parts: list[pd.DataFrame] = []
    for label, spec in specs.items():
        count = int(spec["n"])
        ports = (rng.choice(spec["ports"], count) if spec["ports"] else
                 rng.integers(1, 65536, count))
        duration = _sample_range(rng, spec["duration"], count, log_uniform=True)
        fwd = rng.integers(spec["fwd"][0], spec["fwd"][1] + 1, count)
        bwd = rng.integers(spec["bwd"][0], spec["bwd"][1] + 1, count)
        byte_rate = _sample_range(rng, spec["bytes"], count, log_uniform=True)
        frame = pd.DataFrame({
            "Destination_Port": ports,
            "Flow_Duration": duration,
            "Total_Fwd_Packets": fwd,
            "Total_Bwd_Packets": bwd,
            "Flow_Bytes_s": byte_rate,
            "Flow_Packets_s": (fwd + bwd) / np.maximum(duration / 1e6, 1e-9),
            "Packet_Length_Mean": _sample_range(rng, spec["length"], count),
            "SYN_Flag_Count": rng.integers(spec["syn"][0], spec["syn"][1] + 1, count),
            "Label": label,
        })
        frame.insert(0, "Flow_ID", [f"F-{label}-{i:06d}" for i in range(count)])
        frame.insert(1, "Source_IP", [f"10.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}" for _ in range(count)])
        frame.insert(2, "Dest_IP", [f"192.168.{rng.integers(0, 256)}.{rng.integers(1, 255)}" for _ in range(count)])
        parts.append(frame)

    data = pd.concat(parts, ignore_index=True)
    # Five percent of rows borrow selected measurements from a different class;
    # labels stay attached to the source class so the demo includes realistic overlap.
    feature_cols = ["Destination_Port", "Flow_Duration", "Total_Fwd_Packets",
                    "Total_Bwd_Packets", "Flow_Bytes_s", "Packet_Length_Mean", "SYN_Flag_Count"]
    for label in CLASSES:
        indexes = data.index[data["Label"] == label].to_numpy()
        noise_n = int(round(len(indexes) * 0.05))
        if not noise_n:
            continue
        targets = rng.choice(indexes, noise_n, replace=False)
        others = data.index[data["Label"] != label].to_numpy()
        sources = rng.choice(others, noise_n, replace=True)
        data.loc[targets, feature_cols] = data.loc[sources, feature_cols].to_numpy()
    data["Flow_Packets_s"] = (data["Total_Fwd_Packets"] + data["Total_Bwd_Packets"]) / np.maximum(data["Flow_Duration"] / 1e6, 1e-9)
    return data.iloc[rng.permutation(len(data))].reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    """Load the local MiniLM encoder; HF_HUB_OFFLINE=1 is honored by HF Hub."""
    repository = Path(os.environ["HF_HOME"]) / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
    snapshots = repository / "snapshots"
    cached = snapshots.is_dir() and any((snapshot / "modules.json").is_file() for snapshot in snapshots.iterdir())
    if cached:
        return SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_resource(show_spinner=False)
def build_vector_index() -> tuple[faiss.IndexFlatL2, list[str]]:
    """Build one cached 384-dimensional FAISS index for the six techniques."""
    ordered_ids = list(THREAT_INTEL)
    vectors = load_embedder().encode(
        [technique_text(THREAT_INTEL[technique_id]) for technique_id in ordered_ids],
        convert_to_numpy=True,
    ).astype(np.float32)
    if vectors.shape != (len(ordered_ids), 384):
        raise ValueError(f"Expected 6 embeddings of dimension 384; got {vectors.shape}.")
    index = faiss.IndexFlatL2(384)
    index.add(vectors)
    return index, ordered_ids


def _model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Create model input from the fixed feature list only."""
    return (frame.loc[:, FEATURES].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan).fillna(0).astype(float))


def _baseline_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    values = _model_matrix(frame)
    values.loc[:, list(LOG_FEATURES)] = np.log1p(values[list(LOG_FEATURES)].clip(lower=0))
    return values


@st.cache_resource(show_spinner=False)
def train_model() -> dict[str, Any]:
    """Train a deterministic Random Forest and its benign reference baseline."""
    data = generate_flows()
    x = _model_matrix(data)
    x_train, x_test, y_train, y_test = train_test_split(
        x, data["Label"], test_size=0.25, stratify=data["Label"], random_state=42,
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1,
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    benign = _baseline_matrix(x_train.loc[y_train == "BENIGN"])
    baseline = {
        "mean": benign.mean(),
        "std": benign.std(ddof=0).replace(0, 1e-9),
        "raw_mean": _model_matrix(x_train.loc[y_train == "BENIGN"]).mean(),
    }
    return {
        "model": model,
        "baseline": baseline,
        "importances": pd.Series(model.feature_importances_, index=FEATURES),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "classification_report": classification_report(y_test, prediction, labels=CLASSES, zero_division=0),
    }


# 4. Ingestion ---------------------------------------------------------------
def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize CICIDS and common flow-field aliases to canonical names."""
    normalized = frame.copy()
    normalized.columns = [str(column).strip().replace(" ", "_") for column in normalized.columns]
    aliases = {
        "Flow_Bytes/s": "Flow_Bytes_s", "Flow_Bytes_sec": "Flow_Bytes_s",
        "Flow_Packets/s": "Flow_Packets_s", "Flow_Packets_sec": "Flow_Packets_s",
        "Total_Fwd_Packets": "Total_Fwd_Packets", "Total_Fwd_Packets_": "Total_Fwd_Packets",
        "Total_Bwd_Packets": "Total_Bwd_Packets", "Dst_Port": "Destination_Port",
        "Destination_Port": "Destination_Port", "DstPort": "Destination_Port",
    }
    normalized = normalized.rename(columns=aliases)
    return normalized.loc[:, ~normalized.columns.duplicated()]


def prepare_uploaded_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """Normalize and validate a tabular upload without raising for user errors."""
    normalized = normalize_columns(frame)
    missing = [feature for feature in FEATURES if feature not in normalized.columns]
    if missing:
        return None, missing
    for column in FEATURES:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized[FEATURES] = normalized[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    if "Flow_ID" not in normalized:
        normalized.insert(0, "Flow_ID", [f"UPLOAD-{i + 1:06d}" for i in range(len(normalized))])
    if "Source_IP" not in normalized:
        normalized["Source_IP"] = "Unknown"
    if "Dest_IP" not in normalized:
        normalized["Dest_IP"] = "Unknown"
    return normalized, []


def load_uploaded_csv(file: Any) -> pd.DataFrame | None:
    """Read, normalize, and validate an uploaded CSV; report issues in Streamlit."""
    try:
        content = file.getvalue() if hasattr(file, "getvalue") else file.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        if not content:
            st.error("The selected CSV is empty.")
            return None
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        return None
    if frame.empty:
        st.error("The selected CSV has no data rows.")
        return None
    cleaned, missing = prepare_uploaded_frame(frame)
    if missing:
        st.error("CSV is missing required feature columns: " + ", ".join(missing))
        return None
    return cleaned


# 5. Detection ---------------------------------------------------------------
def predict_flows(frame: pd.DataFrame, bundle: dict[str, Any] | None = None) -> pd.DataFrame:
    """Predict labels and confidence using only FEATURES as model inputs."""
    bundle = bundle or train_model()
    x = _model_matrix(frame)
    model = bundle["model"]
    probabilities = model.predict_proba(x)
    result = frame.copy()
    result["Predicted"] = model.predict(x)
    result["Confidence"] = probabilities.max(axis=1) * 100
    result["Is_Malicious"] = result["Predicted"].ne("BENIGN")
    return result


# 6. Explainability ----------------------------------------------------------
def _transformed_value(feature: str, value: float) -> float:
    return float(np.log1p(max(value, 0))) if feature in LOG_FEATURES else float(value)


def explain_flow(flow: pd.Series | dict[str, Any], bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Return stable top-three local evidence shares with benign baselines."""
    baseline = bundle["baseline"]
    scores: list[dict[str, Any]] = []
    for feature in FEATURES:
        value = float(pd.to_numeric(pd.Series([flow[feature]]), errors="coerce").fillna(0).iloc[0])
        transformed = _transformed_value(feature, value)
        mean = float(baseline["mean"][feature])
        std = max(float(baseline["std"][feature]), 1e-9)
        z_score = min(abs(transformed - mean) / (std + 1e-9), 10.0)
        score = float(bundle["importances"][feature]) * z_score
        raw_baseline = float(baseline["raw_mean"][feature])
        ratio = value / (raw_baseline + 1e-9) if raw_baseline >= 0 else value / (abs(raw_baseline) + 1e-9)
        scores.append({
            "feature": feature, "value": value, "baseline": raw_baseline,
            "ratio": ratio, "direction": "above" if transformed >= mean else "below",
            "score": score,
        })
    top = sorted(scores, key=lambda item: (-item["score"], FEATURES.index(item["feature"])))[:3]
    total = sum(item["score"] for item in top)
    for item in top:
        item["share"] = (100.0 * item["score"] / total) if total else (100.0 / len(top))
    if top:
        top[-1]["share"] += 100.0 - sum(item["share"] for item in top)
    return [{key: value for key, value in item.items() if key != "score"} for item in top]


# 7. Retrieval ---------------------------------------------------------------
def _behavior_phrase(feature: str, direction: str, port: int) -> str:
    if feature == "Destination_Port" and port in {21, 22, 3389}:
        return "repeated login attempts on remote access service"
    phrases = {
        ("SYN_Flag_Count", "above"): "many SYN flags",
        ("Flow_Packets_s", "above"): "high packet rate flood",
        ("Flow_Bytes_s", "above"): "high byte rate traffic",
        ("Total_Fwd_Packets", "above"): "many forward packets",
        ("Total_Bwd_Packets", "below"): "no response packets",
        ("Flow_Duration", "below"): "very short flows",
        ("Destination_Port", "above"): "scanning many destination ports",
        ("Destination_Port", "below"): "traffic to a small set of ports",
    }
    return phrases.get((feature, direction), f"{feature.replace('_', ' ').lower()} {direction} baseline")


def build_query(pred_class: str, evidence: list[dict[str, Any]], port: int | None = None) -> str:
    """Build a classifier-grounded query in network behavior language."""
    prefixes = {
        "PortScan": "port scan reconnaissance: scanning many ports",
        "DDoS": "denial of service SYN flood: high packet rate flood",
        "BruteForce": "credential brute force: repeated login attempts on remote access service",
        "PrivilegeAnomaly": "unusual valid account administrative access",
    }
    port = int(port or 0)
    phrases = [_behavior_phrase(item["feature"], item["direction"], port) for item in evidence]
    return f"{prefixes.get(pred_class, pred_class.lower())}: " + ", ".join(phrases)


def retrieve_technique(pred_class: str, evidence: list[dict[str, Any]],
                       index: faiss.IndexFlatL2 | None = None,
                       ordered_ids: list[str] | None = None,
                       embedder: SentenceTransformer | None = None,
                       port: int | None = None) -> dict[str, Any]:
    """Retrieve nearest technique and expose classifier mapping as primary when known.

    The classifier owns the class decision. If nearest-neighbor retrieval disagrees,
    its ID and distance remain visible as supporting evidence, while the stable class
    mapping is used as the primary report technique to prevent semantic drift.
    """
    if index is None or ordered_ids is None:
        index, ordered_ids = build_vector_index()
    if embedder is None:
        embedder = load_embedder()
    query = build_query(pred_class, evidence, port)
    vector = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, positions = index.search(vector, 1)
    retrieved_id = ordered_ids[int(positions[0][0])]
    distance = float(distances[0][0])
    primary_id = CLASS_TO_TECHNIQUE.get(pred_class, retrieved_id)
    return {
        "technique_id": primary_id, "retrieved_id": retrieved_id,
        "distance": distance, "query": query,
        "technique": THREAT_INTEL[primary_id],
    }


# 8. Deterministic report generator ------------------------------------------
def generate_report(flow: pd.Series | dict[str, Any], pred_class: str,
                    confidence: float, evidence: list[dict[str, Any]],
                    retrieval: dict[str, Any]) -> dict[str, Any]:
    """Build a complete deterministic alert report from measured values and KB."""
    technique = retrieval["technique"]
    flow_id = str(flow.get("Flow_ID", "Unknown"))
    source = str(flow.get("Source_IP", "Unknown"))
    destination = str(flow.get("Dest_IP", "Unknown"))
    port = int(float(flow.get("Destination_Port", 0)))
    priority = "High" if confidence >= 90 and pred_class in HIGH_PRIORITY_CLASSES else "Medium"
    e1, e2, e3 = evidence
    narrative = (
        f"Flow {flow_id} from {source} to {destination}:{port} was classified as {pred_class} "
        f"with {confidence:.1f}% confidence. The decision was driven by "
        f"{e1['feature']} ({e1['value']:.4g} versus a benign baseline of {e1['baseline']:.4g}), "
        f"{e2['feature']} ({e2['value']:.4g} versus {e2['baseline']:.4g}) and "
        f"{e3['feature']} ({e3['value']:.4g} versus {e3['baseline']:.4g}). "
        f"This behavior matches MITRE ATT&CK {technique['id']} ({technique['name']}) "
        f"under the {technique['tactic']} tactic. {technique['impact']}"
    )
    return {
        "flow_id": flow_id, "source_ip": source, "dest_ip": destination, "port": port,
        "classification": pred_class, "confidence": float(confidence), "priority": priority,
        "evidence": evidence, "mitre_id": technique["id"], "mitre_name": technique["name"],
        "tactic": technique["tactic"], "retrieved_id": retrieval["retrieved_id"],
        "retrieval_distance": retrieval["distance"], "narrative": narrative,
        "checklist": [
            {"step": f"Step {number} {label}", "action": action}
            for number, (label, action) in enumerate(zip(
                ("Contain", "Investigate", "Remediate"), technique["actions"]), 1)
        ],
    }


def triage_flow(flow: pd.Series | dict[str, Any], bundle: dict[str, Any],
                index: faiss.IndexFlatL2 | None = None,
                ordered_ids: list[str] | None = None,
                embedder: SentenceTransformer | None = None) -> dict[str, Any]:
    """Explain, retrieve, and report for one already-classified flow."""
    evidence = explain_flow(flow, bundle)
    retrieval = retrieve_technique(
        str(flow["Predicted"]), evidence, index, ordered_ids, embedder,
        port=int(float(flow["Destination_Port"])),
    )
    return generate_report(flow, str(flow["Predicted"]), float(flow["Confidence"]), evidence, retrieval)


def reports_to_frame(reports: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten deterministic triage reports for CSV download."""
    rows = []
    for report in reports:
        row = {key: value for key, value in report.items() if key not in {"evidence", "checklist"}}
        row["evidence"] = json.dumps(report["evidence"], ensure_ascii=False)
        row["checklist"] = json.dumps(report["checklist"], ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


# 9. Streamlit dashboard ------------------------------------------------------
def system_status(bundle: dict[str, Any], index: faiss.IndexFlatL2,
                  ordered_ids: list[str], embedder: SentenceTransformer) -> dict[str, bool]:
    """Derive sidebar health indicators from the loaded runtime objects."""
    model = bundle.get("model")
    model_ok = bool(
        model is not None
        and hasattr(model, "classes_")
        and getattr(model, "n_features_in_", 0) == len(FEATURES)
        and np.isfinite(bundle.get("accuracy", np.nan))
    )
    index_ok = bool(
        index is not None
        and index.is_trained
        and index.d == 384
        and index.ntotal == len(ordered_ids) == len(THREAT_INTEL)
    )
    local_ok = bool(
        model_ok
        and index_ok
        and str(getattr(embedder, "device", "")) == "cpu"
        and isinstance(embedder, SentenceTransformer)
    )
    return {"model": model_ok, "index": index_ok, "local": local_ok}


def _style_flows(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    def color_row(row: pd.Series) -> list[str]:
        color = "#4a2028" if bool(row["Is_Malicious"]) else "#18352f"
        return [f"background-color: {color}; color: #e6edf3"] * len(row)
    return frame.style.apply(color_row, axis=1)


def main() -> None:
    """Render the local Streamlit SOC dashboard."""
    st.set_page_config(layout="wide", page_title="ThreatLens | SOC Co-Pilot", page_icon="🛡️")
    st.title("🛡️ ThreatLens | SOC Co-Pilot")
    st.caption("Evidence-grounded network triage · local Random Forest · MITRE ATT&CK retrieval")

    st.sidebar.header("ThreatLens | SOC Co-Pilot")
    uploaded = st.sidebar.file_uploader("Upload network flow CSV", type=["csv"])
    if st.sidebar.button("Load Synthetic Network Logs", width="stretch"):
        st.session_state["df"] = generate_flows()
    if uploaded is not None:
        cleaned = load_uploaded_csv(uploaded)
        if cleaned is not None:
            st.session_state["df"] = cleaned
        else:
            st.session_state.pop("df", None)

    try:
        with st.spinner("Loading local model and MITRE retrieval index…"):
            bundle = train_model()
            index, ordered_ids = build_vector_index()
            embedder = load_embedder()
        status = system_status(bundle, index, ordered_ids, embedder)
        (st.sidebar.success if status["model"] else st.sidebar.error)(
            f"Model {'loaded' if status['model'] else 'unavailable'} · hold-out accuracy {bundle['accuracy']:.1%}"
        )
        (st.sidebar.success if status["index"] else st.sidebar.error)(
            f"Vector index {'active' if status['index'] else 'unavailable'} · {index.ntotal} techniques"
        )
        (st.sidebar.success if status["local"] else st.sidebar.error)(
            "Zero data leakage · local CPU inference and retrieval" if status["local"]
            else "Local processing unavailable"
        )
    except Exception as exc:
        st.sidebar.error(f"Model or vector index unavailable: {exc}")
        st.error("ThreatLens could not load its local resources. Check installed dependencies and cached model files.")
        return

    with st.sidebar.expander("Priority rule"):
        st.write("High when model confidence is at least 90% and the class is DDoS, BruteForce, or PrivilegeAnomaly. All other alerts are Medium.")
    with st.sidebar.expander("Model Quality"):
        st.caption("Stratified synthetic hold-out split · 25% test size · random_state=42")
        st.code(bundle["classification_report"], language="text")

    if "df" not in st.session_state:
        st.info("Upload a network-flow CSV or load the synthetic demo dataset to begin.")
        return
    data = st.session_state["df"].copy()
    missing = [feature for feature in FEATURES if feature not in data.columns]
    if missing:
        st.error("CSV is missing required feature columns: " + ", ".join(missing))
        return

    try:
        predicted = predict_flows(data, bundle)
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"Could not classify these flows: {exc}")
        return

    malicious = predicted.loc[predicted["Is_Malicious"]].sort_values("Confidence", ascending=False)
    high_count = int((malicious["Confidence"].ge(90) & malicious["Predicted"].isin(HIGH_PRIORITY_CLASSES)).sum())
    display_alerts = malicious.head(50)
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for _, flow in display_alerts.iterrows():
        reports.append(triage_flow(flow, bundle, index, ordered_ids, embedder))
    elapsed_ms = (time.perf_counter() - started) * 1000
    mean_triage_ms = elapsed_ms / len(reports) if reports else None

    metrics = st.columns(4)
    metrics[0].metric("Total Network Flows Analyzed", f"{len(predicted):,}")
    metrics[1].metric("Malicious Anomalies Detected", f"{len(malicious):,}")
    metrics[2].metric("High-Priority Alerts", f"{high_count:,}")
    triage_label = f"{mean_triage_ms:.2f} ms / alert" if mean_triage_ms is not None else "N/A"
    metrics[3].metric("Mean Triage Time", triage_label)

    st.subheader("Network flow overview")
    display = predicted.head(500)
    st.dataframe(_style_flows(display), width="stretch", height=420)
    if len(predicted) > 500:
        st.caption("Showing the first 500 rows for responsiveness.")

    if reports:
        st.download_button(
            "Download triage reports (CSV)", reports_to_frame(reports).to_csv(index=False).encode("utf-8"),
            "threatlens_triage_reports.csv", "text/csv",
        )
    st.subheader("Alert triage")
    if malicious.empty:
        st.success("No malicious flows detected in this dataset.")
        return
    if len(malicious) > len(reports):
        st.caption("Showing the 50 highest-confidence flagged flows.")
    for report in reports:
        title = (f"{report['classification']} · {report['priority']} priority · "
                 f"{report['confidence']:.1f}% confidence · {report['flow_id']}")
        with st.expander(title):
            left, right = st.columns([1, 1])
            with left:
                st.caption("Share of top-3 trigger evidence")
                chart = pd.DataFrame({"Share (%)": [item["share"] for item in report["evidence"]]},
                                     index=[item["feature"] for item in report["evidence"]])
                st.bar_chart(chart, horizontal=True, color="#ff4b4b")
                evidence_table = pd.DataFrame(report["evidence"]).rename(columns={
                    "feature": "Feature", "share": "Share (%)", "value": "Value",
                    "baseline": "Benign baseline", "ratio": "Ratio", "direction": "Direction",
                })
                st.dataframe(evidence_table, hide_index=True, width="stretch")
            with right:
                st.markdown(f"### {report['mitre_id']} · {report['mitre_name']}")
                st.write(f"**Tactic:** {report['tactic']}  ·  **Retrieval distance:** {report['retrieval_distance']:.3f}")
                if report["retrieved_id"] != report["mitre_id"]:
                    st.caption(f"Nearest vector entry: {report['retrieved_id']}; class mapping retained as primary.")
                st.write(report["narrative"])
                st.markdown("**Response checklist**")
                for action in report["checklist"]:
                    st.markdown(f"- **{action['step']}:** {action['action']}")


if __name__ == "__main__":
    main()
