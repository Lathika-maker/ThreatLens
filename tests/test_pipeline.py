"""Acceptance checks for ThreatLens pure pipeline functions."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import app


@pytest.fixture(scope="module")
def bundle():
    return app.train_model()


def test_knowledge_base_shape():
    assert set(app.THREAT_INTEL) == {"T1046", "T1110", "T1071", "T1499", "T1190", "T1078"}
    for technique in app.THREAT_INTEL.values():
        assert set(technique) >= {"id", "name", "tactic", "description", "impact", "signature", "actions"}
        assert len(technique["actions"]) == 3
        assert [action.split(":", 1)[0] for action in technique["actions"]] == ["Contain", "Investigate", "Remediate"]


def test_vector_index_has_six_384d_vectors():
    index, ids = app.build_vector_index()
    assert index.ntotal == 6
    assert index.d == 384
    assert ids == list(app.THREAT_INTEL)
    descriptions = [app.technique_text(app.THREAT_INTEL[technique_id]) for technique_id in ids]
    vectors = app.load_embedder().encode(descriptions, convert_to_numpy=True).astype(np.float32)
    _, positions = index.search(vectors, 1)
    assert [ids[int(position)] for position in positions[:, 0]] == ids
    app.build_vector_index.clear()


def test_synthetic_data_is_complete_and_seeded():
    first = app.generate_flows(n_per_class=80, seed=17)
    second = app.generate_flows(n_per_class=80, seed=17)
    assert set(app.FEATURES).issubset(first.columns)
    assert {"Flow_ID", "Source_IP", "Dest_IP", "Label"}.issubset(first.columns)
    assert set(first.Label) == set(app.CLASSES)
    pd.testing.assert_frame_equal(first, second)
    expected = (first.Total_Fwd_Packets + first.Total_Bwd_Packets) / np.maximum(first.Flow_Duration / 1e6, 1e-9)
    np.testing.assert_allclose(first.Flow_Packets_s, expected)


def test_model_holdout_accuracy_and_model_inputs(bundle):
    assert bundle["accuracy"] > 0.90
    flows = app.generate_flows(n_per_class=8)
    flows["Label"] = "corrupt-label-must-not-be-used"
    flows["Flow_ID"] = "private-id-must-not-be-used"
    output = app.predict_flows(flows, bundle)
    assert {"Predicted", "Confidence", "Is_Malicious"}.issubset(output.columns)
    assert output.Confidence.between(0, 100).all()
    assert (output.Is_Malicious == output.Predicted.ne("BENIGN")).all()


def test_sidebar_health_is_derived_from_runtime_resources(bundle):
    index, ordered_ids = app.build_vector_index()
    status = app.system_status(bundle, index, ordered_ids, app.load_embedder())
    assert status == {"model": True, "index": True, "local": True}
    bad_index = app.faiss.IndexFlatL2(8)
    assert app.system_status(bundle, bad_index, ordered_ids, app.load_embedder())["index"] is False


@pytest.mark.parametrize(
    ("label", "expected"),
    [("PortScan", "T1046"), ("DDoS", "T1499"), ("BruteForce", "T1110")],
)
def test_known_classes_map_to_expected_technique(label, expected):
    index, ordered_ids = app.build_vector_index()
    evidence = [
        {"feature": "SYN_Flag_Count", "direction": "above"},
        {"feature": "Total_Bwd_Packets", "direction": "below"},
        {"feature": "Flow_Duration", "direction": "below"},
    ]
    port = 22 if label == "BruteForce" else 443
    result = app.retrieve_technique(label, evidence, index, ordered_ids, app.load_embedder(), port)
    assert result["technique_id"] == expected


def test_evidence_is_deterministic_and_sums_to_100(bundle):
    flows = app.predict_flows(app.generate_flows(n_per_class=30), bundle)
    flagged = flows.loc[flows.Is_Malicious]
    assert not flagged.empty
    for _, flow in flagged.head(20).iterrows():
        first = app.explain_flow(flow, bundle)
        second = app.explain_flow(flow, bundle)
        assert first == second
        assert len(first) == 3
        assert sum(item["share"] for item in first) == pytest.approx(100, abs=0.01)


def test_upload_loader_reports_missing_columns(monkeypatch):
    errors: list[str] = []
    monkeypatch.setattr(app.st, "error", errors.append)
    loaded = app.load_uploaded_csv(io.BytesIO(b"Label,Some_Value\nBENIGN,1\n"))
    assert loaded is None
    assert errors
    assert all(feature in errors[-1] for feature in app.FEATURES)


def test_upload_cleaning_and_aliases():
    row = {feature: 1 for feature in app.FEATURES}
    row["Flow_Bytes/s"] = "inf"
    row["Flow_Packets/s"] = np.nan
    row.pop("Flow_Bytes_s")
    row.pop("Flow_Packets_s")
    row["Total Fwd Packets"] = "bad"
    row.pop("Total_Fwd_Packets")
    row["Dst Port"] = 443
    row.pop("Destination_Port")
    cleaned, missing = app.prepare_uploaded_frame(pd.DataFrame([row]))
    assert missing == []
    assert cleaned is not None
    assert cleaned.loc[0, "Flow_Bytes_s"] == 0
    assert cleaned.loc[0, "Flow_Packets_s"] == 0
    assert cleaned.loc[0, "Total_Fwd_Packets"] == 0
    assert cleaned.loc[0, "Destination_Port"] == 443
    assert {"Flow_ID", "Source_IP", "Dest_IP"}.issubset(cleaned.columns)


def test_report_is_complete_and_deterministic(bundle):
    flow = app.predict_flows(app.generate_flows(n_per_class=5), bundle).iloc[0]
    if not flow.Is_Malicious:
        flow = app.predict_flows(app.generate_flows(n_per_class=5), bundle).query("Predicted != 'BENIGN'").iloc[0]
    evidence = app.explain_flow(flow, bundle)
    tech_id = app.CLASS_TO_TECHNIQUE[str(flow.Predicted)]
    retrieval = {"technique_id": tech_id, "retrieved_id": tech_id, "distance": 0.0,
                 "technique": app.THREAT_INTEL[tech_id]}
    one = app.generate_report(flow, str(flow.Predicted), float(flow.Confidence), evidence, retrieval)
    two = app.generate_report(flow, str(flow.Predicted), float(flow.Confidence), evidence, retrieval)
    assert one == two
    assert len(one["checklist"]) == 3
    assert [step["step"] for step in one["checklist"]] == ["Step 1 Contain", "Step 2 Investigate", "Step 3 Remediate"]
    assert one["priority"] in {"High", "Medium"}
