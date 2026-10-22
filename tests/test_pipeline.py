"""Tests for ThreatLens preprocessing and hybrid local inference."""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import pytest

import app


def test_mitre_knowledge_base_has_six_complete_entries():
    assert set(app.MITRE) == {"T1046", "T1110", "T1071", "T1499", "T1190", "T1078"}
    for key, item in app.MITRE.items():
        assert item["id"] == key
        assert item["name"] and item["tactic"] and item["description"]
        assert len(item["mitigations"]) == 3


def test_column_normalization_maps_common_cicids_names():
    frame = pd.DataFrame(columns=[
        "Destination Port", "Flow Duration", "Total Fwd Packets",
        "Total Backward Packets", "Flow Bytes/s", "Average Packet Size",
        "SYN Flag Count", "Src IP", "Dst IP",
    ])
    normalized = app.normalize_columns(frame)
    assert set(app.FEATURES).issubset(normalized.columns)
    assert {"Source_IP", "Dest_IP"}.issubset(normalized.columns)


def test_preprocessor_cleans_invalid_values_and_explains_missing_features():
    raw = pd.DataFrame([{
        "Destination Port": "443", "Flow Duration": "inf",
        "Total Fwd Packets": "oops", "Total Backward Packets": 8,
        "Flow Bytes/s": np.nan, "Average Packet Size": 400, "SYN Flag Count": 3,
    }])
    cleaned, report = app.preprocess_network_chunk(raw)
    assert set(app.FEATURES).issubset(cleaned.columns)
    assert cleaned.loc[0, "Destination_Port"] == 443
    assert cleaned.loc[0, "Flow_Duration"] == 0
    assert cleaned.loc[0, "Total_Fwd_Packets"] == 0
    assert report["invalid_numeric_cells_replaced_with_zero"] >= 3
    assert "None" not in cleaned.loc[0, "Preprocessing_Warning"]


def test_preprocessor_derives_flow_bytes_rate_from_total_bytes():
    raw = pd.DataFrame([{
        "Flow Duration": 2_000_000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 5,
        "Total Length of Fwd Packets": 1000,
        "Total Length of Bwd Packets": 500,
        "Destination Port": 443,
        "Average Packet Size": 100,
        "SYN Flag Count": 1,
    }])
    cleaned, report = app.preprocess_network_chunk(raw)
    assert cleaned.loc[0, "Flow_Bytes_s"] == pytest.approx(750)
    assert report["derived_features"]["Flow_Bytes_s"] == 1


def test_arbitrary_schema_is_processed_but_missing_fields_are_disclosed():
    cleaned, report = app.preprocess_network_chunk(pd.DataFrame({"timestamp": ["2026-01-01"], "bytes": [5]}))
    assert set(app.FEATURES).issubset(cleaned.columns)
    assert len(report["missing_features"]) > 0
    assert "filled with zero" in cleaned.loc[0, "Preprocessing_Warning"]


def test_synthetic_dataset_has_500_rows_and_binary_labels():
    data = app.generate_synthetic_data(500, seed=7)
    assert len(data) == 500
    assert set(app.FEATURES).issubset(data.columns)
    assert set(data["Label"].unique()) == {0, 1}
    assert {"Flow_ID", "Source_IP", "Dest_IP"}.issubset(data.columns)


@pytest.fixture(scope="module")
def bundle():
    return app.train_model()


def test_xgboost_bundle_and_prediction(bundle):
    assert bundle["metrics"]["test_rows"] == 125
    assert 0 <= bundle["metrics"]["accuracy"] <= 1
    sample, _ = app.preprocess_network_chunk(app.generate_synthetic_data(30))
    predicted = app.predict_chunk(sample, bundle)
    assert {"Threat_Probability", "Prediction", "Prediction_Label", "Priority", "Quantum_Routed"}.issubset(predicted.columns)
    assert predicted["Threat_Probability"].between(0, 1).all()
    assert set(predicted["Prediction"].unique()).issubset({0, 1})


def test_shap_explanation_returns_top_three(bundle):
    sample, _ = app.preprocess_network_chunk(app.generate_synthetic_data(20))
    predicted = app.predict_chunk(sample, bundle)
    explanation = app.explain_flow(predicted.iloc[0], bundle)
    assert len(explanation) == 3
    assert all(item["feature"] in app.FEATURES for item in explanation)
    assert all(item["absolute_shap"] >= 0 for item in explanation)


def test_quantum_score_is_mapped_to_zero_one(bundle):
    sample, _ = app.preprocess_network_chunk(app.generate_synthetic_data(10))
    predicted = app.predict_chunk(sample, bundle)
    score = app.quantum_risk_score(predicted.iloc[0], bundle)
    assert 0.0 <= score <= 1.0


def test_chunked_csv_analysis_streams_all_rows_and_preserves_preview(bundle):
    data = app.generate_synthetic_data(500)
    result = app.analyze_csv_in_chunks(io.StringIO(data.to_csv(index=False)), bundle, chunk_size=73)
    try:
        assert result["total_rows"] == 500
        assert len(result["preview"]) == 500
        assert len(result["alerts"]) <= 200
        on_disk = pd.read_csv(result["output_path"])
        assert len(on_disk) == 500
        assert {"Prediction", "Threat_Probability", "Preprocessing_Warning"}.issubset(on_disk.columns)
    finally:
        if os.path.exists(result["output_path"]):
            os.unlink(result["output_path"])
