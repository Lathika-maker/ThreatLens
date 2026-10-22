"""Focused tests for ThreatLens schema safety and evidence logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app
import prepare_inference_csv
import train_model


def feature_names():
    return [f"feature_{i:02d}" for i in range(40)]


def valid_frame(rows=2):
    return pd.DataFrame(
        {name: np.arange(rows, dtype=np.float64) + i for i, name in enumerate(feature_names())}
    )


def test_feature_contract_is_exactly_40_columns():
    assert app.MAX_FEATURES == 40
    assert len(feature_names()) == app.MAX_FEATURES


def test_normalizers_trim_bom_and_convert_spaces():
    assert app.normalize_column("\ufeffDestination Port ") == "Destination_Port"
    assert train_model.normalize("Flow Duration") == "Flow_Duration"
    assert prepare_inference_csv.normalize("Flow Duration") == "Flow_Duration"


def test_exact_schema_accepts_valid_numeric_frame_as_float32():
    result = app.validate_input_frame(valid_frame(), feature_names())
    assert list(result.columns) == feature_names()
    assert result.shape == (2, 40)
    assert all(dtype == np.dtype("float32") for dtype in result.dtypes)


def test_strict_schema_rejects_missing_feature():
    frame = valid_frame().drop(columns=["feature_03"])
    with pytest.raises(ValueError, match="missing required features"):
        app.validate_input_frame(frame, feature_names())


def test_strict_schema_rejects_extra_label_column():
    frame = valid_frame()
    frame["Label"] = "BENIGN"
    with pytest.raises(ValueError, match="unexpected columns"):
        app.validate_input_frame(frame, feature_names())


def test_strict_schema_rejects_duplicate_names_after_normalization():
    frame = valid_frame()
    frame["feature 00"] = frame["feature_00"]
    with pytest.raises(ValueError, match="duplicate normalized columns"):
        app.validate_input_frame(frame, feature_names())


@pytest.mark.parametrize("bad_value", ["bad", np.nan, np.inf, -np.inf])
def test_strict_schema_rejects_invalid_numeric_values(bad_value):
    frame = valid_frame()
    frame.loc[0, "feature_02"] = bad_value
    with pytest.raises(ValueError, match="non-numeric, missing, or infinite"):
        app.validate_input_frame(frame, feature_names())


def test_row_limit_is_checked_before_inference():
    with pytest.raises(ValueError, match="expected 1–1 data rows"):
        app.validate_input_frame(valid_frame(rows=2), feature_names(), max_rows=1)


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(0.0, 100.0), (1.0, 75.0), (2.0, 50.0), (4.0, 0.0), (9.0, 0.0)],
)
def test_contextual_match_confidence_is_bounded(distance, expected):
    assert app.contextual_match_confidence(distance) == pytest.approx(expected)


def test_shap_class_extraction_supports_samples_features_classes_layout():
    class Explainer:
        def shap_values(self, _frame):
            return np.array([[[0.1, 0.2], [0.3, -0.4], [0.5, 0.9]]])

    row = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["a", "b", "c"])
    result = app.shap_values_for_class(Explainer(), row, class_index=1)
    np.testing.assert_allclose(result, [0.2, -0.4, 0.9])


def test_evidence_ranks_absolute_shap_impact_and_disclaims_causation():
    row = pd.Series([10.0, 20.0, 30.0], index=["a", "b", "c"])
    evidence, items = app.make_evidence(row, np.array([0.1, -0.8, 0.4]), ["a", "b", "c"])
    assert [item["feature"] for item in items] == ["b", "c", "a"]
    assert "SHAP describes model influence, not causation" in evidence
    assert "b=20" in evidence


def test_safe_text_removes_control_characters_and_limits_length():
    assert app.safe_text("abc\n\x00def", limit=20) == "abcdef"
    assert len(app.safe_text("x" * 200, limit=32)) == 32
