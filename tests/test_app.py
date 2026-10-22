"""Unit tests for ThreatLens' local artifact and hybrid inference helpers.

These tests intentionally avoid loading trained model artifacts, downloading models,
or making network calls. They test deterministic functions and architecture contracts.
Run with: python -m pytest -q
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app


def test_no_psutil_or_hardware_monitoring_dependency():
    """The app should import without psutil and should not gate behavior on RAM."""
    source = Path(app.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])
    assert "psutil" not in imported_modules
    assert "psutil.virtual_memory" not in source
    assert "MIN_RAM_FOR_SLM" not in source


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Web Attack – SQL Injection", "Web Attack – SQL Injection"),
        ("Web Attack - SQL Injection", "Web Attack – SQL Injection"),
        ("web attack - xss", "Web Attack – XSS"),
        ("Web Attack – XSS", "Web Attack – XSS"),
        ("Heartbleed", "Heartbleed"),
        ("Infiltration", "Infiltration"),
    ],
)
def test_canonical_attack_labels(raw, expected):
    assert app.canonical_attack_label(raw) == expected


def test_xss_mitre_mapping_is_available():
    details = app.get_mitre_details("Web Attack – XSS")
    assert "T1190" in details["technique"]
    assert details.get("mitigation")
    assert details.get("description")


def test_sqli_mitre_mapping_is_available():
    details = app.get_mitre_details("Web Attack – SQL Injection")
    assert "T1190" in details["technique"]
    assert details.get("mitigation")


def test_threat_vault_matches_exact_prototype_above_threshold():
    prototype = np.arange(1, 41, dtype=float)
    vault = {"Web Attack – XSS": prototype.tolist()}
    label, similarity = app.match_threat_vault(prototype.copy(), vault, threshold=0.95)
    assert label == "Web Attack – XSS"
    assert similarity > 0.95
    assert similarity == pytest.approx(1.0, abs=1e-8)


def test_threat_vault_does_not_match_unrelated_vector():
    prototype = np.zeros(40, dtype=float)
    prototype[0] = 1.0
    unrelated = np.zeros(40, dtype=float)
    unrelated[1] = 1.0
    label, similarity = app.match_threat_vault(
        unrelated, {"Web Attack – XSS": prototype.tolist()}, threshold=0.95
    )
    assert label is None
    assert similarity <= 0.95


def test_threat_vault_handles_invalid_and_zero_vectors_safely():
    vault = {
        "zero": [0.0] * 40,
        "wrong_size": [1.0, 2.0],
        "valid": [float("nan")] * 40,
    }
    label, similarity = app.match_threat_vault(np.ones(40), vault, threshold=0.95)
    assert label is None
    assert similarity == 0.0


def test_align_network_features_reorders_pads_and_coerces():
    expected = ["feature_a", "feature_b", "feature_c"]
    uploaded = pd.DataFrame({
        "feature_c": ["3.5", "not-a-number"],
        "unrelated_foreign_column": ["ignored", "ignored"],
        "feature_a": ["1", "2"],
    })
    aligned = app.align_network_features(uploaded, expected)
    assert list(aligned.columns) == expected
    assert aligned.dtypes.map(lambda dtype: np.issubdtype(dtype, np.number)).all()
    assert aligned.iloc[0].tolist() == pytest.approx([1.0, 0.0, 3.5])
    assert aligned.iloc[1].tolist() == pytest.approx([2.0, 0.0, 0.0])
    assert np.isfinite(aligned.to_numpy()).all()


def test_align_network_features_rejects_duplicate_normalized_columns():
    uploaded = pd.DataFrame([[1, 2]], columns=["feature_a", " feature_a "])
    with pytest.raises(ValueError, match="Duplicate columns"):
        app.align_network_features(uploaded, ["feature_a", "feature_b"])


def test_confidence_helper_is_bounded():
    assert app.contextual_match_confidence(0.0) == pytest.approx(100.0)
    assert app.contextual_match_confidence(4.0) == pytest.approx(0.0)
    assert 0.0 <= app.contextual_match_confidence(1.5) <= 100.0


def test_required_simulation_buttons_exist_in_source():
    """Smoke-check the Streamlit UI labels without launching a Streamlit server."""
    source = Path(app.__file__).read_text(encoding="utf-8")
    assert 'st.button("Simulate SQLi Attack"' in source
    assert 'st.button("Simulate XSS Attack"' in source
