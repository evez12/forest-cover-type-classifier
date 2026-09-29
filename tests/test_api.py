"""Integration tests for the FastAPI inference service (uses the exported artifacts)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import backend.main as api
from covtype.constants import CLASS_NAMES, FEATURE_NAMES, NUM_CLASSES


def test_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client: TestClient) -> None:
    res = client.get("/model-info")
    assert res.status_code == 200
    body = res.json()
    assert body["input_size"] == len(FEATURE_NAMES)
    assert body["num_classes"] == NUM_CLASSES
    assert [c["class_name"] for c in body["classes"]] == CLASS_NAMES
    assert 0.0 <= body["test_metrics"]["accuracy"] <= 1.0


def test_predict_returns_valid_distribution(client: TestClient, valid_payload: dict[str, Any]) -> None:
    res = client.post("/predict", json=valid_payload)
    assert res.status_code == 200
    body = res.json()

    probs = [p["probability"] for p in body["probabilities"]]
    assert len(probs) == NUM_CLASSES
    assert sum(probs) == pytest.approx(1.0, abs=1e-5)
    assert probs == sorted(probs, reverse=True)
    assert body["confidence"] == pytest.approx(probs[0])
    assert body["cover_type"] == body["class_index"] + 1
    assert body["class_name"] == CLASS_NAMES[body["class_index"]]
    assert body["warnings"] == []


def test_examples_are_classified_correctly(client: TestClient) -> None:
    """Every preset is a correctly classified test-set row -> guards against pipeline drift."""
    examples = client.get("/examples").json()
    assert len(examples) == NUM_CLASSES
    for ex in examples:
        pred = client.post("/predict", json=ex["features"]).json()
        assert pred["class_index"] == ex["class_index"], ex["class_name"]


def test_out_of_range_input_emits_warning(client: TestClient, valid_payload: dict[str, Any]) -> None:
    res = client.post("/predict", json={**valid_payload, "Elevation": 5000})
    assert res.status_code == 200
    assert any("Elevation" in w for w in res.json()["warnings"])


@pytest.mark.parametrize(
    ("patch", "drop"),
    [
        ({"Soil_Type_3": 1}, None),               # two soil types active
        ({}, "Wilderness_Area_0"),                # no wilderness area active
        ({"Unknown_Feature": 1}, None),           # extra fields are forbidden
        ({"Slope": -5}, None),                    # out of schema bounds
        ({"Soil_Type_28": 2}, None),              # one-hot value must be 0/1
    ],
)
def test_predict_rejects_invalid_payloads(
    client: TestClient, valid_payload: dict[str, Any], patch: dict[str, Any], drop: str | None
) -> None:
    payload = {**valid_payload, **patch}
    if drop is not None:
        payload.pop(drop)
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_prediction(client: TestClient, valid_payload: dict[str, Any]) -> None:
    res = client.post("/predict/batch", json={"instances": [valid_payload, valid_payload]})
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 2
    assert body["predictions"][0]["class_index"] == body["predictions"][1]["class_index"]


def test_batch_size_limit(
    client: TestClient, valid_payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "settings", replace(api.settings, max_batch_size=1))
    res = client.post("/predict/batch", json={"instances": [valid_payload, valid_payload]})
    assert res.status_code == 413


def test_frontend_is_served(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "Forest Cover Type Classifier" in res.text
