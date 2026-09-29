"""The static GitHub Pages build must reproduce the PyTorch model exactly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from scripts.build_static_site import build, numpy_forward


def test_static_site_build(client: TestClient, tmp_path: Path) -> None:
    out = tmp_path / "site"
    build(out)  # raises if the BatchNorm-folded network diverges from PyTorch

    for name in ("index.html", "app.js", "static-api.js", "api/model.json", "api/examples.json", "api/model-info.json"):
        assert (out / name).is_file(), name
    html = (out / "index.html").read_text(encoding="utf-8")
    assert html.index("static-api.js") < html.index("app.js")

    model = json.loads((out / "api/model.json").read_text(encoding="utf-8"))
    assert len(model["scaler"]["mean"]) == len(model["numeric_bounds"]) == 10
    assert model["layers"][-1]["shape"] == [7, 32]
    assert not model["layers"][-1]["relu"]


def test_numpy_forward_is_a_distribution() -> None:
    layers = [{"W": np.eye(3, dtype=np.float32), "b": np.zeros(3, dtype=np.float32), "relu": False}]
    p = numpy_forward(layers, np.array([[1.0, 2.0, 3.0]]))
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 2
