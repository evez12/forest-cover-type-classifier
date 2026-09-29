"""Build a fully static version of the web app (GitHub Pages) with in-browser inference.

The FastAPI backend is replaced by ``frontend/static-api.js``, which serves the same
endpoints from pre-exported JSON and runs the MLP forward pass in JavaScript.

Export steps:
  1. Load the production artifacts through ``CoverTypePredictor`` (same code path as the API).
  2. Fold every eval-mode BatchNorm1d into the preceding Linear layer:
        W' = diag(s) W,  b' = s * b + (beta - running_mean * s),  s = gamma / sqrt(running_var + eps)
  3. Export StandardScaler statistics, schema bounds, class labels and weights (float32, base64).
  4. Dump /model-info and /examples responses from the real API so the UI sees identical data.
  5. Verify the folded network against the PyTorch model on every preset example.

Usage:
    python scripts/build_static_site.py --out _site
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings  # noqa: E402
from backend.predictor import CoverTypePredictor  # noqa: E402
from backend.schemas import CoverTypeFeatures  # noqa: E402
from covtype.constants import (  # noqa: E402
    BINARY_FEATURES,
    FEATURE_NAMES,
    NUMERIC_FEATURES,
    SOIL_FEATURES,
    WILDERNESS_FEATURES,
)

STATIC_SCRIPT = "static-api.js"


def fold_batchnorm(model: nn.Module) -> list[dict[str, Any]]:
    """Return affine layers (Linear with BatchNorm folded in) and their activation flags."""
    modules = list(model.layers)
    layers: list[dict[str, Any]] = []
    for i, module in enumerate(modules):
        if not isinstance(module, nn.Linear):
            continue
        W = module.weight.detach().double()
        b = module.bias.detach().double()
        nxt = modules[i + 1] if i + 1 < len(modules) else None
        if isinstance(nxt, nn.BatchNorm1d):
            s = nxt.weight.detach().double() / torch.sqrt(nxt.running_var.double() + nxt.eps)
            W = W * s[:, None]
            b = b * s + (nxt.bias.detach().double() - nxt.running_mean.double() * s)
            after = modules[i + 2] if i + 2 < len(modules) else None
        else:
            after = nxt
        layers.append({"W": W.float().numpy(), "b": b.float().numpy(), "relu": isinstance(after, nn.ReLU)})
    return layers


def numpy_forward(layers: list[dict[str, Any]], X: np.ndarray) -> np.ndarray:
    h = X.astype(np.float32)
    for layer in layers:
        h = h @ layer["W"].T + layer["b"]
        if layer["relu"]:
            h = np.maximum(h, 0.0)
    e = np.exp(h - h.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def schema_bounds() -> dict[str, list[float | None]]:
    props = CoverTypeFeatures.model_json_schema()["properties"]
    return {name: [props[name].get("minimum"), props[name].get("maximum")] for name in NUMERIC_FEATURES}


def b64_float32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def build(out_dir: Path) -> None:
    from fastapi.testclient import TestClient

    from backend.main import app

    settings = get_settings()
    predictor = CoverTypePredictor.from_artifacts(settings.artifacts_dir, "cpu")
    scaler = predictor.preprocessor.named_transformers_["num"]
    layers = fold_batchnorm(predictor.model)

    with TestClient(app) as client:
        model_info = client.get("/model-info").json()
        examples = client.get("/examples").json()

    # ---- Parity check: folded numpy network vs. PyTorch pipeline ----
    records = [ex["features"] for ex in examples]
    X = np.asarray(predictor.preprocessor.transform(predictor._to_frame(records)), dtype=np.float32)
    max_err = float(np.abs(numpy_forward(layers, X) - predictor.predict_proba(records)).max())
    print(f"BatchNorm folding parity: max |Δp| = {max_err:.2e}")
    if max_err > 1e-4:
        raise SystemExit("Folded network diverges from the PyTorch model")

    model_json = {
        "feature_names": FEATURE_NAMES,
        "binary_features": BINARY_FEATURES,
        "wilderness_features": WILDERNESS_FEATURES,
        "soil_features": SOIL_FEATURES,
        "numeric_bounds": schema_bounds(),
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "classes": [
            {"class_index": c.class_index, "cover_type": c.cover_type, "class_name": c.class_name}
            for c in predictor.labels
        ],
        "feature_stats": {k: {"min": v["min"], "max": v["max"]} for k, v in predictor.feature_stats.items()},
        "layers": [
            {"shape": list(layer["W"].shape), "relu": layer["relu"], "weight": b64_float32(layer["W"]), "bias": b64_float32(layer["b"])}
            for layer in layers
        ],
    }

    # ---- Assemble the site ----
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(settings.frontend_dir, out_dir)
    api_dir = out_dir / "api"
    api_dir.mkdir()
    for name, payload in (("model.json", model_json), ("model-info.json", model_info), ("examples.json", examples)):
        (api_dir / name).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    index = out_dir / "index.html"
    html = index.read_text(encoding="utf-8")
    tag = '<script src="app.js" defer></script>'
    if tag not in html:
        raise SystemExit("Could not find the app.js script tag in index.html")
    index.write_text(html.replace(tag, f'<script src="{STATIC_SCRIPT}"></script>\n  {tag}'), encoding="utf-8")
    (out_dir / ".nojekyll").touch()

    size_kb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1024
    print(f"Static site written to {out_dir.resolve()} ({size_kb:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "_site")
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
