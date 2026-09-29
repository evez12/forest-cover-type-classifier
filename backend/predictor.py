"""Artifact loading and inference.

Inference pipeline (identical to training):
    raw 54 features (DataFrame, notebook column order)
      -> fitted ColumnTransformer (StandardScaler on numeric, passthrough on one-hot)
      -> float32 tensor
      -> CoverTypeClassifier in eval mode (BatchNorm uses running statistics)
      -> softmax -> class_index (0..6) -> cover_type = class_index + 1
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer

from covtype.constants import CLASS_COLORS, FEATURE_NAMES, NUMERIC_FEATURES
from covtype.model import CoverTypeClassifier

logger = logging.getLogger(__name__)

MODEL_FILENAME = "covtype_model.pth"
PREPROCESSOR_FILENAME = "preprocessor.joblib"
METADATA_FILENAME = "metadata.json"


class ArtifactsNotFoundError(FileNotFoundError):
    """Raised when training artifacts are missing."""


@dataclass(frozen=True)
class ClassLabel:
    class_index: int
    cover_type: int
    class_name: str
    color: str


@dataclass(frozen=True)
class Prediction:
    class_index: int
    cover_type: int
    class_name: str
    confidence: float
    probabilities: list[dict[str, Any]]
    warnings: list[str]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")
    return device


class CoverTypePredictor:
    def __init__(
        self,
        model: CoverTypeClassifier,
        preprocessor: ColumnTransformer,
        metadata: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.device = device

        self.labels: list[ClassLabel] = [
            ClassLabel(
                class_index=c["class_index"],
                cover_type=c["cover_type"],
                class_name=c["class_name"],
                color=CLASS_COLORS[c["class_index"]],
            )
            for c in metadata["target"]["classes"]
        ]
        self.feature_stats: dict[str, dict[str, float]] = metadata.get("feature_stats", {})

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @classmethod
    def from_artifacts(cls, artifacts_dir: Path, device: str = "cpu") -> Self:
        paths = {name: artifacts_dir / name for name in (MODEL_FILENAME, PREPROCESSOR_FILENAME, METADATA_FILENAME)}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise ArtifactsNotFoundError(
                f"Missing artifacts: {missing}. Run `python train_and_export.py` first."
            )

        metadata: dict[str, Any] = json.loads(paths[METADATA_FILENAME].read_text(encoding="utf-8"))
        preprocessor: ColumnTransformer = joblib.load(paths[PREPROCESSOR_FILENAME])

        # Guard against a preprocessor fit on a different column order
        fitted_cols = list(getattr(preprocessor, "feature_names_in_", []))
        if fitted_cols != FEATURE_NAMES:
            raise ValueError("Preprocessor feature order does not match covtype.constants.FEATURE_NAMES")

        torch_device = resolve_device(device)
        model_meta = metadata["model"]
        model = CoverTypeClassifier(model_meta["input_size"], model_meta["num_classes"])
        state_dict = torch.load(paths[MODEL_FILENAME], map_location=torch_device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(torch_device).eval()

        logger.info("Loaded CoverTypeClassifier (%d features) on %s", model_meta["input_size"], torch_device)
        return cls(model, preprocessor, metadata, torch_device)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _to_frame(self, records: Sequence[Mapping[str, float]]) -> pd.DataFrame:
        return pd.DataFrame.from_records(list(records), columns=FEATURE_NAMES).astype(np.float64)

    @torch.inference_mode()
    def predict_proba(self, records: Sequence[Mapping[str, float]], batch_size: int = 16_384) -> np.ndarray:
        """Return softmax probabilities of shape (N, num_classes)."""
        X = np.asarray(self.preprocessor.transform(self._to_frame(records)), dtype=np.float32)
        X_t = torch.from_numpy(X)
        probs: list[torch.Tensor] = []
        for i in range(0, X_t.size(0), batch_size):
            logits = self.model(X_t[i : i + batch_size].to(self.device))
            probs.append(torch.softmax(logits.float(), dim=1).cpu())
        return torch.cat(probs).numpy()

    def _range_warnings(self, record: Mapping[str, float]) -> list[str]:
        warnings: list[str] = []
        for name in NUMERIC_FEATURES:
            stats = self.feature_stats.get(name)
            if stats is None:
                continue
            value = float(record[name])
            if value < stats["min"] or value > stats["max"]:
                warnings.append(
                    f"{name}={value:g} is outside the training range "
                    f"[{stats['min']:g}, {stats['max']:g}] — prediction is an extrapolation."
                )
        return warnings

    def predict(self, records: Sequence[Mapping[str, float]]) -> list[Prediction]:
        proba = self.predict_proba(records)
        results: list[Prediction] = []
        for record, p in zip(records, proba, strict=True):
            top = int(p.argmax())
            ranked = sorted(
                (
                    {
                        "class_index": lbl.class_index,
                        "cover_type": lbl.cover_type,
                        "class_name": lbl.class_name,
                        "probability": float(p[lbl.class_index]),
                    }
                    for lbl in self.labels
                ),
                key=lambda d: d["probability"],
                reverse=True,
            )
            label = self.labels[top]
            results.append(
                Prediction(
                    class_index=label.class_index,
                    cover_type=label.cover_type,
                    class_name=label.class_name,
                    confidence=float(p[top]),
                    probabilities=ranked,
                    warnings=self._range_warnings(record),
                )
            )
        return results
