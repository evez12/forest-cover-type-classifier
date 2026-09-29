"""Unit tests for the shared model package."""

from __future__ import annotations

import torch

from covtype.constants import CLASS_NAMES, FEATURE_NAMES, NUM_CLASSES
from covtype.model import CoverTypeClassifier


def test_forward_output_shape() -> None:
    model = CoverTypeClassifier(input_size=len(FEATURE_NAMES), num_class=NUM_CLASSES).eval()
    x = torch.randn(8, len(FEATURE_NAMES))
    with torch.inference_mode():
        logits = model(x)
    assert logits.shape == (8, NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_constants_are_consistent() -> None:
    assert len(FEATURE_NAMES) == 54
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert len(CLASS_NAMES) == NUM_CLASSES == 7
