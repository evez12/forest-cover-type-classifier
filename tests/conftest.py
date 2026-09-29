"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.config import get_settings
from backend.predictor import METADATA_FILENAME, MODEL_FILENAME, PREPROCESSOR_FILENAME

REQUIRED_ARTIFACTS = (MODEL_FILENAME, PREPROCESSOR_FILENAME, METADATA_FILENAME)


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """API client with the model loaded (the lifespan runs inside the context manager)."""
    artifacts_dir = get_settings().artifacts_dir
    missing = [name for name in REQUIRED_ARTIFACTS if not (artifacts_dir / name).is_file()]
    if missing:
        pytest.skip(f"Model artifacts missing in {artifacts_dir}: {missing}")

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def valid_payload() -> dict[str, Any]:
    """First row of the Covertype dataset (sparse one-hot form)."""
    return {
        "Elevation": 2596,
        "Aspect": 51,
        "Slope": 3,
        "Horizontal_Distance_To_Hydrology": 258,
        "Vertical_Distance_To_Hydrology": 0,
        "Horizontal_Distance_To_Roadways": 510,
        "Hillshade_9am": 221,
        "Hillshade_Noon": 232,
        "Hillshade_3pm": 148,
        "Horizontal_Distance_To_Fire_Points": 6279,
        "Wilderness_Area_0": 1,
        "Soil_Type_28": 1,
    }
