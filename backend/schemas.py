"""Request / response schemas.

Input field names are exactly the 54 column names of
``sklearn.datasets.fetch_covtype(as_frame=True)`` used in the notebook.
One-hot fields default to 0, so a client may send only the active ones;
exactly one ``Wilderness_Area_*`` and exactly one ``Soil_Type_*`` must be 1.
"""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from covtype.constants import SOIL_FEATURES, WILDERNESS_FEATURES

Binary = Annotated[int, Field(ge=0, le=1, description="One-hot indicator (0 or 1)")]

# First row of the Covertype dataset (true label: Cover_Type 5 = Aspen)
_EXAMPLE: dict[str, Any] = {
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
    **{name: 0 for name in WILDERNESS_FEATURES},
    **{name: 0 for name in SOIL_FEATURES},
    "Wilderness_Area_0": 1,
    "Soil_Type_28": 1,
}


class CoverTypeFeatures(BaseModel):
    """A single 30 m × 30 m cell described by the 54 Covertype features."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": _EXAMPLE})

    # ---- Continuous features (StandardScaler) ----
    Elevation: float = Field(ge=0, le=9000, description="Elevation in meters")
    Aspect: float = Field(ge=0, le=360, description="Aspect in degrees azimuth")
    Slope: float = Field(ge=0, le=90, description="Slope in degrees")
    Horizontal_Distance_To_Hydrology: float = Field(ge=0, description="Horizontal distance to nearest surface water (m)")
    Vertical_Distance_To_Hydrology: float = Field(ge=-2000, le=2000, description="Vertical distance to nearest surface water (m)")
    Horizontal_Distance_To_Roadways: float = Field(ge=0, description="Horizontal distance to nearest roadway (m)")
    Hillshade_9am: float = Field(ge=0, le=255, description="Hillshade index at 9am, summer solstice (0–255)")
    Hillshade_Noon: float = Field(ge=0, le=255, description="Hillshade index at noon, summer solstice (0–255)")
    Hillshade_3pm: float = Field(ge=0, le=255, description="Hillshade index at 3pm, summer solstice (0–255)")
    Horizontal_Distance_To_Fire_Points: float = Field(ge=0, description="Horizontal distance to nearest wildfire ignition point (m)")

    # ---- Wilderness area one-hot (passthrough) ----
    Wilderness_Area_0: Binary = 0
    Wilderness_Area_1: Binary = 0
    Wilderness_Area_2: Binary = 0
    Wilderness_Area_3: Binary = 0

    # ---- Soil type one-hot (passthrough) ----
    Soil_Type_0: Binary = 0
    Soil_Type_1: Binary = 0
    Soil_Type_2: Binary = 0
    Soil_Type_3: Binary = 0
    Soil_Type_4: Binary = 0
    Soil_Type_5: Binary = 0
    Soil_Type_6: Binary = 0
    Soil_Type_7: Binary = 0
    Soil_Type_8: Binary = 0
    Soil_Type_9: Binary = 0
    Soil_Type_10: Binary = 0
    Soil_Type_11: Binary = 0
    Soil_Type_12: Binary = 0
    Soil_Type_13: Binary = 0
    Soil_Type_14: Binary = 0
    Soil_Type_15: Binary = 0
    Soil_Type_16: Binary = 0
    Soil_Type_17: Binary = 0
    Soil_Type_18: Binary = 0
    Soil_Type_19: Binary = 0
    Soil_Type_20: Binary = 0
    Soil_Type_21: Binary = 0
    Soil_Type_22: Binary = 0
    Soil_Type_23: Binary = 0
    Soil_Type_24: Binary = 0
    Soil_Type_25: Binary = 0
    Soil_Type_26: Binary = 0
    Soil_Type_27: Binary = 0
    Soil_Type_28: Binary = 0
    Soil_Type_29: Binary = 0
    Soil_Type_30: Binary = 0
    Soil_Type_31: Binary = 0
    Soil_Type_32: Binary = 0
    Soil_Type_33: Binary = 0
    Soil_Type_34: Binary = 0
    Soil_Type_35: Binary = 0
    Soil_Type_36: Binary = 0
    Soil_Type_37: Binary = 0
    Soil_Type_38: Binary = 0
    Soil_Type_39: Binary = 0

    @model_validator(mode="after")
    def check_one_hot_groups(self) -> Self:
        for group_name, fields in (("Wilderness_Area", WILDERNESS_FEATURES), ("Soil_Type", SOIL_FEATURES)):
            active = sum(getattr(self, f) for f in fields)
            if active != 1:
                raise ValueError(
                    f"Exactly one {group_name}_* field must be 1 (got {active}); "
                    "the model was trained on strictly one-hot encoded groups."
                )
        return self


class BatchPredictionRequest(BaseModel):
    instances: list[CoverTypeFeatures] = Field(min_length=1)


class ClassProbability(BaseModel):
    class_index: int = Field(description="Model output index (0..6) = Cover_Type - 1")
    cover_type: int = Field(description="Original UCI Cover_Type label (1..7)")
    class_name: str
    probability: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    class_index: int
    cover_type: int
    class_name: str
    confidence: float = Field(ge=0, le=1, description="Softmax probability of the predicted class")
    probabilities: list[ClassProbability] = Field(description="All classes, sorted by probability (desc)")
    warnings: list[str] = Field(default_factory=list, description="Inputs outside the training range")
    latency_ms: float | None = None


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None = None
    detail: str | None = None


class ClassInfo(BaseModel):
    class_index: int
    cover_type: int
    class_name: str
    color: str


class ModelInfoResponse(BaseModel):
    model_class: str
    input_size: int
    num_classes: int
    hidden_layers: list[int]
    created_at: str | None
    exported_epoch: int | None
    feature_names: list[str]
    numeric_features: list[str]
    binary_features: list[str]
    classes: list[ClassInfo]
    feature_stats: dict[str, dict[str, float]]
    test_metrics: dict[str, float]


class ExampleResponse(BaseModel):
    class_index: int
    cover_type: int
    class_name: str
    model_confidence: float
    features: dict[str, float]
