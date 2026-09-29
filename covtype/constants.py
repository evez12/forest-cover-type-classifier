"""Dataset constants shared by training and serving.

Feature order and names match ``sklearn.datasets.fetch_covtype(as_frame=True)``,
exactly as used in ``notebooks/covtype_training_and_evaluation.ipynb``.
"""

from __future__ import annotations

from typing import Final

NUM_CLASSES: Final[int] = 7
RANDOM_STATE: Final[int] = 11

# ---- Continuous features (scaled with StandardScaler) ----
NUMERIC_FEATURES: Final[list[str]] = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

# ---- One-hot binary features (passthrough) ----
WILDERNESS_FEATURES: Final[list[str]] = [f"Wilderness_Area_{i}" for i in range(4)]
SOIL_FEATURES: Final[list[str]] = [f"Soil_Type_{i}" for i in range(40)]
BINARY_FEATURES: Final[list[str]] = WILDERNESS_FEATURES + SOIL_FEATURES

# Raw column order of the DataFrame returned by fetch_covtype (54 features)
FEATURE_NAMES: Final[list[str]] = NUMERIC_FEATURES + BINARY_FEATURES

# ---- Target: Cover_Type 1..7 shifted to 0..6 (df["Cover_Type"] - 1) ----
TARGET_COLUMN: Final[str] = "Cover_Type"
TARGET_OFFSET: Final[int] = 1

CLASS_NAMES: Final[list[str]] = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]

# Fixed-order categorical palette from the notebook (one color per class)
CLASS_COLORS: Final[list[str]] = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
]

# ---- Human-readable labels for the one-hot groups (UCI covtype.info) ----
WILDERNESS_AREAS: Final[list[str]] = [
    "Rawah",
    "Neota",
    "Comanche Peak",
    "Cache la Poudre",
]

SOIL_TYPES: Final[list[str]] = [
    "2702 · Cathedral family – Rock outcrop complex, extremely stony",
    "2703 · Vanet – Ratake families complex, very stony",
    "2704 · Haploborolis – Rock outcrop complex, rubbly",
    "2705 · Ratake family – Rock outcrop complex, rubbly",
    "2706 · Vanet family – Rock outcrop complex, rubbly",
    "2717 · Vanet – Wetmore families – Rock outcrop complex, stony",
    "3501 · Gothic family",
    "3502 · Supervisor – Limber families complex",
    "4201 · Troutville family, very stony",
    "4703 · Bullwark – Catamount families – Rock outcrop complex, rubbly",
    "4704 · Bullwark – Catamount families – Rock land complex, rubbly",
    "4744 · Legault family – Rock land complex, stony",
    "4758 · Catamount family – Rock land – Bullwark family complex, rubbly",
    "5101 · Pachic Argiborolis – Aquolis complex",
    "5151 · Unspecified in the USFS Soil and ELU Survey",
    "6101 · Cryaquolis – Cryoborolis complex",
    "6102 · Gateview family – Cryaquolis complex",
    "6731 · Rogert family, very stony",
    "7101 · Typic Cryaquolis – Borohemists complex",
    "7102 · Typic Cryaquepts – Typic Cryaquolls complex",
    "7103 · Typic Cryaquolls – Leighcan family, till substratum complex",
    "7201 · Leighcan family, till substratum, extremely bouldery",
    "7202 · Leighcan family, till substratum – Typic Cryaquolls complex",
    "7700 · Leighcan family, extremely stony",
    "7701 · Leighcan family, warm, extremely stony",
    "7702 · Granile – Catamount families complex, very stony",
    "7709 · Leighcan family, warm – Rock outcrop complex, extremely stony",
    "7710 · Leighcan family – Rock outcrop complex, extremely stony",
    "7745 · Como – Legault families complex, extremely stony",
    "7746 · Como family – Rock land – Legault family complex, extremely stony",
    "7755 · Leighcan – Catamount families complex, extremely stony",
    "7756 · Catamount family – Rock outcrop – Leighcan family complex, extremely stony",
    "7757 · Leighcan – Catamount families – Rock outcrop complex, extremely stony",
    "7790 · Cryorthents – Rock land complex, extremely stony",
    "8703 · Cryumbrepts – Rock outcrop – Cryaquepts complex",
    "8707 · Bross family – Rock land – Cryumbrepts complex, extremely stony",
    "8708 · Rock outcrop – Cryumbrepts – Cryorthents complex, extremely stony",
    "8771 · Leighcan – Moran families – Cryaquolls complex, extremely stony",
    "8772 · Moran family – Cryorthents – Leighcan family complex, extremely stony",
    "8776 · Moran family – Cryorthents – Rock land complex, extremely stony",
]

assert len(FEATURE_NAMES) == 54
assert len(SOIL_TYPES) == len(SOIL_FEATURES)
assert len(CLASS_NAMES) == len(CLASS_COLORS) == NUM_CLASSES
