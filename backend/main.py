"""FastAPI application.

Run from the project root:
    uvicorn backend.main:app --reload

Endpoints:
    GET  /health          liveness + model status
    GET  /model-info      architecture, classes, feature stats, test metrics
    GET  /examples        one representative test row per class
    POST /predict         single prediction
    POST /predict/batch   batch prediction
    GET  /docs            Swagger UI
    GET  /                web UI (served from ./frontend)
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.predictor import ArtifactsNotFoundError, CoverTypePredictor, Prediction
from backend.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ClassInfo,
    ClassProbability,
    CoverTypeFeatures,
    ExampleResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("covtype.api")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model + preprocessor once at startup."""
    app.state.predictor = None
    app.state.load_error = None
    try:
        app.state.predictor = CoverTypePredictor.from_artifacts(settings.artifacts_dir, settings.device)
    except ArtifactsNotFoundError as exc:
        app.state.load_error = str(exc)
        logger.error(str(exc))
    except Exception as exc:  # noqa: BLE001 — keep the server up, report via /health
        app.state.load_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed to load model artifacts")
    yield
    app.state.predictor = None


app = FastAPI(
    title="Forest Cover Type Classifier API",
    description="PyTorch MLP trained on the UCI Covertype dataset (54 features → 7 cover types).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_predictor(request: Request) -> CoverTypePredictor:
    predictor: CoverTypePredictor | None = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=request.app.state.load_error or "Model is not loaded",
        )
    return predictor


PredictorDep = Annotated[CoverTypePredictor, Depends(get_predictor)]


def _to_response(pred: Prediction, latency_ms: float | None = None) -> PredictionResponse:
    return PredictionResponse(
        class_index=pred.class_index,
        cover_type=pred.cover_type,
        class_name=pred.class_name,
        confidence=pred.confidence,
        probabilities=[ClassProbability(**p) for p in pred.probabilities],
        warnings=pred.warnings,
        latency_ms=latency_ms,
    )


# --------------------------------------------------------------------------- #
# Routes (sync `def` -> executed in the threadpool, so inference never blocks the event loop)
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    predictor: CoverTypePredictor | None = request.app.state.predictor
    if predictor is None:
        return HealthResponse(status="degraded", model_loaded=False, detail=request.app.state.load_error)
    return HealthResponse(status="ok", model_loaded=True, device=str(predictor.device))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info(predictor: PredictorDep) -> ModelInfoResponse:
    meta = predictor.metadata
    test = meta.get("metrics", {}).get("test", {})
    return ModelInfoResponse(
        model_class=meta["model"]["class"],
        input_size=meta["model"]["input_size"],
        num_classes=meta["model"]["num_classes"],
        hidden_layers=meta["model"]["hidden_layers"],
        created_at=meta.get("created_at"),
        exported_epoch=meta.get("training", {}).get("exported_epoch"),
        feature_names=meta["preprocessing"]["feature_names"],
        numeric_features=meta["preprocessing"]["numeric_features"],
        binary_features=meta["preprocessing"]["binary_features"],
        classes=[ClassInfo(**vars(lbl)) for lbl in predictor.labels],
        feature_stats=predictor.feature_stats,
        test_metrics={k: v for k, v in test.items() if isinstance(v, (int, float))},
    )


@app.get("/examples", response_model=list[ExampleResponse], tags=["model"])
def examples(predictor: PredictorDep) -> list[ExampleResponse]:
    return [ExampleResponse(**ex) for ex in predictor.metadata.get("examples", [])]


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(features: CoverTypeFeatures, predictor: PredictorDep) -> PredictionResponse:
    start = time.perf_counter()
    pred = predictor.predict([features.model_dump()])[0]
    return _to_response(pred, latency_ms=(time.perf_counter() - start) * 1000)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(payload: BatchPredictionRequest, predictor: PredictorDep) -> BatchPredictionResponse:
    if len(payload.instances) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Batch size {len(payload.instances)} exceeds limit {settings.max_batch_size}",
        )
    start = time.perf_counter()
    preds = predictor.predict([inst.model_dump() for inst in payload.instances])
    return BatchPredictionResponse(
        predictions=[_to_response(p) for p in preds],
        count=len(preds),
        latency_ms=(time.perf_counter() - start) * 1000,
    )


# --------------------------------------------------------------------------- #
# Frontend (mounted last so API routes take precedence)
# --------------------------------------------------------------------------- #
if settings.frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
else:
    logger.warning("Frontend directory not found: %s", settings.frontend_dir)
