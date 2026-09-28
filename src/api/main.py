"""NIDS inference API.

Start it from the project root:
    python scripts/run_api.py
then open http://127.0.0.1:8000/docs for the interactive page.

Endpoints:
    GET  /health              is the service up and is a model loaded
    GET  /model-info          model name, classes, required feature names
    POST /predict             classify one flow
    POST /predict/batch       classify many flows at once
    GET  /predictions/recent  latest logged predictions
    GET  /stats               counts per predicted class
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, Response

from src.api.inference import InferenceService, MissingFeaturesError
from src.api.schemas import BatchRequest, BatchResponse, FlowRequest, PredictionResponse
from src.api.storage import PredictionStore
from src.utils.logging_config import get_logger
from src.utils.paths import load_config, resolve_path

logger = get_logger(__name__)


def create_app(
    model_path: Path | None = None,
    preprocessing_path: Path | None = None,
    db_path: Path | None = None,
    benign_label: str | None = None,
    max_batch_size: int | None = None,
) -> FastAPI:
    """Build the app. Every argument defaults to the value in configs/config.yaml;
    the arguments exist so tests can point the app at temporary files."""
    config = load_config()
    api_cfg = config["api"]

    model_path = model_path or resolve_path(config["training"]["model_path"])
    preprocessing_path = preprocessing_path or resolve_path(config["preprocessing"]["artifact_path"])
    db_path = db_path or resolve_path(api_cfg["db_path"])
    benign_label = benign_label or api_cfg["benign_label"]
    max_batch_size = max_batch_size or api_cfg["max_batch_size"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # loaded once at startup, not per request
        app.state.service = InferenceService.from_files(model_path, preprocessing_path, benign_label)
        app.state.store = PredictionStore(db_path)
        yield

    app = FastAPI(
        title="NIDS Inference API",
        description="Classifies network flows as benign or a specific attack type.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def _predict(request: Request, rows: list[dict]) -> list[dict]:
        service: InferenceService = request.app.state.service
        try:
            predictions = service.predict(rows)
        except MissingFeaturesError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        request.app.state.store.log(rows, predictions)
        return predictions

    @app.get("/health")
    def health(request: Request):
        service: InferenceService = request.app.state.service
        return {"status": "ok", "model_loaded": True, "model_name": service.model_name}

    @app.get("/model-info")
    def model_info(request: Request):
        service: InferenceService = request.app.state.service
        return {
            "model_name": service.model_name,
            "classes": service.class_names,
            "benign_label": service.benign_label,
            "feature_count": len(service.feature_columns),
            "feature_columns": service.feature_columns,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: FlowRequest, request: Request):
        return _predict(request, [payload.features])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(payload: BatchRequest, request: Request):
        if len(payload.flows) > max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"Batch too large: {len(payload.flows)} flows (limit {max_batch_size}).",
            )
        predictions = _predict(request, payload.flows)
        return {
            "count": len(predictions),
            "attacks_detected": sum(p["is_attack"] for p in predictions),
            "predictions": predictions,
        }

    @app.get("/predictions/recent")
    def recent_predictions(request: Request, limit: int = 50):
        limit = max(1, min(limit, 500))
        return request.app.state.store.recent(limit)

    @app.get("/stats")
    def stats(request: Request):
        service: InferenceService = request.app.state.service
        data = request.app.state.store.stats()
        benign = sum(n for c, n in data["per_class"].items() if c.lower() == service.benign_label.lower())
        data["attacks"] = data["total"] - benign
        return data

    @app.get("/", include_in_schema=False)
    def root():
        # opening http://127.0.0.1:8000 in a browser lands on the interactive docs page
        return RedirectResponse(url="/docs")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        # browsers ask for this automatically; answer quietly instead of logging a 404
        return Response(status_code=204)

    return app


app = create_app()
