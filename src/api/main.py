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
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, RedirectResponse, Response

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
        docs_url=None,  # we serve a custom dark-themed docs page below
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

    @app.get("/docs", include_in_schema=False)
    def custom_docs():
        """Serve Swagger UI with a dark theme matching the dashboard."""
        html = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} — Docs",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
            swagger_ui_parameters={"deepLinking": True, "defaultModelsExpandDepth": 1},
        )
        # Inject our dark-theme CSS right before </head>
        patched = html.body.decode().replace("</head>", _SWAGGER_DARK_HEAD + "</head>")
        return HTMLResponse(content=patched)

    @app.get("/", include_in_schema=False)
    def root():
        # opening http://127.0.0.1:8000 in a browser lands on the interactive docs page
        return RedirectResponse(url="/docs")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        # browsers ask for this automatically; answer quietly instead of logging a 404
        return Response(status_code=204)

    return app


_SWAGGER_DARK_HEAD = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* ── Base ── */
  html, body { background-color: #09090B !important; }
  body { font-family: 'Inter', sans-serif !important; }
  .swagger-ui {
    font-family: 'Inter', sans-serif !important;
    color: #D4D4D8;
  }
  .swagger-ui .wrapper { padding: 0 20px; }

  /* ── Top bar ── */
  .swagger-ui .topbar {
    background-color: #0F0F12 !important;
    border-bottom: 1px solid #27272A;
    padding: 10px 0;
  }
  .swagger-ui .topbar .download-url-wrapper .select-label span { color: #A1A1AA !important; }
  .swagger-ui .topbar input[type=text] {
    background: #18181B !important; color: #FAFAFA !important;
    border: 1px solid #27272A !important; border-radius: 4px;
  }

  /* ── Info header ── */
  .swagger-ui .info { margin: 30px 0 20px; }
  .swagger-ui .info .title { color: #FAFAFA !important; font-weight: 700; }
  .swagger-ui .info .title a.link { display: none !important; }
  .swagger-ui .info a.link { display: none !important; }
  .swagger-ui .info .title small {
    background-color: #3F3F46 !important;
    border-radius: 4px; padding: 2px 8px;
  }
  .swagger-ui .info .title small pre {
    background: transparent !important;
    color: #FAFAFA !important;
    border: none !important;
  }
  .swagger-ui .info .title small.version-stamp {
    background-color: #D946EF !important; color: #FFF !important;
    border-radius: 4px; font-weight: 600; padding: 2px 8px;
  }
  .swagger-ui .info .title small.version-stamp pre {
    color: #FFF !important;
  }
  .swagger-ui .info .base-url { color: #71717A !important; }
  .swagger-ui .info .description p,
  .swagger-ui .info li { color: #A1A1AA !important; }
  .swagger-ui .info a { color: #60A5FA !important; }

  /* ── Section tag (e.g. "default") ── */
  .swagger-ui .opblock-tag {
    color: #FAFAFA !important;
    border-bottom: 1px solid #27272A !important;
    font-weight: 600;
  }
  .swagger-ui .opblock-tag:hover { color: #D946EF !important; }
  .swagger-ui .opblock-tag small { color: #71717A !important; }
  .swagger-ui svg.arrow { fill: #71717A !important; }

  /* ── All endpoint rows — clean dark card style ── */
  .swagger-ui .opblock {
    background: #18181B !important;
    border: 1px solid #27272A !important;
    border-radius: 8px !important;
    margin-bottom: 8px;
    box-shadow: none !important;
  }
  .swagger-ui .opblock:hover {
    border-color: #3F3F46 !important;
  }
  .swagger-ui .opblock .opblock-summary {
    border: none !important;
    padding: 8px 12px;
  }

  /* ── Method badges — only these carry color ── */
  .swagger-ui .opblock-summary-method {
    font-weight: 700 !important;
    border-radius: 4px !important;
    padding: 4px 12px !important;
    font-size: 0.75rem !important;
    min-width: 60px;
    text-align: center;
  }
  .swagger-ui .opblock-get .opblock-summary-method {
    background: #3B82F6 !important; color: #FFF !important;
  }
  .swagger-ui .opblock-post .opblock-summary-method {
    background: #D946EF !important; color: #FFF !important;
  }
  .swagger-ui .opblock-put .opblock-summary-method {
    background: #F59E0B !important; color: #000 !important;
  }
  .swagger-ui .opblock-delete .opblock-summary-method {
    background: #EF4444 !important; color: #FFF !important;
  }
  .swagger-ui .opblock-patch .opblock-summary-method {
    background: #06B6D4 !important; color: #FFF !important;
  }

  /* ── Path & description text ── */
  .swagger-ui .opblock-summary-path { color: #FAFAFA !important; font-weight: 500; }
  .swagger-ui .opblock-summary-path__deprecated { color: #71717A !important; }
  .swagger-ui .opblock-summary-description { color: #A1A1AA !important; }

  /* ── Expanded operation body ── */
  .swagger-ui .opblock-body {
    background: #111113 !important;
    border-top: 1px solid #27272A;
  }
  .swagger-ui .opblock .opblock-section-header {
    background: #18181B !important;
    border-bottom: 1px solid #27272A;
    box-shadow: none !important;
  }
  .swagger-ui .opblock .opblock-section-header h4 { color: #E4E4E7 !important; }
  .swagger-ui .opblock .opblock-section-header label { color: #A1A1AA !important; }
  .swagger-ui .opblock-description-wrapper p { color: #A1A1AA !important; }

  /* ── Parameters table ── */
  .swagger-ui table thead tr th,
  .swagger-ui table thead tr td {
    color: #A1A1AA !important;
    border-bottom: 1px solid #27272A !important;
  }
  .swagger-ui .parameter__name { color: #FAFAFA !important; }
  .swagger-ui .parameter__name.required::after { color: #EF4444 !important; }
  .swagger-ui .parameter__type { color: #06B6D4 !important; }
  .swagger-ui .parameter__in { color: #71717A !important; }
  .swagger-ui .parameters-col_description p { color: #A1A1AA !important; }

  /* ── Response section ── */
  .swagger-ui .responses-inner { background: transparent !important; }
  .swagger-ui .response-col_status { color: #22C55E !important; font-weight: 600; }
  .swagger-ui .response-col_description { color: #A1A1AA !important; }
  .swagger-ui .responses-table thead td { color: #A1A1AA !important; }
  .swagger-ui .response-col_links { color: #60A5FA !important; }

  /* ── Code blocks / JSON ── */
  .swagger-ui .highlight-code,
  .swagger-ui .example,
  .swagger-ui .body-param__text {
    background: #0F0F12 !important;
    border: 1px solid #27272A !important;
    border-radius: 6px;
    color: #D4D4D8 !important;
  }
  .swagger-ui pre { background: #0F0F12 !important; color: #D4D4D8 !important; }
  .swagger-ui .microlight { background: #0F0F12 !important; color: #D4D4D8 !important; }

  /* ── Models / Schemas section ── */
  .swagger-ui section.models {
    border: 1px solid #27272A !important;
    border-radius: 8px;
    background: #0F0F12 !important;
  }
  .swagger-ui section.models h4 { color: #FAFAFA !important; }
  .swagger-ui section.models.is-open h4 { border-bottom: 1px solid #27272A; padding-bottom: 10px; }

  /* Model containers */
  .swagger-ui section.models .model-container {
    background: #18181B !important;
    border: 1px solid #27272A !important;
    border-radius: 6px;
    margin: 8px 0;
    padding: 12px 16px;
  }

  /* Model title (BatchRequest, BatchResponse, etc.) — remove ugly border box */
  .swagger-ui .model-title,
  .swagger-ui .model-title__text,
  .swagger-ui span.model-title {
    color: #FAFAFA !important;
    font-weight: 600;
    border: none !important;
    background: transparent !important;
  }

  /* Model body text */
  .swagger-ui .model { color: #D4D4D8 !important; }
  .swagger-ui .model .property { color: #D4D4D8 !important; background: transparent !important; }
  .swagger-ui .model-box { background: transparent !important; border: none !important; }

  /* Property types */
  .swagger-ui .prop-type { color: #06B6D4 !important; }
  .swagger-ui .prop-format { color: #71717A !important; }

  /* Property names — remove bordered boxes */
  .swagger-ui .model .property.primitive {
    color: #D4D4D8 !important;
  }
  
  .swagger-ui table.model tbody tr td {
      background: transparent !important;
  }

  /* Required asterisk */
  .swagger-ui .model span.star { color: #EF4444 !important; }

  /* Toggle arrows and buttons in schemas */
  .swagger-ui span.model-toggle,
  .swagger-ui span.model-toggle:hover,
  .swagger-ui span.model-toggle:focus {
    cursor: pointer;
    background: transparent !important;
    border: none !important;
  }
  .swagger-ui .model-toggle::after { color: #71717A !important; }

  /* Expand/Collapse all buttons — clean flat style */
  .swagger-ui button.model-box-control,
  .swagger-ui button.model-box-control:focus,
  .swagger-ui button.model-box-control:hover,
  .swagger-ui button.model-box-control:active,
  .swagger-ui span.model-toggle-text,
  .swagger-ui span.model-toggle-text:hover,
  .swagger-ui .model-collapse-toggle,
  .swagger-ui .model-collapse-toggle:hover {
    background: transparent !important;
    border: none !important;
    color: #A1A1AA !important;
    cursor: pointer;
    box-shadow: none !important;
    outline: none !important;
  }
  .swagger-ui button.model-box-control:hover { color: #D946EF !important; }

  /* Constraint badges (e.g. "2 1 items") */
  .swagger-ui .model .property .model-hint,
  .swagger-ui span.model-hint {
    background: #27272A !important;
    color: #A1A1AA !important;
    border: none !important;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.7rem;
  }

  /* Inner toggle buttons (Collapse all / Expand all text links) */
  .swagger-ui .model-collapse-toggle,
  .swagger-ui .model-toggle-text {
    color: #71717A !important;
    background: transparent !important;
    border: none !important;
  }
  
  /* OVERRIDE ALL WHITE SPANS IN MODELS */
  .swagger-ui section.models span, 
  .swagger-ui section.models button {
      background-color: transparent !important;
  }
  .swagger-ui section.models .model-title { background: transparent !important; }
  .swagger-ui section.models span.model-title { background: transparent !important; }
  .swagger-ui section.models .model-title__text { background: transparent !important; }

  /* Nested object containers */
  .swagger-ui .model .inner-object {
    border-left: 2px solid #27272A;
    margin-left: 8px;
    padding-left: 12px;
  }

  /* Description text under properties */
  .swagger-ui .model .property .markdown p,
  .swagger-ui .model-description {
    color: #71717A !important;
    font-style: italic;
  }

  /* "Additional properties" text */
  .swagger-ui .model .additional-properties { color: #71717A !important; }

  /* Remove any remaining ugly outlines/borders on interactive schema elements */
  .swagger-ui .model-box-control:focus,
  .swagger-ui .model-toggle:focus,
  .swagger-ui .models-control:focus {
    outline: none !important;
    box-shadow: none !important;
  }

  /* ── Buttons ── */
  .swagger-ui .btn {
    border-radius: 6px; font-weight: 600;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.15s ease;
  }
  .swagger-ui .btn.execute {
    background: #D946EF !important; color: #FFF !important;
    border-color: #D946EF !important;
  }
  .swagger-ui .btn.execute:hover {
    background: #E879F9 !important; border-color: #E879F9 !important;
  }
  .swagger-ui .btn.cancel {
    border-color: #EF4444 !important; color: #EF4444 !important;
    background: transparent !important;
  }
  .swagger-ui .btn.authorize {
    border-color: #22C55E !important; color: #22C55E !important;
  }
  .swagger-ui .btn.authorize svg { fill: #22C55E !important; }
  .swagger-ui .try-out__btn {
    border-color: #3B82F6 !important; color: #3B82F6 !important;
  }
  .swagger-ui .try-out__btn:hover {
    background: rgba(59, 130, 246, 0.1) !important;
  }

  /* ── Inputs ── */
  .swagger-ui input[type=text],
  .swagger-ui textarea,
  .swagger-ui select {
    background: #18181B !important;
    color: #FAFAFA !important;
    border: 1px solid #27272A !important;
    border-radius: 4px;
    font-family: 'Inter', sans-serif !important;
  }
  .swagger-ui input[type=text]:focus,
  .swagger-ui textarea:focus {
    border-color: #D946EF !important; outline: none;
  }

  /* ── Misc ── */
  .swagger-ui .responses-wrapper .responses-inner h4 { color: #E4E4E7 !important; }
  .swagger-ui button.model-box-control { background: transparent !important; }
  .swagger-ui .loading-container .loading::after { color: #D946EF !important; }
  .swagger-ui .scheme-container {
    background: #0F0F12 !important;
    border: 1px solid #27272A;
    border-radius: 6px;
    padding: 12px;
    box-shadow: none !important;
  }
  .swagger-ui .scheme-container .schemes > label { color: #A1A1AA !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #09090B; }
  ::-webkit-scrollbar-thumb { background: #27272A; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #D946EF; }
</style>
"""

app = create_app()
