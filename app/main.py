"""FaceID face-service — FastAPI application.

Endpoints (all except ``GET /health`` require the ``X-Internal-Api-Key`` header):

* ``POST /extract``  — 1..5 enrollment photos -> embeddings + quality
* ``POST /verify``   — 1:1 verification against provided embeddings
* ``POST /identify`` — 1:N identification via pgvector (read-only DB)
* ``POST /liveness`` — passive anti-spoofing score
* ``GET  /health``   — model / db / liveness status (no auth)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app import __version__
from app.config import Settings, get_settings
from app.logging import configure_logging, get_logger
from app.schemas import (
    ExtractItem,
    ExtractResponse,
    HealthResponse,
    IdentifyRequest,
    IdentifyResponse,
    LivenessRequest,
    LivenessResponse,
    VerifyRequest,
    VerifyResponse,
)
from app.security import require_api_key
from app.services.face import (
    ERROR_FACE_NOT_FOUND,
    ERROR_INVALID_IMAGE,
    DetectedFace,
    FaceEngine,
    decode_image_b64,
)
from app.services.liveness import LivenessEngine
from app.services.matcher import best_similarity, identify_top

logger = get_logger(__name__)

MAX_IMAGES_PER_EXTRACT = 5


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models and create the read-only DB pool at startup."""
    settings = get_settings()
    configure_logging(settings.log_level)

    if not settings.internal_api_key:
        logger.warning(
            "internal_api_key_missing",
            hint="INTERNAL_API_KEY is empty — every authenticated endpoint will return 401",
        )

    face_engine = FaceEngine(
        model_name=settings.face_model_name,
        model_root=settings.face_model_root,
        det_size=settings.face_det_size,
        use_gpu=settings.face_use_gpu,
    )
    try:
        await run_in_threadpool(face_engine.load)
    except Exception as exc:  # noqa: BLE001 — keep serving /health with degraded state
        logger.error("face_model_load_failed", error=str(exc))
    app.state.face_engine = face_engine

    liveness_engine = LivenessEngine(
        model_path=settings.liveness_model_path,
        model_url=settings.liveness_model_url,
        input_size=settings.liveness_input_size,
        bbox_scale=settings.liveness_bbox_scale,
        live_index=settings.liveness_live_index,
        use_gpu=settings.face_use_gpu,
        download_timeout=settings.liveness_download_timeout,
    )
    await run_in_threadpool(liveness_engine.load)
    app.state.liveness_engine = liveness_engine

    pool = None
    if settings.face_service_database_url:
        try:
            import asyncpg  # deferred so unit tests do not need asyncpg installed

            pool = await asyncpg.create_pool(
                dsn=settings.face_service_database_url,
                min_size=settings.db_pool_min_size,
                max_size=settings.db_pool_max_size,
                server_settings={"default_transaction_read_only": "on"},
            )
            logger.info("db_pool_created", max_size=settings.db_pool_max_size)
        except Exception as exc:  # noqa: BLE001 — asyncpg yo'q yoki DB uzilgan → /identify 503
            logger.error("db_pool_create_failed", error=str(exc))
    else:
        logger.warning(
            "db_url_missing",
            hint="FACE_SERVICE_DATABASE_URL is empty — /identify will return 503",
        )
    app.state.db_pool = pool

    logger.info(
        "service_started",
        version=__version__,
        model_loaded=face_engine.loaded,
        liveness=liveness_engine.status,
        db="ok" if pool is not None else "error",
    )
    yield

    if pool is not None:
        await pool.close()
    logger.info("service_stopped")


app = FastAPI(
    title="FaceID Face Service",
    description="Internal face recognition microservice (InsightFace buffalo_l + MiniFASNet liveness).",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def access_log_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Any]],
) -> Any:
    """JSON access log for every request."""
    started = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
    )
    return response


def _face_engine(request: Request) -> FaceEngine:
    """Return the loaded face engine or fail with 503."""
    engine = getattr(request.app.state, "face_engine", None)
    if engine is None or not engine.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "MODEL_NOT_LOADED", "message": "Face model is not loaded yet"},
        )
    return engine


def _liveness_engine(request: Request) -> LivenessEngine | None:
    return getattr(request.app.state, "liveness_engine", None)


async def _liveness_check(
    engine: LivenessEngine | None,
    img: Any,
    face: DetectedFace,
    enabled: bool,
    settings: Settings,
) -> tuple[float, bool]:
    """Compute (liveness_score, liveness_passed) for a detected face."""
    if not enabled or engine is None or not engine.available:
        return 1.0, True
    score = await run_in_threadpool(engine.score, img, face.bbox)
    return round(score, 4), score >= settings.liveness_threshold


@app.post(
    "/extract",
    response_model=ExtractResponse,
    dependencies=[Depends(require_api_key)],
)
async def extract(
    request: Request,
    images: list[UploadFile] = File(..., description="1..5 enrollment photos"),
) -> ExtractResponse:
    """Extract one ArcFace embedding + quality score per enrollment photo."""
    if not 1 <= len(images) <= MAX_IMAGES_PER_EXTRACT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "VALIDATION_ERROR",
                "message": f"Between 1 and {MAX_IMAGES_PER_EXTRACT} images are required",
            },
        )
    engine = _face_engine(request)
    settings = get_settings()
    results: list[ExtractItem] = []
    for upload in images:
        data = await upload.read()
        outcome = await run_in_threadpool(engine.enroll, data, settings.face_quality_threshold)
        results.append(
            ExtractItem(
                ok=outcome.ok,
                embedding=outcome.embedding,
                quality=outcome.quality,
                error=outcome.error,
            )
        )
    logger.info(
        "extract_done",
        images=len(results),
        accepted=sum(1 for item in results if item.ok),
    )
    return ExtractResponse(results=results)


@app.post(
    "/verify",
    response_model=VerifyResponse,
    dependencies=[Depends(require_api_key)],
)
async def verify(request: Request, payload: VerifyRequest) -> VerifyResponse:
    """1:1 verification: image vs provided candidate embeddings."""
    engine = _face_engine(request)
    settings = get_settings()

    img = decode_image_b64(payload.image_b64)
    if img is None:
        return VerifyResponse(match=False, error=ERROR_INVALID_IMAGE)

    face = await run_in_threadpool(engine.best_face, img)
    if face is None:
        return VerifyResponse(match=False, error=ERROR_FACE_NOT_FOUND)

    liveness_score, liveness_passed = await _liveness_check(
        _liveness_engine(request), img, face, payload.check_liveness, settings
    )
    threshold = (
        payload.match_threshold
        if payload.match_threshold is not None
        else settings.face_match_threshold
    )
    similarity = best_similarity(face.embedding, payload.embeddings)
    response = VerifyResponse(
        match=similarity >= threshold,
        similarity=round(similarity, 4),
        liveness_score=liveness_score,
        liveness_passed=liveness_passed,
        error=None,
    )
    logger.info(
        "verify_done",
        match=response.match,
        similarity=response.similarity,
        liveness_passed=liveness_passed,
    )
    return response


@app.post(
    "/identify",
    response_model=IdentifyResponse,
    dependencies=[Depends(require_api_key)],
)
async def identify(request: Request, payload: IdentifyRequest) -> IdentifyResponse:
    """1:N identification within one company via pgvector (read-only)."""
    engine = _face_engine(request)
    settings = get_settings()
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "DB_UNAVAILABLE", "message": "Database pool is not available"},
        )

    img = decode_image_b64(payload.image_b64)
    if img is None:
        return IdentifyResponse(found=False, error=ERROR_INVALID_IMAGE)

    face = await run_in_threadpool(engine.best_face, img)
    if face is None:
        return IdentifyResponse(found=False, error=ERROR_FACE_NOT_FOUND)

    liveness_score, liveness_passed = await _liveness_check(
        _liveness_engine(request), img, face, payload.check_liveness, settings
    )
    matches = await identify_top(
        pool,
        face.embedding,
        str(payload.company_id),
        limit=5,
        branch_id=str(payload.branch_id) if payload.branch_id else None,
    )
    if not matches:
        return IdentifyResponse(
            found=False,
            liveness_score=liveness_score,
            liveness_passed=liveness_passed,
            error=None,
        )

    employee_id, similarity = matches[0]
    found = similarity >= settings.face_match_threshold
    response = IdentifyResponse(
        found=found,
        employee_id=employee_id if found else None,
        similarity=round(similarity, 4),
        liveness_score=liveness_score,
        liveness_passed=liveness_passed,
        error=None,
    )
    logger.info(
        "identify_done",
        found=found,
        similarity=response.similarity,
        liveness_passed=liveness_passed,
    )
    return response


@app.post(
    "/liveness",
    response_model=LivenessResponse,
    dependencies=[Depends(require_api_key)],
)
async def liveness(request: Request, payload: LivenessRequest) -> LivenessResponse:
    """Passive anti-spoofing score for the largest face in the image."""
    engine = _face_engine(request)
    settings = get_settings()

    img = decode_image_b64(payload.image_b64)
    if img is None:
        return LivenessResponse(liveness_score=0.0, passed=False, error=ERROR_INVALID_IMAGE)

    face = await run_in_threadpool(engine.best_face, img)
    if face is None:
        return LivenessResponse(liveness_score=0.0, passed=False, error=ERROR_FACE_NOT_FOUND)

    score, passed = await _liveness_check(
        _liveness_engine(request), img, face, enabled=True, settings=settings
    )
    return LivenessResponse(liveness_score=score, passed=passed, error=None)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness/readiness probe — no API key required."""
    engine = getattr(request.app.state, "face_engine", None)
    liveness_engine = getattr(request.app.state, "liveness_engine", None)
    pool = getattr(request.app.state, "db_pool", None)

    model_loaded = bool(engine is not None and engine.loaded)
    db_status = "error"
    if pool is not None:
        try:
            await pool.fetchval("SELECT 1")
            db_status = "ok"
        except Exception as exc:  # noqa: BLE001 — health must never raise
            logger.error("health_db_check_failed", error=str(exc))
    liveness_status = liveness_engine.status if liveness_engine is not None else "disabled"

    return HealthResponse(
        status="ok" if (model_loaded and db_status == "ok") else "degraded",
        model_loaded=model_loaded,
        db=db_status,
        liveness=liveness_status,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=get_settings().face_service_port,
    )
