"""FaceID face-service — FastAPI application.

Endpoints (all except ``GET /health`` require the ``X-Internal-Api-Key`` header):

* ``POST /extract``     — 1..5 enrollment photos -> embeddings + quality
* ``POST /verify``      — 1:1 verification against provided embeddings
* ``POST /verify-live`` — multi-frame (burst) verification: identity +
  passive anti-spoof ensemble + head-turn/blink challenge
* ``POST /identify``    — 1:N identification via pgvector (read-only DB)
* ``POST /liveness``    — passive anti-spoofing score
* ``GET  /health``      — model / db / liveness status (no auth)
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable

import cv2
import numpy as np

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool

from app import __version__
from app.config import Settings, get_settings
from app.logging import configure_logging, get_logger
from app.schemas import (
    EMBEDDING_DIM,
    AnalyzeRequest,
    AnalyzeResponse,
    ExtractItem,
    ExtractResponse,
    HealthResponse,
    IdentifyRequest,
    IdentifyResponse,
    LivenessRequest,
    LivenessResponse,
    VerifyLiveResponse,
    VerifyRequest,
    VerifyResponse,
)
from app.security import require_api_key
from app.services.burst import (
    CHALLENGE_NONE,
    CHALLENGE_TURN,
    ERROR_LIVENESS_FAILED,
    BurstThresholds,
    FrameObservation,
    decide_burst,
)
from app.services.face import (
    ERROR_FACE_NOT_FOUND,
    ERROR_INVALID_IMAGE,
    DetectedFace,
    FaceEngine,
    blur_variance,
    decode_image_b64,
    decode_image_bytes,
)
from app.services.liveness import LivenessEngine, LivenessEnsemble
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

    # Real-time oqim uchun ALOHIDA yengil engine: faqat detektor + landmark
    # modellari (SCRFD + 2d106 + 3d68), kichik kirish — ArcFace recognition
    # har kadrda ishlamaydi. Yuklanmasa /analyze to'liq engine'da ishlayveradi.
    stream_engine = None
    if settings.analyze_lite_engine:
        stream_engine = FaceEngine(
            model_name=settings.face_model_name,
            model_root=settings.face_model_root,
            det_size=settings.analyze_det_size,
            use_gpu=settings.face_use_gpu,
            allowed_modules=["detection", "landmark_2d_106", "landmark_3d_68"],
        )
        try:
            await run_in_threadpool(stream_engine.load)
        except Exception as exc:  # noqa: BLE001 — degrade to the full engine
            logger.warning("stream_engine_load_failed", error=str(exc))
            stream_engine = None
    app.state.stream_engine = stream_engine

    liveness_models = [
        LivenessEngine(
            model_path=settings.liveness_model_path,
            model_url=settings.liveness_model_url,
            input_size=settings.liveness_input_size,
            bbox_scale=settings.liveness_bbox_scale,
            live_index=settings.liveness_live_index,
            use_gpu=settings.face_use_gpu,
            download_timeout=settings.liveness_download_timeout,
        )
    ]
    # Ikkinchi (print/replay) model — URL bo'sh bo'lsa ansambl bitta modelda qoladi.
    if settings.liveness_model2_url:
        liveness_models.append(
            LivenessEngine(
                model_path=settings.liveness_model2_path,
                model_url=settings.liveness_model2_url,
                input_size=settings.liveness_input_size,
                bbox_scale=settings.liveness_bbox_scale,
                live_index=settings.liveness_model2_live_index,
                use_gpu=settings.face_use_gpu,
                download_timeout=settings.liveness_download_timeout,
            )
        )
    liveness_engine = LivenessEnsemble(liveness_models, settings.liveness_aggregation)
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


def _liveness_engine(request: Request) -> LivenessEnsemble | LivenessEngine | None:
    return getattr(request.app.state, "liveness_engine", None)


async def _liveness_check(
    engine: LivenessEnsemble | LivenessEngine | None,
    img: Any,
    face: DetectedFace,
    enabled: bool,
    settings: Settings,
) -> tuple[float | None, bool]:
    """Compute (liveness_score, liveness_passed) for a detected face.

    Engine mavjud bo'lmasa yoki tekshiruv o'chirilgan bo'lsa score=None
    qaytadi — "tekshirilmadi" degani. Ilgari 1.0 qaytarilardi va bu haqiqiy
    100% skor bilan farqlanmasdi (rasm/ekran jimgina o'tib ketardi);
    backend endi None ni fail-closed (rad) deb talqin qiladi.
    """
    if not enabled or engine is None or not engine.available:
        return None, True
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
    similar_enough = similarity >= threshold
    # XAVFSIZLIK: liveness o'tmagan yuz HECH QACHON match=true bo'lmaydi —
    # rasm/ekran ko'rsatilganda embedding mos kelsa ham autentifikatsiya yo'q.
    response = VerifyResponse(
        match=similar_enough and liveness_passed,
        similarity=round(similarity, 4),
        liveness_score=liveness_score,
        liveness_passed=liveness_passed,
        error=ERROR_LIVENESS_FAILED if similar_enough and not liveness_passed else None,
    )
    logger.info(
        "verify_done",
        match=response.match,
        similarity=response.similarity,
        liveness_passed=liveness_passed,
    )
    return response


def _parse_enrolled_embeddings(raw: str) -> list[list[float]]:
    """Parse & validate the ``embeddings`` form field of ``/verify-live``."""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "VALIDATION_ERROR", "message": "embeddings is not valid JSON"},
        ) from exc
    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(
            isinstance(item, list)
            and len(item) == EMBEDDING_DIM
            and all(isinstance(v, (int, float)) for v in item)
            for item in parsed
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "VALIDATION_ERROR",
                "message": f"embeddings must be a non-empty list of {EMBEDDING_DIM}-float lists",
            },
        )
    return [[float(v) for v in item] for item in parsed]


def _analyze_frame(
    engine: FaceEngine,
    liveness: LivenessEnsemble | LivenessEngine | None,
    data: bytes,
    rotation: int = 0,
) -> FrameObservation:
    """Sync per-frame analysis for the burst: rotate + detect + passive anti-spoof."""
    img = decode_image_bytes(data)
    if img is None:
        return FrameObservation(has_face=False)
    if rotation:
        img = apply_rotation(img, rotation)
    face = engine.best_face(img)
    if face is None:
        return FrameObservation(has_face=False)
    score = 1.0
    if liveness is not None and liveness.available:
        score = liveness.score(img, face.bbox)
    return FrameObservation(
        has_face=True,
        liveness_score=float(score),
        embedding=face.embedding,
        yaw=face.yaw,
        ear=face.ear,
        quality=float(face.det_score),
    )


_ROTATE_FLAGS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def apply_rotation(img: np.ndarray, rotation: int) -> np.ndarray:
    """Kadrni soat mili bo'yicha buradi (0/90/180/270); boshqa qiymat — 0."""
    flag = _ROTATE_FLAGS.get(rotation % 360)
    return cv2.rotate(img, flag) if flag is not None else img


def _analyze_sync(
    engine: FaceEngine,
    liveness: LivenessEnsemble | LivenessEngine | None,
    img: Any,
    check_liveness: bool,
    rotation: int = 0,
    try_rotations: bool = False,
) -> AnalyzeResponse:
    """Real-time pipeline'ning bitta kadr uchun sinxron qismi (threadpool'da):
    rotatsiya → detect → pose/landmarklar → sifat → passiv anti-spoof.

    ``try_rotations`` — sessiya boshidagi BIR MARTALIK kalibrlash: berilgan
    rotatsiyada yuz topilmasa 270/90/180 sinab ko'riladi (skipProcessing bilan
    kelgan sensor-orientatsiyali kadrlar uchun); topilgani javobda qaytadi.
    """
    rotation = rotation % 360
    work = apply_rotation(img, rotation)
    faces = engine.detect(work)

    if not faces and try_rotations:
        for candidate in (270, 90, 180):
            if candidate == rotation:
                continue
            rotated = apply_rotation(img, candidate)
            faces = engine.detect(rotated)
            if faces:
                work = rotated
                rotation = candidate
                break

    height, width = work.shape[:2]
    if not faces:
        return AnalyzeResponse(
            found=False,
            rotation_applied=rotation,
            frame_width=width,
            frame_height=height,
        )
    img = work

    face = faces[0]
    x1, y1, x2, y2 = (float(v) for v in face.bbox[:4])

    # Sifat metrikalari — yuz hududидан (gate: too_dark / xira kadr)
    crop = FaceEngine.face_crop(img, face)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    brightness = float(np.mean(gray))
    sharpness = blur_variance(crop)

    liveness_score: float | None = None
    if check_liveness and liveness is not None and liveness.available:
        liveness_score = round(float(liveness.score(img, face.bbox)), 4)

    landmarks: list[list[float]] | None = None
    if face.landmarks is not None:
        landmarks = [
            [round(float(px) / width, 4), round(float(py) / height, 4)]
            for px, py in face.landmarks
        ]

    return AnalyzeResponse(
        found=True,
        multiple=len(faces) > 1,
        x=max(0.0, x1 / width),
        y=max(0.0, y1 / height),
        width=max(0.0, (x2 - x1) / width),
        height=max(0.0, (y2 - y1) / height),
        yaw=face.yaw,
        pitch=face.pitch,
        roll=face.roll,
        ear=face.ear,
        det_score=round(face.det_score, 4),
        brightness=round(brightness, 2),
        sharpness=round(sharpness, 2),
        liveness_score=liveness_score,
        landmarks=landmarks,
        rotation_applied=rotation,
        frame_width=width,
        frame_height=height,
        error=None,
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(require_api_key)],
)
async def analyze(request: Request, payload: AnalyzeRequest) -> AnalyzeResponse:
    """Bitta kadrni real-time pipeline bo'yicha tahlil qilish (WS oqimi).

    Yuz joyi (normalized bbox), poza (yaw/pitch/roll), 106 landmark, sifat
    (yorqinlik/keskinlik) va passiv anti-spoof skori qaytadi — mobil klient
    yuz kvadrati va mesh'ni shundan chizadi, backend'dagi jonlilik darvozasi
    esa ko'p-signalli dalil yig'adi.
    """
    # Yengil stream engine (det-lite) bo'lsa — shu; aks holda to'liq engine
    stream = getattr(request.app.state, "stream_engine", None)
    engine = stream if (stream is not None and stream.loaded) else _face_engine(request)
    img = decode_image_b64(payload.image_b64)
    if img is None:
        return AnalyzeResponse(found=False, error=ERROR_INVALID_IMAGE)
    return await run_in_threadpool(
        _analyze_sync,
        engine,
        _liveness_engine(request),
        img,
        payload.check_liveness,
        payload.rotation,
        payload.try_rotations,
    )


@app.post(
    "/verify-live",
    response_model=VerifyLiveResponse,
    dependencies=[Depends(require_api_key)],
)
async def verify_live(
    request: Request,
    frames: list[UploadFile] = File(..., description="Tartiblangan burst kadrlari (2..8)"),
    embeddings: str = Form(..., description="JSON: xodimning enrolled embeddinglari"),
    challenge: str = Form(CHALLENGE_TURN, description='"turn" yoki "none"'),
    match_threshold: float | None = Form(None, ge=0.0, le=1.0),
    rotation: int = Form(0, description="Har kadrga qo'llanadigan rotatsiya (gradus)"),
) -> VerifyLiveResponse:
    """Ko'p kadrli verifikatsiya: identity + passiv anti-spoof + challenge.

    Bir odam bir burst ichida bo'lishi, har kadr anti-spoof ansamblidan
    o'tishi va bosh burilishi (yoki blink) kuzatilishi shart — statik rasm,
    ekran yoki almashtirilgan surat bu zanjirdan o'ta olmaydi.
    """
    engine = _face_engine(request)
    settings = get_settings()
    if not 2 <= len(frames) <= settings.burst_max_frames:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "VALIDATION_ERROR",
                "message": f"Between 2 and {settings.burst_max_frames} frames are required",
            },
        )
    enrolled = _parse_enrolled_embeddings(embeddings)
    challenge_mode = challenge if challenge in (CHALLENGE_TURN, CHALLENGE_NONE) else CHALLENGE_TURN

    liveness_engine = _liveness_engine(request)
    # Kadrlar PARALLEL tahlil qilinadi (onnxruntime sessiyalari thread-safe):
    # ketma-ket rejimda umumiy vaqt = kadrlar soni x kadr vaqti edi; endi
    # dekodlash/preprocessing bir-birini kutmaydi va umumiy vaqt ~eng sekin
    # kadr darajasiga tushadi. Bu mobil check-in javobining asosiy qismi.
    frame_bytes = [await upload.read() for upload in frames]
    observations: list[FrameObservation] = list(
        await asyncio.gather(
            *(
                run_in_threadpool(_analyze_frame, engine, liveness_engine, data, rotation)
                for data in frame_bytes
            )
        )
    )

    thresholds = BurstThresholds(
        min_valid_frames=min(settings.burst_min_valid_frames, len(frames)),
        consistency_threshold=settings.burst_consistency_threshold,
        liveness_threshold=settings.liveness_threshold,
        yaw_range_deg=settings.challenge_yaw_range_deg,
        blink_ear_close=settings.blink_ear_close,
        blink_ear_open=settings.blink_ear_open,
        match_threshold=(
            match_threshold if match_threshold is not None else settings.face_match_threshold
        ),
    )
    decision = decide_burst(observations, enrolled, thresholds, challenge_mode)
    logger.info(
        "verify_live_done",
        match=decision.match,
        similarity=decision.similarity,
        liveness_score=decision.liveness_score,
        challenge_passed=decision.challenge_passed,
        consistency=decision.consistency,
        frames_valid=decision.frames_valid,
        frames_total=decision.frames_total,
        error=decision.error,
        reasons=decision.reasons,
    )
    return VerifyLiveResponse(
        match=decision.match,
        similarity=decision.similarity,
        liveness_score=decision.liveness_score,
        liveness_passed=decision.liveness_passed,
        challenge_passed=decision.challenge_passed,
        consistency=decision.consistency,
        frames_total=decision.frames_total,
        frames_valid=decision.frames_valid,
        error=decision.error,
        reasons=decision.reasons,
    )


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
    similar_enough = similarity >= settings.face_match_threshold
    # XAVFSIZLIK: liveness o'tmasa found=false — xodim ID'si ham oshkor qilinmaydi.
    found = similar_enough and liveness_passed
    response = IdentifyResponse(
        found=found,
        employee_id=employee_id if found else None,
        similarity=round(similarity, 4),
        liveness_score=liveness_score,
        liveness_passed=liveness_passed,
        error=ERROR_LIVENESS_FAILED if similar_enough and not liveness_passed else None,
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
