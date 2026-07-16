"""Application configuration.

All values are read from environment variables (or a local ``.env`` file).
Variable names follow the repository-wide ``.env.example`` contract:
``INTERNAL_API_KEY``, ``FACE_MATCH_THRESHOLD``, ``LIVENESS_THRESHOLD``,
``FACE_SERVICE_DATABASE_URL``, ``FACE_USE_GPU``, ``FACE_SERVICE_PORT``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Ready-made ONNX conversion of MiniFASNetV2 (Silent-Face-Anti-Spoofing).
#: Binary live/spoof model, 128x128 input, bbox expanded by 1.5x, live class index 0.
DEFAULT_LIVENESS_MODEL_URL = (
    "https://github.com/hairymax/Face-AntiSpoofing/raw/main/"
    "saved_models/AntiSpoofing_bin_1.5_128.onnx"
)

#: Second ensemble member: the print/replay-specialised MiniFASNet conversion
#: from the same repository (classes: [live, print-attack, replay-attack]).
#: Catching screen-replay attacks is its training objective, which the binary
#: model alone is weak at. Set LIVENESS_MODEL2_URL="" to disable the ensemble.
DEFAULT_LIVENESS_MODEL2_URL = (
    "https://github.com/hairymax/Face-AntiSpoofing/raw/main/"
    "saved_models/AntiSpoofing_print-replay_1.5_128.onnx"
)


class Settings(BaseSettings):
    """Runtime settings (pydantic-settings, env-driven)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Security -----------------------------------------------------------
    #: Shared secret between the Node backend and this service
    #: (header ``X-Internal-Api-Key``). Empty value => every request is rejected.
    internal_api_key: str = ""

    # --- Thresholds ----------------------------------------------------------
    face_match_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    liveness_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    face_quality_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    #: True bo'lsa — liveness ansambl mavjud bo'lmasa (modellar yuklanmagan)
    #: /verify, /identify, /verify-live, /liveness FAIL-CLOSED bo'ladi (rad).
    #: Default False — hujjatlashtirilgan "disabled" fallback saqlanadi; prod'da
    #: True qo'yish tavsiya etiladi (spoof modellar tushmay qolsa ochiq qolmasin).
    liveness_required: bool = False

    # --- Runtime -------------------------------------------------------------
    face_use_gpu: bool = False
    face_service_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"

    # --- Database (read-only pgvector search) --------------------------------
    face_service_database_url: str = ""
    db_pool_min_size: int = Field(default=1, ge=0)
    db_pool_max_size: int = Field(default=5, ge=1)

    # --- InsightFace ----------------------------------------------------------
    face_model_name: str = "buffalo_l"
    face_model_root: str = "~/.insightface"
    face_det_size: int = Field(default=640, ge=160)
    #: Real-time /analyze uchun ALOHIDA yengil engine: kichikroq detektor
    #: kirishi + faqat detection/landmark modullari (ArcFace har kadrda YO'Q).
    #: 320 — yaqin masofadagi bitta yuz uchun yetarli va ~2x tezroq.
    analyze_det_size: int = Field(default=320, ge=160)
    analyze_lite_engine: bool = True

    # --- Liveness (MiniFASNet ONNX ansambl) -----------------------------------
    liveness_model_url: str = DEFAULT_LIVENESS_MODEL_URL
    liveness_model_path: str = "models/AntiSpoofing_bin_1.5_128.onnx"
    liveness_input_size: int = Field(default=128, ge=32)
    liveness_bbox_scale: float = Field(default=1.5, gt=0.0)
    liveness_live_index: int = Field(default=0, ge=0)
    liveness_download_timeout: float = Field(default=60.0, gt=0.0)
    #: Ikkinchi (print/replay) model — bo'sh URL ansamblni o'chiradi.
    liveness_model2_url: str = DEFAULT_LIVENESS_MODEL2_URL
    liveness_model2_path: str = "models/AntiSpoofing_print-replay_1.5_128.onnx"
    liveness_model2_live_index: int = Field(default=0, ge=0)
    #: Ansambl skorini birlashtirish: "min" (qattiqroq, default) yoki "mean".
    liveness_aggregation: str = "min"

    # --- Burst (ko'p kadrli /verify-live) --------------------------------------
    #: Kamida shuncha kadrda yuz topilishi shart, aks holda FACE_NOT_FOUND.
    burst_min_valid_frames: int = Field(default=3, ge=1)
    burst_max_frames: int = Field(default=8, ge=2)
    #: Kadrlararo minimal juftlik o'xshashligi (bir odam ekanligi) chegarasi.
    burst_consistency_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    #: "turn" challenge: burst davomida yaw (bosh burilishi) diapazoni, gradus.
    challenge_yaw_range_deg: float = Field(default=12.0, ge=0.0)
    #: Blink (EAR) chegaralari — yopiq/ochiq ko'z; blink challenge'ga muqobil dalil.
    #: Mobil klient blink PAYTIDA suratga oladi, lekin takePhoto kechikishi
    #: tufayli ko'z to'liq yopiq bo'lmasligi mumkin — chegaralar shunga tolerantli.
    blink_ear_close: float = Field(default=0.20, gt=0.0)
    blink_ear_open: float = Field(default=0.26, gt=0.0)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (call ``get_settings.cache_clear()`` in tests)."""
    return Settings()
