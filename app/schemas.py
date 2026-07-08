"""Pydantic v2 request/response models."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

EMBEDDING_DIM = 512

#: A single ArcFace embedding — exactly 512 floats.
Embedding = Annotated[list[float], Field(min_length=EMBEDDING_DIM, max_length=EMBEDDING_DIM)]


class ExtractItem(BaseModel):
    """Per-image result of ``POST /extract``."""

    ok: bool
    embedding: Embedding | None = None
    quality: float = Field(default=0.0, ge=0.0, le=1.0)
    error: str | None = None


class ExtractResponse(BaseModel):
    results: list[ExtractItem]


class VerifyRequest(BaseModel):
    """1:1 verification request."""

    image_b64: str = Field(min_length=1)
    embeddings: list[Embedding] = Field(min_length=1)
    match_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    check_liveness: bool = True


class VerifyResponse(BaseModel):
    match: bool
    similarity: float = 0.0
    liveness_score: float = 0.0
    liveness_passed: bool = False
    error: str | None = None


class IdentifyRequest(BaseModel):
    """1:N identification request (pgvector search scoped to a company + optional branch)."""

    image_b64: str = Field(min_length=1)
    company_id: UUID
    #: Berilsa — qidiruv shu filial xodimlari bilan cheklanadi (boshqa filial/kompaniya
    #: xodimi nomzod ham bo'lmaydi → "yuz aniqlanmadi").
    branch_id: UUID | None = None
    check_liveness: bool = True


class IdentifyResponse(BaseModel):
    found: bool
    employee_id: str | None = None
    similarity: float = 0.0
    liveness_score: float = 0.0
    liveness_passed: bool = False
    error: str | None = None


class VerifyLiveResponse(BaseModel):
    """Multi-frame (burst) verification: identity + passive liveness + challenge."""

    #: YAKUNIY qaror: identity mos VA liveness o'tdi VA challenge bajarildi.
    match: bool
    similarity: float = 0.0
    liveness_score: float = 0.0
    liveness_passed: bool = False
    challenge_passed: bool = False
    #: Kadrlararo minimal juftlik o'xshashligi (bir odam ekanligi dalili).
    consistency: float = 0.0
    frames_total: int = 0
    frames_valid: int = 0
    #: FACE_NOT_FOUND | LIVENESS_FAILED | CHALLENGE_FAILED | FACE_NOT_RECOGNIZED | None
    error: str | None = None
    #: Diagnostika (masalan IDENTITY_INCONSISTENT, NO_HEAD_TURN) — log/audit uchun.
    reasons: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    """Bitta kadr uchun real-time pipeline tahlili (WS oqimi)."""

    image_b64: str = Field(min_length=1)
    #: Har kadrda passiv anti-spoof skorini ham hisoblash (ansambl, ~30ms).
    check_liveness: bool = True
    #: Kadrga qo'llanadigan rotatsiya (soat mili bo'yicha, gradus).
    #: Mobil `skipProcessing` bilan tez suratga oladi — sensor orientatsiyasi
    #: tuzatilmagan bo'ladi; server shu parametr bilan to'g'irlaydi.
    rotation: int = Field(default=0)
    #: True — 0° da yuz topilmasa 270/90/180 ni sinab ko'radi (sessiya boshida
    #: BIR MARTA kalibrlash; topilgan qiymat `rotation_applied` da qaytadi).
    try_rotations: bool = False


class AnalyzeResponse(BaseModel):
    """Kadr tahlili: detect → pose → landmarklar → sifat → passiv anti-spoof.

    bbox va landmark koordinatalari kadr o'lchamiga nisbatan 0..1 oralig'ida —
    klient ularni ekran o'lchamiga o'zi moslaydi (cover-fit). Embedding
    QAYTMAYDI — identifikatsiya faqat yakuniy /verify-live bosqichida.
    """

    found: bool
    multiple: bool = False
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    ear: float | None = None
    det_score: float = 0.0
    #: Yuz hududining o'rtacha yorqinligi (0..255) — "juda qorong'i" gate.
    brightness: float | None = None
    #: Laplacian dispersiyasi (keskinlik) — xira kadrlarni rad etish uchun.
    sharpness: float | None = None
    #: Passiv anti-spoof ansambl skori (0..1); liveness o'chiq bo'lsa None.
    liveness_score: float | None = None
    #: 106 ta 2D landmark, normalized [[x,y]...] — real-time mesh rendering.
    landmarks: list[list[float]] | None = None
    #: Kadrga amalda qo'llangan rotatsiya (kalibrlash natijasi).
    rotation_applied: int = 0
    frame_width: int = 0
    frame_height: int = 0
    error: str | None = None


class LivenessRequest(BaseModel):
    image_b64: str = Field(min_length=1)


class LivenessResponse(BaseModel):
    liveness_score: float = 0.0
    passed: bool = False
    error: str | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    db: str
    liveness: str
