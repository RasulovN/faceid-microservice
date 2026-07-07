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
