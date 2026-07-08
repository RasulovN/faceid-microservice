"""Shared test fixtures.

The heavy ML stack (insightface / onnxruntime / asyncpg) is NOT imported by the
tests: model access happens only inside the FastAPI lifespan, which the tests
never enter. Endpoints are exercised against fakes injected into ``app.state``.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Sequence

import numpy as np
import pytest

TEST_API_KEY = "test-internal-key"

# Must be set before app.config / app.main are imported anywhere.
os.environ["INTERNAL_API_KEY"] = TEST_API_KEY
os.environ["FACE_SERVICE_DATABASE_URL"] = ""

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.face import DetectedFace, EnrollResult  # noqa: E402

get_settings.cache_clear()


def make_embedding(dim: int = 512, seed: int = 1) -> list[float]:
    """Deterministic L2-normalized embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim)
    vec /= np.linalg.norm(vec)
    return [float(x) for x in vec]


class FakeFaceEngine:
    """Stands in for FaceEngine without loading any model.

    ``yaws`` / ``ears`` (per-call sequences) let burst tests simulate head
    turns and blinks: the i-th ``best_face`` call returns the i-th value.
    """

    def __init__(
        self,
        embedding: Sequence[float] | None = None,
        enroll_result: EnrollResult | None = None,
        face_found: bool = True,
        yaws: Sequence[float | None] | None = None,
        ears: Sequence[float | None] | None = None,
    ) -> None:
        self._embedding = np.asarray(
            embedding if embedding is not None else make_embedding(), dtype=np.float32
        )
        self._enroll_result = enroll_result
        self._face_found = face_found
        self._yaws = list(yaws) if yaws is not None else None
        self._ears = list(ears) if ears is not None else None
        self._calls = 0
        self.loaded = True

    def best_face(self, img: Any) -> DetectedFace | None:
        call = self._calls
        self._calls += 1
        if not self._face_found:
            return None
        yaw = self._yaws[call % len(self._yaws)] if self._yaws else None
        ear = self._ears[call % len(self._ears)] if self._ears else None
        return DetectedFace(
            bbox=np.array([10.0, 10.0, 120.0, 130.0], dtype=np.float32),
            det_score=0.95,
            embedding=self._embedding,
            yaw=yaw,
            ear=ear,
        )

    def detect(self, img: Any) -> list[DetectedFace]:
        face = self.best_face(img)
        return [face] if face is not None else []

    def enroll(self, data: bytes, quality_threshold: float) -> EnrollResult:
        if self._enroll_result is not None:
            return self._enroll_result
        return EnrollResult(
            ok=True,
            embedding=[float(x) for x in self._embedding],
            quality=0.87,
            error=None,
        )


class FakeLivenessEngine:
    """Stands in for LivenessEngine."""

    def __init__(self, score: float = 0.9, available: bool = True) -> None:
        self._score = score
        self.available = available

    @property
    def status(self) -> str:
        return "ok" if self.available else "disabled"

    def score(self, img: Any, bbox: Any) -> float:
        return self._score


class FakePool:
    """Stands in for an asyncpg pool (identify + health checks)."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.queries: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.queries.append((sql, args))
        return self.rows

    async def fetchval(self, sql: str, *args: Any) -> int:
        return 1


def _reset_state() -> None:
    for attr in ("face_engine", "stream_engine", "liveness_engine", "db_pool"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


@pytest.fixture()
def client() -> TestClient:
    """TestClient with clean app.state (lifespan intentionally not run)."""
    _reset_state()
    return TestClient(app)


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    return {"X-Internal-Api-Key": TEST_API_KEY}


@pytest.fixture()
def image_b64() -> str:
    """Base64 of a tiny valid PNG (decodable by cv2)."""
    import cv2

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = (10, 120, 240)
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")
