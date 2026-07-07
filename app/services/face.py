"""InsightFace wrapper: detection (RetinaFace), embedding (ArcFace) and quality.

The heavy ``insightface`` / ``onnxruntime`` imports are deliberately deferred to
:meth:`FaceEngine.load` (called from the FastAPI lifespan) so that unit tests can
import this module with only ``numpy`` and ``opencv-python-headless`` installed.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.logging import get_logger

logger = get_logger(__name__)

EMBEDDING_DIM = 512

#: Face side (px) at which the size component of the quality score saturates.
#: 112 px is the native ArcFace input resolution.
MIN_FACE_SIDE_PX = 112.0
#: Laplacian variance at which the sharpness component saturates.
BLUR_NORM = 100.0
#: Weights for (detection score, face size, sharpness) — must sum to 1.0.
QUALITY_WEIGHTS = (0.5, 0.25, 0.25)
#: Side of the normalized crop used for the blur measurement.
_BLUR_CROP_SIDE = 112

ERROR_INVALID_IMAGE = "INVALID_IMAGE"
ERROR_FACE_NOT_FOUND = "FACE_NOT_FOUND"
ERROR_FACE_MULTIPLE = "FACE_MULTIPLE"
ERROR_FACE_LOW_QUALITY = "FACE_LOW_QUALITY"


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    """Decode raw image bytes (jpeg/png/webp/...) into a BGR ndarray."""
    if not data:
        return None
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return img


def decode_image_b64(data: str) -> np.ndarray | None:
    """Decode a base64 string (optionally a ``data:image/...;base64,`` URL)."""
    if not data:
        return None
    payload = data.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload)
    except (ValueError, binascii.Error):
        return None
    return decode_image_bytes(raw)


def blur_variance(img: np.ndarray) -> float:
    """Variance of the Laplacian on a normalized 112x112 grayscale crop.

    Resizing to a fixed side makes the value comparable across image sizes.
    Higher values mean sharper images; values below ~100 usually mean blur.
    """
    if img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if gray.shape[0] != _BLUR_CROP_SIDE or gray.shape[1] != _BLUR_CROP_SIDE:
        gray = cv2.resize(gray, (_BLUR_CROP_SIDE, _BLUR_CROP_SIDE))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_quality(
    det_score: float,
    face_width: float,
    face_height: float,
    blur_var: float,
) -> float:
    """Combine detector confidence, face size and sharpness into a 0..1 score.

    quality = 0.5 * det_score + 0.25 * size_score + 0.25 * sharpness_score
    where size saturates at 112 px (min side) and sharpness at Laplacian
    variance 100. The result is clipped to [0, 1] and rounded to 4 decimals.
    """
    det = float(np.clip(det_score, 0.0, 1.0))
    size = float(np.clip(min(face_width, face_height) / MIN_FACE_SIDE_PX, 0.0, 1.0))
    sharpness = float(np.clip(blur_var / BLUR_NORM, 0.0, 1.0))
    w_det, w_size, w_sharp = QUALITY_WEIGHTS
    quality = w_det * det + w_size * size + w_sharp * sharpness
    return round(float(np.clip(quality, 0.0, 1.0)), 4)


@dataclass(slots=True)
class DetectedFace:
    """A single detected & aligned face."""

    bbox: np.ndarray  # [x1, y1, x2, y2], float32
    det_score: float
    embedding: np.ndarray  # 512-dim, L2-normalized (ArcFace)

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


@dataclass(slots=True)
class EnrollResult:
    """Result of processing one enrollment photo (``POST /extract``)."""

    ok: bool
    embedding: list[float] | None
    quality: float
    error: str | None


class FaceEngine:
    """Thin wrapper around ``insightface.app.FaceAnalysis`` (buffalo_l pack)."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        model_root: str = "~/.insightface",
        det_size: int = 640,
        use_gpu: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_root = model_root
        self.det_size = det_size
        self.use_gpu = use_gpu
        self._app: Any = None

    @property
    def loaded(self) -> bool:
        return self._app is not None

    def load(self) -> None:
        """Load RetinaFace + ArcFace models. Heavy — call once at startup."""
        from insightface.app import FaceAnalysis  # deferred heavy import

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )
        analyzer = FaceAnalysis(
            name=self.model_name,
            root=self.model_root,
            providers=providers,
        )
        analyzer.prepare(
            ctx_id=0 if self.use_gpu else -1,
            det_size=(self.det_size, self.det_size),
        )
        self._app = analyzer
        logger.info(
            "face_model_loaded",
            model=self.model_name,
            det_size=self.det_size,
            providers=providers,
        )

    def detect(self, img_bgr: np.ndarray) -> list[DetectedFace]:
        """Detect all faces, largest first. Requires :meth:`load` beforehand."""
        if self._app is None:
            raise RuntimeError("Face model is not loaded; call FaceEngine.load() first")
        faces = self._app.get(img_bgr)
        detected = [
            DetectedFace(
                bbox=np.asarray(face.bbox, dtype=np.float32),
                det_score=float(face.det_score),
                embedding=np.asarray(face.normed_embedding, dtype=np.float32),
            )
            for face in faces
        ]
        detected.sort(key=lambda f: f.area, reverse=True)
        return detected

    def best_face(self, img_bgr: np.ndarray) -> DetectedFace | None:
        """Largest detected face (used by verify/identify/liveness) or ``None``."""
        faces = self.detect(img_bgr)
        return faces[0] if faces else None

    @staticmethod
    def face_crop(img_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Clamped bbox crop of the face region (for the blur measurement)."""
        height, width = img_bgr.shape[:2]
        x1 = int(np.clip(face.bbox[0], 0, width - 1))
        y1 = int(np.clip(face.bbox[1], 0, height - 1))
        x2 = int(np.clip(face.bbox[2], x1 + 1, width))
        y2 = int(np.clip(face.bbox[3], y1 + 1, height))
        crop = img_bgr[y1:y2, x1:x2]
        return crop if crop.size else img_bgr

    def enroll(self, data: bytes, quality_threshold: float) -> EnrollResult:
        """Process one enrollment photo: strict single-face + quality gate."""
        img = decode_image_bytes(data)
        if img is None:
            return EnrollResult(ok=False, embedding=None, quality=0.0, error=ERROR_INVALID_IMAGE)
        faces = self.detect(img)
        if not faces:
            return EnrollResult(ok=False, embedding=None, quality=0.0, error=ERROR_FACE_NOT_FOUND)
        if len(faces) > 1:
            return EnrollResult(ok=False, embedding=None, quality=0.0, error=ERROR_FACE_MULTIPLE)
        face = faces[0]
        quality = compute_quality(
            det_score=face.det_score,
            face_width=face.width,
            face_height=face.height,
            blur_var=blur_variance(self.face_crop(img, face)),
        )
        if quality < quality_threshold:
            return EnrollResult(
                ok=False, embedding=None, quality=quality, error=ERROR_FACE_LOW_QUALITY
            )
        return EnrollResult(
            ok=True,
            embedding=[float(x) for x in face.embedding],
            quality=quality,
            error=None,
        )
