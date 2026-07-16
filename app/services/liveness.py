"""Passive anti-spoofing (liveness) via a MiniFASNet ONNX ensemble.

Primary path: two Silent-Face-Anti-Spoofing conversions (binary live/spoof +
print/replay-specialised, see ``DEFAULT_LIVENESS_MODEL_URL`` /
``DEFAULT_LIVENESS_MODEL2_URL`` in :mod:`app.config`) are downloaded on first
start and served with onnxruntime. :class:`LivenessEnsemble` aggregates the
per-model live probabilities (``min`` by default — a face is only "live" if
EVERY model agrees), which is substantially stronger against screen-replay
photos than the single binary model.

Fallback: a model that cannot be downloaded/loaded is skipped; if NO model
loads the ensemble switches to ``disabled`` mode — a WARNING is logged at
startup, every request gets ``liveness_score=1.0, passed=True`` and
``GET /health`` reports ``liveness: "disabled"``. There is no heuristic
pseudo-liveness.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from typing import Any

import cv2
import numpy as np

from app.logging import get_logger

logger = get_logger(__name__)

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_DISABLED = "disabled"

AGGREGATION_MIN = "min"
AGGREGATION_MEAN = "mean"


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def expand_crop(img_bgr: np.ndarray, bbox: np.ndarray, scale: float) -> np.ndarray:
    """Square crop centered on ``bbox``, side = max(w, h) * scale.

    Regions outside the image are zero-padded, matching the preprocessing the
    MiniFASNet ONNX conversion was trained/exported with.
    """
    height, width = img_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    side = max(x2 - x1, y2 - y1) * scale
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx2 = int(round(cx + side / 2.0))
    ny2 = int(round(cy + side / 2.0))

    pad_left = max(0, -nx1)
    pad_top = max(0, -ny1)
    pad_right = max(0, nx2 - width)
    pad_bottom = max(0, ny2 - height)

    crop = img_bgr[max(0, ny1) : min(height, ny2), max(0, nx1) : min(width, nx2)]
    if crop.size == 0:
        return np.zeros((max(int(side), 1), max(int(side), 1), 3), dtype=img_bgr.dtype)
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    return crop


class LivenessEngine:
    """MiniFASNet ONNX anti-spoofing engine with a graceful "disabled" mode."""

    def __init__(
        self,
        model_path: str,
        model_url: str,
        input_size: int = 128,
        bbox_scale: float = 1.5,
        live_index: int = 0,
        use_gpu: bool = False,
        download_timeout: float = 60.0,
    ) -> None:
        self.model_path = model_path
        self.model_url = model_url
        self.input_size = input_size
        self.bbox_scale = bbox_scale
        self.live_index = live_index
        self.use_gpu = use_gpu
        self.download_timeout = download_timeout
        self._session: Any = None
        self._input_name: str = ""

    @property
    def available(self) -> bool:
        return self._session is not None

    @property
    def status(self) -> str:
        return STATUS_OK if self.available else STATUS_DISABLED

    def _ensure_model_file(self) -> None:
        """Download the ONNX model on first start if it is not on disk yet."""
        if os.path.isfile(self.model_path):
            return
        directory = os.path.dirname(self.model_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        logger.info("liveness_model_download_start", url=self.model_url, path=self.model_path)
        # Jarayonga XOS vaqtinchalik fayl: ko'p worker (uvicorn --workers>1) bir
        # vaqtda yuklab olsa, sobit "*.part" ustma-ust yozilib ONNX buzilardi
        # (keyin os.replace buzuq faylni qo'yib, engine jimgina disabled bo'lardi).
        # mkstemp har jarayonga alohida fayl beradi; os.replace atomik.
        fd, tmp_path = tempfile.mkstemp(dir=directory or ".", suffix=".part")
        try:
            request = urllib.request.Request(
                self.model_url, headers={"User-Agent": "faceid-face-service/1.0"}
            )
            with urllib.request.urlopen(request, timeout=self.download_timeout) as response:
                with os.fdopen(fd, "wb") as output:
                    shutil.copyfileobj(response, output)
            os.replace(tmp_path, self.model_path)
        except BaseException:
            # Yarim yuklangan vaqtinchalik faylni tozalaymiz
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        logger.info(
            "liveness_model_download_done",
            path=self.model_path,
            size_bytes=os.path.getsize(self.model_path),
        )

    def load(self) -> None:
        """Load (downloading if needed) the ONNX model; never raises.

        On any failure the engine stays in ``disabled`` mode and a WARNING is
        logged — the service keeps working, returning ``liveness_score=1.0``.
        """
        try:
            self._ensure_model_file()
            import onnxruntime as ort  # deferred heavy import

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.use_gpu
                else ["CPUExecutionProvider"]
            )
            # 128x128 kirish uchun ko'p thread foyda bermaydi; default (barcha
            # yadrolar) esa InsightFace bilan bir jarayonda CPU talashib,
            # parallel burst tahlilini sekinlashtiradi — 2 ta yetarli.
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                self.model_path, sess_options=options, providers=providers
            )
            self._input_name = session.get_inputs()[0].name
            self._session = session
            logger.info("liveness_model_loaded", path=self.model_path, providers=providers)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully by design
            self._session = None
            logger.warning(
                "liveness_disabled",
                error=str(exc),
                url=self.model_url,
                hint=(
                    "Anti-spoofing model unavailable; running in disabled mode: "
                    "liveness_score=1.0 / passed=true for every request"
                ),
            )

    def score(self, img_bgr: np.ndarray, bbox: np.ndarray) -> float:
        """Live-face probability (0..1) for the face at ``bbox``.

        Returns 1.0 in disabled mode; fails closed (0.0) on inference errors.
        """
        if self._session is None:
            return 1.0
        try:
            crop = expand_crop(img_bgr, bbox, self.bbox_scale)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (self.input_size, self.input_size))
            blob = crop.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
            output = self._session.run(None, {self._input_name: blob})[0]
            probs = softmax(np.asarray(output, dtype=np.float64).ravel())
            index = self.live_index if self.live_index < probs.size else 0
            return float(probs[index])
        except Exception as exc:  # noqa: BLE001 — fail closed on spoof-check errors
            logger.error("liveness_inference_failed", error=str(exc))
            return 0.0


class LivenessEnsemble:
    """Aggregates several :class:`LivenessEngine` models into one score.

    Duck-type compatible with a single engine (``available`` / ``status`` /
    ``score``), so callers do not care whether one or many models back it.

    Aggregation: ``min`` (default) — every model must consider the face live,
    the strongest anti-spoof posture; ``mean`` — softer, fewer false rejects.
    """

    def __init__(self, models: list[LivenessEngine], aggregation: str = AGGREGATION_MIN) -> None:
        self.models = models
        self.aggregation = aggregation if aggregation in (AGGREGATION_MIN, AGGREGATION_MEAN) else AGGREGATION_MIN

    @property
    def _loaded(self) -> list[LivenessEngine]:
        return [m for m in self.models if m.available]

    @property
    def available(self) -> bool:
        return len(self._loaded) > 0

    @property
    def status(self) -> str:
        loaded = len(self._loaded)
        if loaded == 0:
            return STATUS_DISABLED
        if loaded < len(self.models):
            return STATUS_PARTIAL
        return STATUS_OK

    def load(self) -> None:
        """Load every member model (each degrades gracefully on failure)."""
        for model in self.models:
            model.load()
        logger.info(
            "liveness_ensemble_ready",
            models_total=len(self.models),
            models_loaded=len(self._loaded),
            aggregation=self.aggregation,
        )

    def score(self, img_bgr: np.ndarray, bbox: np.ndarray) -> float:
        """Aggregated live probability; 1.0 when no model is loaded (disabled)."""
        loaded = self._loaded
        if not loaded:
            return 1.0
        scores = [model.score(img_bgr, bbox) for model in loaded]
        if self.aggregation == AGGREGATION_MEAN:
            return float(np.mean(scores))
        return float(min(scores))
