"""Unit tests for the liveness helpers (no onnxruntime required)."""

from __future__ import annotations

import numpy as np

from app.services.liveness import LivenessEngine, expand_crop, softmax


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        probs = softmax(np.array([1.0, 2.0, 3.0]))
        assert abs(float(probs.sum()) - 1.0) < 1e-12

    def test_orders_preserved(self) -> None:
        probs = softmax(np.array([0.1, 5.0, -2.0]))
        assert int(np.argmax(probs)) == 1

    def test_numerically_stable_for_large_logits(self) -> None:
        probs = softmax(np.array([1000.0, 1001.0]))
        assert np.all(np.isfinite(probs))
        assert abs(float(probs.sum()) - 1.0) < 1e-12


class TestExpandCrop:
    def test_center_crop_is_square_and_scaled(self) -> None:
        img = np.full((200, 200, 3), 50, dtype=np.uint8)
        bbox = np.array([80.0, 80.0, 120.0, 120.0])  # 40x40 face
        crop = expand_crop(img, bbox, scale=1.5)
        assert crop.shape[0] == crop.shape[1] == 60  # 40 * 1.5

    def test_edge_face_is_zero_padded(self) -> None:
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        bbox = np.array([0.0, 0.0, 40.0, 40.0])
        crop = expand_crop(img, bbox, scale=2.0)
        assert crop.shape[0] == crop.shape[1] == 80
        # top-left corner lies outside the source image -> padded with zeros
        assert int(crop[0, 0, 0]) == 0
        # bottom-right of the crop is inside the source image -> original pixels
        assert int(crop[-1, -1, 0]) == 255

    def test_rectangular_bbox_uses_longest_side(self) -> None:
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        bbox = np.array([100.0, 100.0, 140.0, 180.0])  # 40x80 face
        crop = expand_crop(img, bbox, scale=1.0)
        assert crop.shape[0] == crop.shape[1] == 80


class TestDisabledMode:
    def _disabled_engine(self) -> LivenessEngine:
        engine = LivenessEngine(
            model_path="definitely/missing/model.onnx",
            model_url="invalid://nowhere/model.onnx",
            download_timeout=0.1,
        )
        engine.load()  # download fails -> disabled mode, must not raise
        return engine

    def test_load_failure_switches_to_disabled(self) -> None:
        engine = self._disabled_engine()
        assert engine.available is False
        assert engine.status == "disabled"

    def test_disabled_engine_scores_one(self) -> None:
        engine = self._disabled_engine()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = np.array([10.0, 10.0, 60.0, 60.0])
        assert engine.score(img, bbox) == 1.0
