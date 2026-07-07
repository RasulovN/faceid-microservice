"""Unit tests for the quality score and image decoding helpers."""

from __future__ import annotations

import base64

import numpy as np

from app.services.face import (
    BLUR_NORM,
    MIN_FACE_SIDE_PX,
    blur_variance,
    compute_quality,
    decode_image_b64,
    decode_image_bytes,
)


class TestComputeQuality:
    def test_perfect_inputs_give_max_score(self) -> None:
        quality = compute_quality(
            det_score=1.0,
            face_width=MIN_FACE_SIDE_PX,
            face_height=MIN_FACE_SIDE_PX,
            blur_var=BLUR_NORM,
        )
        assert quality == 1.0

    def test_worst_inputs_give_zero(self) -> None:
        assert compute_quality(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_result_is_clipped_to_unit_interval(self) -> None:
        quality = compute_quality(5.0, 10_000.0, 10_000.0, 1e9)
        assert 0.0 <= quality <= 1.0

    def test_negative_inputs_are_clipped(self) -> None:
        assert compute_quality(-1.0, -50.0, -50.0, -100.0) == 0.0

    def test_higher_det_score_increases_quality(self) -> None:
        low = compute_quality(0.3, 100.0, 100.0, 80.0)
        high = compute_quality(0.9, 100.0, 100.0, 80.0)
        assert high > low

    def test_larger_face_increases_quality(self) -> None:
        small = compute_quality(0.9, 40.0, 40.0, 80.0)
        large = compute_quality(0.9, 112.0, 112.0, 80.0)
        assert large > small

    def test_sharper_image_increases_quality(self) -> None:
        blurry = compute_quality(0.9, 100.0, 100.0, 5.0)
        sharp = compute_quality(0.9, 100.0, 100.0, 100.0)
        assert sharp > blurry

    def test_min_side_drives_size_component(self) -> None:
        narrow = compute_quality(0.9, 20.0, 200.0, 80.0)
        square = compute_quality(0.9, 200.0, 200.0, 80.0)
        assert square > narrow

    def test_low_quality_threshold_case(self) -> None:
        """Tiny blurry low-confidence face must land below the 0.35 gate."""
        quality = compute_quality(0.4, 30.0, 30.0, 10.0)
        assert quality < 0.35


class TestBlurVariance:
    def test_flat_image_has_zero_variance(self) -> None:
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        assert blur_variance(flat) == 0.0

    def test_noisy_image_has_high_variance(self) -> None:
        rng = np.random.default_rng(7)
        noisy = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        assert blur_variance(noisy) > BLUR_NORM

    def test_accepts_grayscale_input(self) -> None:
        gray = np.zeros((50, 70), dtype=np.uint8)
        assert blur_variance(gray) == 0.0

    def test_empty_image_returns_zero(self) -> None:
        assert blur_variance(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


class TestDecode:
    def _png_bytes(self) -> bytes:
        import cv2

        img = np.zeros((16, 16, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", img)
        assert ok
        return encoded.tobytes()

    def test_decode_bytes_roundtrip(self) -> None:
        img = decode_image_bytes(self._png_bytes())
        assert img is not None
        assert img.shape == (16, 16, 3)

    def test_decode_bytes_rejects_garbage(self) -> None:
        assert decode_image_bytes(b"definitely-not-an-image") is None

    def test_decode_bytes_rejects_empty(self) -> None:
        assert decode_image_bytes(b"") is None

    def test_decode_b64_roundtrip(self) -> None:
        payload = base64.b64encode(self._png_bytes()).decode("ascii")
        img = decode_image_b64(payload)
        assert img is not None
        assert img.shape == (16, 16, 3)

    def test_decode_b64_supports_data_url_prefix(self) -> None:
        payload = base64.b64encode(self._png_bytes()).decode("ascii")
        img = decode_image_b64(f"data:image/png;base64,{payload}")
        assert img is not None

    def test_decode_b64_rejects_invalid_base64(self) -> None:
        assert decode_image_b64("!!!not-base64!!!") is None

    def test_decode_b64_rejects_empty(self) -> None:
        assert decode_image_b64("") is None
