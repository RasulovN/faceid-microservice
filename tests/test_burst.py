"""Unit tests for the pure burst-decision logic (no models, numpy only)."""

from __future__ import annotations

import numpy as np

from app.services.burst import (
    CHALLENGE_NONE,
    BurstThresholds,
    FrameObservation,
    blink_detected,
    decide_burst,
    pairwise_min_similarity,
    yaw_range,
)
from app.services.face import estimate_yaw_from_kps, eye_aspect_ratio_68
from tests.conftest import make_embedding


def _emb(seed: int = 1) -> np.ndarray:
    return np.asarray(make_embedding(seed=seed), dtype=np.float32)


def _obs(
    seed: int = 1,
    yaw: float | None = 0.0,
    ear: float | None = None,
    liveness: float = 0.9,
    has_face: bool = True,
) -> FrameObservation:
    return FrameObservation(
        has_face=has_face,
        liveness_score=liveness,
        embedding=_emb(seed) if has_face else None,
        yaw=yaw,
        ear=ear,
        quality=0.9,
    )


THRESHOLDS = BurstThresholds(
    min_valid_frames=3,
    consistency_threshold=0.55,
    liveness_threshold=0.7,
    yaw_range_deg=12.0,
    blink_ear_close=0.18,
    blink_ear_open=0.24,
    match_threshold=0.5,
)


class TestHelpers:
    def test_yaw_range_ignores_none(self) -> None:
        assert yaw_range([-8.0, None, 6.0]) == 14.0

    def test_yaw_range_single_value_is_zero(self) -> None:
        assert yaw_range([5.0, None]) == 0.0

    def test_blink_needs_both_states(self) -> None:
        assert blink_detected([0.30, 0.12, 0.28], 0.18, 0.24) is True
        assert blink_detected([0.30, 0.29, 0.28], 0.18, 0.24) is False  # doim ochiq
        assert blink_detected([0.10, 0.12, 0.11], 0.18, 0.24) is False  # doim yopiq

    def test_pairwise_min_similarity_same_person(self) -> None:
        assert pairwise_min_similarity([_emb(1), _emb(1)]) > 0.999

    def test_pairwise_min_similarity_different_people(self) -> None:
        assert pairwise_min_similarity([_emb(1), _emb(2)]) < 0.3

    def test_pairwise_single_embedding_is_one(self) -> None:
        assert pairwise_min_similarity([_emb(1)]) == 1.0


class TestFaceGeometry:
    def test_estimate_yaw_centered_nose_is_zero(self) -> None:
        kps = np.array([[40.0, 50.0], [80.0, 50.0], [60.0, 70.0], [45.0, 90.0], [75.0, 90.0]])
        assert abs(estimate_yaw_from_kps(kps)) < 1e-9

    def test_estimate_yaw_offset_nose_gives_sign(self) -> None:
        kps = np.array([[40.0, 50.0], [80.0, 50.0], [70.0, 70.0], [45.0, 90.0], [75.0, 90.0]])
        assert estimate_yaw_from_kps(kps) > 5.0

    def test_estimate_yaw_degenerate_returns_none(self) -> None:
        kps = np.array([[60.0, 50.0], [60.0, 50.0], [60.0, 70.0]])
        assert estimate_yaw_from_kps(kps) is None

    def test_ear_open_eye_synthetic(self) -> None:
        # 68 nuqta: hammasi nolda, faqat ko'z nuqtalari sintetik joylashtiriladi
        pts = np.zeros((68, 2))
        for base in (36, 42):
            pts[base] = (0.0, 0.0)      # tashqi burchak
            pts[base + 3] = (10.0, 0.0)  # ichki burchak
            pts[base + 1] = (3.0, -1.5)
            pts[base + 2] = (7.0, -1.5)
            pts[base + 5] = (3.0, 1.5)
            pts[base + 4] = (7.0, 1.5)
        ear = eye_aspect_ratio_68(pts)
        assert ear is not None
        assert abs(ear - 0.3) < 1e-9  # (3 + 3) / (2 * 10)

    def test_ear_none_for_missing_landmarks(self) -> None:
        assert eye_aspect_ratio_68(None) is None
        assert eye_aspect_ratio_68(np.zeros((10, 2))) is None


class TestDecideBurst:
    def test_happy_path_head_turn(self) -> None:
        observations = [
            _obs(yaw=-9.0),
            _obs(yaw=-2.0),
            _obs(yaw=4.0),
            _obs(yaw=8.0),
        ]
        decision = decide_burst(observations, [make_embedding(seed=1)], THRESHOLDS)
        assert decision.match is True
        assert decision.error is None
        assert decision.challenge_passed is True
        assert decision.similarity > 0.99

    def test_too_few_faces(self) -> None:
        observations = [_obs(), _obs(has_face=False), _obs(has_face=False)]
        decision = decide_burst(observations, [make_embedding(seed=1)], THRESHOLDS)
        assert decision.match is False
        assert decision.error == "FACE_NOT_FOUND"
        assert decision.frames_valid == 1

    def test_photo_swap_fails_consistency(self) -> None:
        """Burst o'rtasida boshqa odam/rasm almashtirildi → LIVENESS_FAILED."""
        observations = [
            _obs(seed=1, yaw=-9.0),
            _obs(seed=1, yaw=0.0),
            _obs(seed=2, yaw=8.0),  # boshqa odam
        ]
        decision = decide_burst(observations, [make_embedding(seed=1)], THRESHOLDS)
        assert decision.match is False
        assert decision.error == "LIVENESS_FAILED"
        assert "IDENTITY_INCONSISTENT" in decision.reasons

    def test_passive_spoof_fails_before_match(self) -> None:
        observations = [
            _obs(yaw=-9.0, liveness=0.2),
            _obs(yaw=0.0, liveness=0.3),
            _obs(yaw=8.0, liveness=0.25),
        ]
        decision = decide_burst(observations, [make_embedding(seed=1)], THRESHOLDS)
        assert decision.match is False
        assert decision.error == "LIVENESS_FAILED"
        assert "PASSIVE_ANTISPOOF_LOW" in decision.reasons
        # Xavfsizlik: liveness o'tmaganda similarity hisoblanmaydi (oracle yo'q)
        assert decision.similarity == 0.0

    def test_static_photo_fails_challenge(self) -> None:
        observations = [_obs(yaw=1.0), _obs(yaw=1.4), _obs(yaw=0.8)]
        decision = decide_burst(observations, [make_embedding(seed=1)], THRESHOLDS)
        assert decision.match is False
        assert decision.error == "CHALLENGE_FAILED"

    def test_blink_satisfies_challenge(self) -> None:
        observations = [
            _obs(yaw=1.0, ear=0.30),
            _obs(yaw=1.2, ear=0.12),
            _obs(yaw=0.9, ear=0.29),
        ]
        decision = decide_burst(observations, [make_embedding(seed=1)], THRESHOLDS)
        assert decision.match is True
        assert decision.challenge_passed is True

    def test_wrong_person_after_liveness(self) -> None:
        observations = [_obs(seed=3, yaw=-9.0), _obs(seed=3, yaw=0.0), _obs(seed=3, yaw=8.0)]
        decision = decide_burst(observations, [make_embedding(seed=1)], THRESHOLDS)
        assert decision.match is False
        assert decision.error == "FACE_NOT_RECOGNIZED"
        assert decision.liveness_passed is True

    def test_challenge_none_mode(self) -> None:
        observations = [_obs(yaw=0.0), _obs(yaw=0.1), _obs(yaw=0.0)]
        decision = decide_burst(
            observations, [make_embedding(seed=1)], THRESHOLDS, challenge=CHALLENGE_NONE
        )
        assert decision.match is True
        assert decision.challenge_passed is True

    def test_missing_yaw_data_fails_challenge_closed(self) -> None:
        """Yaw ma'lumoti yo'q (None) → challenge o'tmaydi (fail-closed)."""
        observations = [_obs(yaw=None), _obs(yaw=None), _obs(yaw=None)]
        decision = decide_burst(observations, [make_embedding(seed=1)], THRESHOLDS)
        assert decision.match is False
        assert decision.error == "CHALLENGE_FAILED"
