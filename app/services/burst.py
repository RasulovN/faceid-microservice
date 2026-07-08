"""Multi-frame (burst) liveness + verification analysis for ``POST /verify-live``.

A burst is a short ordered sequence of frames (~2-3 s) captured while the user
performs a small challenge (turning the head slightly). The decision layers:

1. **Face presence** — enough frames must contain a face (``FACE_NOT_FOUND``).
2. **Identity consistency** — every frame must show the SAME person (min
   pairwise ArcFace similarity); photo-swap mid-burst fails here.
3. **Passive anti-spoofing** — mean of the per-frame MiniFASNet ensemble
   scores must clear the threshold (``LIVENESS_FAILED``).
4. **Identity match** — the best frame must match the enrolled embeddings
   (``FACE_NOT_RECOGNIZED``). Checked AFTER liveness so a spoof never learns
   whether its photo matches.
5. **Challenge** — head-yaw range across the burst (3D structure evidence) or
   a detected blink (``CHALLENGE_FAILED``). A flat photo neither turns with
   realistic 3D pose change nor blinks.

Every function here is pure (numpy only) and unit-testable without models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from app.services.face import ERROR_FACE_NOT_FOUND
from app.services.matcher import best_similarity, cosine_similarity

ERROR_LIVENESS_FAILED = "LIVENESS_FAILED"
ERROR_CHALLENGE_FAILED = "CHALLENGE_FAILED"
ERROR_FACE_NOT_RECOGNIZED = "FACE_NOT_RECOGNIZED"

REASON_TOO_FEW_FACES = "TOO_FEW_FRAMES_WITH_FACE"
REASON_IDENTITY_INCONSISTENT = "IDENTITY_INCONSISTENT"
REASON_PASSIVE_SPOOF = "PASSIVE_ANTISPOOF_LOW"
REASON_NO_HEAD_TURN = "NO_HEAD_TURN"
REASON_LOW_SIMILARITY = "LOW_SIMILARITY"

CHALLENGE_TURN = "turn"
CHALLENGE_NONE = "none"


@dataclass(slots=True)
class FrameObservation:
    """Per-frame facts extracted by the ML engines (or None when no face)."""

    has_face: bool
    liveness_score: float = 1.0
    embedding: np.ndarray | None = None
    yaw: float | None = None
    ear: float | None = None
    quality: float = 0.0


@dataclass(slots=True)
class BurstThresholds:
    """Decision thresholds (populated from :mod:`app.config` settings)."""

    min_valid_frames: int = 3
    consistency_threshold: float = 0.55
    liveness_threshold: float = 0.7
    yaw_range_deg: float = 12.0
    blink_ear_close: float = 0.18
    blink_ear_open: float = 0.24
    match_threshold: float = 0.5


@dataclass(slots=True)
class BurstDecision:
    """Aggregate decision returned by :func:`decide_burst`."""

    match: bool
    similarity: float = 0.0
    liveness_score: float = 0.0
    liveness_passed: bool = False
    challenge_passed: bool = False
    consistency: float = 0.0
    frames_total: int = 0
    frames_valid: int = 0
    error: str | None = None
    reasons: list[str] = field(default_factory=list)


def yaw_range(yaws: list[float | None]) -> float:
    """Spread (max - min) of the finite yaw values; 0.0 when fewer than two."""
    finite = [y for y in yaws if y is not None and np.isfinite(y)]
    if len(finite) < 2:
        return 0.0
    return float(max(finite) - min(finite))


def blink_detected(ears: list[float | None], close_thr: float, open_thr: float) -> bool:
    """True when the EAR sequence shows both an open and a closed eye state."""
    finite = [e for e in ears if e is not None and np.isfinite(e)]
    if len(finite) < 2:
        return False
    return min(finite) <= close_thr and max(finite) >= open_thr


def pairwise_min_similarity(embeddings: list[np.ndarray]) -> float:
    """Minimal pairwise cosine similarity — the weakest same-person link.

    1.0 for a single embedding (nothing to contradict); different persons in
    one burst give values near 0, a swapped photo drops it sharply.
    """
    if len(embeddings) < 2:
        return 1.0
    return min(
        cosine_similarity(a, b) for a, b in combinations(embeddings, 2)
    )


def decide_burst(
    observations: list[FrameObservation],
    enrolled_embeddings: list[list[float]],
    thresholds: BurstThresholds,
    challenge: str = CHALLENGE_TURN,
) -> BurstDecision:
    """Layered burst decision (see module docstring for the order rationale)."""
    decision = BurstDecision(match=False, frames_total=len(observations))
    valid = [o for o in observations if o.has_face and o.embedding is not None]
    decision.frames_valid = len(valid)

    # 1) Yuz yetarli kadrda bormi?
    if len(valid) < thresholds.min_valid_frames:
        decision.error = ERROR_FACE_NOT_FOUND
        decision.reasons.append(REASON_TOO_FEW_FACES)
        return decision

    # 2) Bir odammi? (kadrlararo izchillik)
    decision.consistency = round(
        pairwise_min_similarity([o.embedding for o in valid if o.embedding is not None]), 4
    )
    if decision.consistency < thresholds.consistency_threshold:
        decision.error = ERROR_LIVENESS_FAILED
        decision.reasons.append(REASON_IDENTITY_INCONSISTENT)
        return decision

    # 3) Passiv anti-spoof (har kadr ansambl skorining o'rtachasi)
    decision.liveness_score = round(float(np.mean([o.liveness_score for o in valid])), 4)
    decision.liveness_passed = decision.liveness_score >= thresholds.liveness_threshold
    if not decision.liveness_passed:
        decision.error = ERROR_LIVENESS_FAILED
        decision.reasons.append(REASON_PASSIVE_SPOOF)
        return decision

    # 4) Identity match — liveness'dan KEYIN (spoof "mos keldimi"ni bilmasin).
    #    Zamonaviy usul: kadr embeddinglari SIFAT-VAZNLI O'RTACHASI (aggregate
    #    template) bilan ham solishtiriladi — bitta kadr shovqini (motion blur,
    #    qiya burchak) genuine foydalanuvchini yiqitmaydi; yakuniy o'xshashlik
    #    per-kadr maksimum va agregat natijaning kattasi.
    per_frame_best = max(
        best_similarity(o.embedding, enrolled_embeddings) for o in valid
    )
    embeddings_stack = np.stack([o.embedding for o in valid])
    weights = np.array([max(o.quality, 1e-3) for o in valid], dtype=np.float64)
    aggregate = np.average(embeddings_stack, axis=0, weights=weights)
    aggregate_sim = best_similarity(aggregate, enrolled_embeddings)
    decision.similarity = round(max(per_frame_best, aggregate_sim), 4)
    if decision.similarity < thresholds.match_threshold:
        decision.error = ERROR_FACE_NOT_RECOGNIZED
        decision.reasons.append(REASON_LOW_SIMILARITY)
        return decision

    # 5) Challenge: bosh burilishi (3D dalil) YOKI blink
    if challenge == CHALLENGE_NONE:
        decision.challenge_passed = True
    else:
        turned = yaw_range([o.yaw for o in valid]) >= thresholds.yaw_range_deg
        blinked = blink_detected(
            [o.ear for o in valid], thresholds.blink_ear_close, thresholds.blink_ear_open
        )
        decision.challenge_passed = turned or blinked
        if not decision.challenge_passed:
            decision.error = ERROR_CHALLENGE_FAILED
            decision.reasons.append(REASON_NO_HEAD_TURN)
            return decision

    decision.match = True
    return decision
