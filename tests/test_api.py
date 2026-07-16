"""API tests: api-key guard, schema validation and endpoint flows with fakes."""

from __future__ import annotations

from app.main import app
from app.services.face import EnrollResult
from tests.conftest import (
    FakeFaceEngine,
    FakeLivenessEngine,
    FakePool,
    make_embedding,
)

EMPLOYEE_ID = "9b2e4c1a-0f2b-4a44-9a55-2f6d1a3c9e77"
COMPANY_ID = "4e8f1f60-6f0e-4c1e-8f7a-1b2c3d4e5f60"


def _png_upload() -> tuple[str, tuple[str, bytes, str]]:
    return ("images", ("face.png", b"\x89PNG\r\n\x1a\nfakebody", "image/png"))


class TestApiKeyGuard:
    def test_extract_without_key_is_401(self, client) -> None:
        response = client.post("/extract", files=[_png_upload()])
        assert response.status_code == 401

    def test_verify_without_key_is_401(self, client, image_b64) -> None:
        response = client.post(
            "/verify", json={"image_b64": image_b64, "embeddings": [make_embedding()]}
        )
        assert response.status_code == 401

    def test_identify_without_key_is_401(self, client, image_b64) -> None:
        response = client.post(
            "/identify", json={"image_b64": image_b64, "company_id": COMPANY_ID}
        )
        assert response.status_code == 401

    def test_liveness_without_key_is_401(self, client, image_b64) -> None:
        response = client.post("/liveness", json={"image_b64": image_b64})
        assert response.status_code == 401

    def test_wrong_key_is_401(self, client, image_b64) -> None:
        response = client.post(
            "/liveness",
            json={"image_b64": image_b64},
            headers={"X-Internal-Api-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_health_does_not_require_key(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200


class TestSchemaValidation:
    def test_verify_rejects_wrong_embedding_dim(self, client, auth_headers, image_b64) -> None:
        response = client.post(
            "/verify",
            json={"image_b64": image_b64, "embeddings": [[0.1, 0.2, 0.3]]},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_verify_rejects_empty_embeddings(self, client, auth_headers, image_b64) -> None:
        response = client.post(
            "/verify",
            json={"image_b64": image_b64, "embeddings": []},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_verify_rejects_missing_image(self, client, auth_headers) -> None:
        response = client.post(
            "/verify", json={"embeddings": [make_embedding()]}, headers=auth_headers
        )
        assert response.status_code == 422

    def test_verify_rejects_out_of_range_threshold(self, client, auth_headers, image_b64) -> None:
        response = client.post(
            "/verify",
            json={
                "image_b64": image_b64,
                "embeddings": [make_embedding()],
                "match_threshold": 1.5,
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_identify_rejects_non_uuid_company(self, client, auth_headers, image_b64) -> None:
        response = client.post(
            "/identify",
            json={"image_b64": image_b64, "company_id": "not-a-uuid"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_liveness_rejects_empty_image(self, client, auth_headers) -> None:
        response = client.post("/liveness", json={"image_b64": ""}, headers=auth_headers)
        assert response.status_code == 422


class TestModelNotLoaded:
    def test_extract_returns_503_without_engine(self, client, auth_headers) -> None:
        response = client.post("/extract", files=[_png_upload()], headers=auth_headers)
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "MODEL_NOT_LOADED"


class TestExtract:
    def test_happy_path(self, client, auth_headers) -> None:
        app.state.face_engine = FakeFaceEngine()
        response = client.post("/extract", files=[_png_upload()], headers=auth_headers)
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["ok"] is True
        assert len(results[0]["embedding"]) == 512
        assert results[0]["error"] is None

    def test_face_error_is_reported_per_image(self, client, auth_headers) -> None:
        app.state.face_engine = FakeFaceEngine(
            enroll_result=EnrollResult(
                ok=False, embedding=None, quality=0.12, error="FACE_LOW_QUALITY"
            )
        )
        response = client.post("/extract", files=[_png_upload()], headers=auth_headers)
        assert response.status_code == 200
        item = response.json()["results"][0]
        assert item["ok"] is False
        assert item["embedding"] is None
        assert item["error"] == "FACE_LOW_QUALITY"
        assert item["quality"] == 0.12

    def test_more_than_five_images_rejected(self, client, auth_headers) -> None:
        app.state.face_engine = FakeFaceEngine()
        files = [_png_upload() for _ in range(6)]
        response = client.post("/extract", files=files, headers=auth_headers)
        assert response.status_code == 422


class TestVerify:
    def test_match(self, client, auth_headers, image_b64) -> None:
        embedding = make_embedding(seed=42)
        app.state.face_engine = FakeFaceEngine(embedding=embedding)
        app.state.liveness_engine = FakeLivenessEngine(score=0.92)
        response = client.post(
            "/verify",
            json={"image_b64": image_b64, "embeddings": [embedding]},
            headers=auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["match"] is True
        assert body["similarity"] > 0.99
        assert body["liveness_score"] == 0.92
        assert body["liveness_passed"] is True
        assert body["error"] is None

    def test_no_match_for_different_embedding(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine(embedding=make_embedding(seed=1))
        app.state.liveness_engine = FakeLivenessEngine(score=0.92)
        response = client.post(
            "/verify",
            json={"image_b64": image_b64, "embeddings": [make_embedding(seed=2)]},
            headers=auth_headers,
        )
        body = response.json()
        assert body["match"] is False
        assert body["similarity"] < 0.5

    def test_liveness_below_threshold_blocks_match(
        self, client, auth_headers, image_b64
    ) -> None:
        """XAVFSIZLIK: embedding mos bo'lsa ham liveness o'tmasa match=false."""
        embedding = make_embedding(seed=42)
        app.state.face_engine = FakeFaceEngine(embedding=embedding)
        app.state.liveness_engine = FakeLivenessEngine(score=0.3)
        response = client.post(
            "/verify",
            json={"image_b64": image_b64, "embeddings": [embedding]},
            headers=auth_headers,
        )
        body = response.json()
        assert body["liveness_passed"] is False
        assert body["match"] is False
        assert body["error"] == "LIVENESS_FAILED"
        assert body["similarity"] > 0.99  # o'xshashlik hisobot uchun saqlanadi

    def test_liveness_required_failcloses_when_engine_unavailable(
        self, client, auth_headers, image_b64
    ) -> None:
        """LIVENESS_REQUIRED=true + ansambl yo'q → embedding mos bo'lsa ham
        match=false (modellar tushmay qolganda spoof ochiq o'tib ketmaydi)."""
        from app.config import get_settings

        embedding = make_embedding(seed=42)
        app.state.face_engine = FakeFaceEngine(embedding=embedding)
        app.state.liveness_engine = FakeLivenessEngine(available=False)
        get_settings().liveness_required = True
        try:
            response = client.post(
                "/verify",
                json={"image_b64": image_b64, "embeddings": [embedding]},
                headers=auth_headers,
            )
            body = response.json()
            assert body["liveness_passed"] is False
            assert body["match"] is False
            assert body["error"] == "LIVENESS_FAILED"
        finally:
            get_settings().liveness_required = False

    def test_check_liveness_false_skips_check(self, client, auth_headers, image_b64) -> None:
        embedding = make_embedding(seed=42)
        app.state.face_engine = FakeFaceEngine(embedding=embedding)
        app.state.liveness_engine = FakeLivenessEngine(score=0.0)
        response = client.post(
            "/verify",
            json={
                "image_b64": image_b64,
                "embeddings": [embedding],
                "check_liveness": False,
            },
            headers=auth_headers,
        )
        body = response.json()
        # Tekshirilmagan liveness endi 1.0 EMAS, None — "skor yo'q" ma'nosida
        # (backend kiosk yo'lida None fail-closed, ya'ni rad etiladi).
        assert body["liveness_score"] is None
        assert body["liveness_passed"] is True

    def test_custom_threshold_overrides_default(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine(embedding=make_embedding(seed=1))
        app.state.liveness_engine = FakeLivenessEngine()
        response = client.post(
            "/verify",
            json={
                "image_b64": image_b64,
                "embeddings": [make_embedding(seed=2)],
                "match_threshold": 0.0,
            },
            headers=auth_headers,
        )
        assert response.json()["match"] is True

    def test_face_not_found(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine(face_found=False)
        response = client.post(
            "/verify",
            json={"image_b64": image_b64, "embeddings": [make_embedding()]},
            headers=auth_headers,
        )
        body = response.json()
        assert body["match"] is False
        assert body["error"] == "FACE_NOT_FOUND"

    def test_invalid_base64_image(self, client, auth_headers) -> None:
        app.state.face_engine = FakeFaceEngine()
        response = client.post(
            "/verify",
            json={"image_b64": "%%%invalid%%%", "embeddings": [make_embedding()]},
            headers=auth_headers,
        )
        body = response.json()
        assert body["match"] is False
        assert body["error"] == "INVALID_IMAGE"


class TestIdentify:
    def _payload(self, image_b64: str) -> dict:
        return {"image_b64": image_b64, "company_id": COMPANY_ID}

    def test_found(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine()
        app.state.liveness_engine = FakeLivenessEngine(score=0.88)
        app.state.db_pool = FakePool(
            rows=[{"employee_id": EMPLOYEE_ID, "similarity": 0.83}]
        )
        response = client.post(
            "/identify", json=self._payload(image_b64), headers=auth_headers
        )
        assert response.status_code == 200
        body = response.json()
        assert body["found"] is True
        assert body["employee_id"] == EMPLOYEE_ID
        assert body["similarity"] == 0.83
        assert body["liveness_passed"] is True

    def test_below_threshold_not_found(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine()
        app.state.liveness_engine = FakeLivenessEngine()
        app.state.db_pool = FakePool(
            rows=[{"employee_id": EMPLOYEE_ID, "similarity": 0.31}]
        )
        response = client.post(
            "/identify", json=self._payload(image_b64), headers=auth_headers
        )
        body = response.json()
        assert body["found"] is False
        assert body["employee_id"] is None
        assert body["similarity"] == 0.31

    def test_empty_search_result(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine()
        app.state.liveness_engine = FakeLivenessEngine()
        app.state.db_pool = FakePool(rows=[])
        response = client.post(
            "/identify", json=self._payload(image_b64), headers=auth_headers
        )
        body = response.json()
        assert body["found"] is False
        assert body["employee_id"] is None

    def test_no_db_pool_returns_503(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine()
        response = client.post(
            "/identify", json=self._payload(image_b64), headers=auth_headers
        )
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "DB_UNAVAILABLE"

    def test_liveness_failed_hides_employee(self, client, auth_headers, image_b64) -> None:
        """XAVFSIZLIK: spoof aniqlanganda xodim ID'si ham oshkor bo'lmaydi."""
        app.state.face_engine = FakeFaceEngine()
        app.state.liveness_engine = FakeLivenessEngine(score=0.2)
        app.state.db_pool = FakePool(
            rows=[{"employee_id": EMPLOYEE_ID, "similarity": 0.85}]
        )
        response = client.post(
            "/identify", json=self._payload(image_b64), headers=auth_headers
        )
        body = response.json()
        assert body["found"] is False
        assert body["employee_id"] is None
        assert body["error"] == "LIVENESS_FAILED"


def _real_png_bytes() -> bytes:
    """cv2 bilan dekodlanadigan haqiqiy PNG (frame'lar real decode qilinadi)."""
    import cv2
    import numpy as _np

    img = _np.zeros((32, 32, 3), dtype=_np.uint8)
    img[8:24, 8:24] = (10, 120, 240)
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    return encoded.tobytes()


class TestVerifyLive:
    """POST /verify-live — burst verifikatsiya oqimi (fakes bilan)."""

    def _post(self, client, auth_headers, embedding, frames=4, challenge="turn"):
        png = _real_png_bytes()
        files = [
            ("frames", (f"frame{i}.png", png, "image/png")) for i in range(frames)
        ]
        import json as _json

        return client.post(
            "/verify-live",
            files=files,
            data={"embeddings": _json.dumps([embedding]), "challenge": challenge},
            headers=auth_headers,
        )

    def test_head_turn_pass(self, client, auth_headers) -> None:
        embedding = make_embedding(seed=7)
        app.state.face_engine = FakeFaceEngine(
            embedding=embedding, yaws=[-9.0, -2.0, 5.0, 8.0]
        )
        app.state.liveness_engine = FakeLivenessEngine(score=0.9)
        response = self._post(client, auth_headers, embedding)
        assert response.status_code == 200
        body = response.json()
        assert body["match"] is True
        assert body["challenge_passed"] is True
        assert body["liveness_passed"] is True
        assert body["frames_valid"] == 4
        assert body["error"] is None

    def test_static_photo_fails_challenge(self, client, auth_headers) -> None:
        """Statik rasm: yaw o'zgarmaydi, blink yo'q → CHALLENGE_FAILED."""
        embedding = make_embedding(seed=7)
        app.state.face_engine = FakeFaceEngine(
            embedding=embedding, yaws=[1.0, 1.2, 0.8, 1.1]
        )
        app.state.liveness_engine = FakeLivenessEngine(score=0.9)
        response = self._post(client, auth_headers, embedding)
        body = response.json()
        assert body["match"] is False
        assert body["error"] == "CHALLENGE_FAILED"

    def test_blink_passes_challenge_without_turn(self, client, auth_headers) -> None:
        embedding = make_embedding(seed=7)
        app.state.face_engine = FakeFaceEngine(
            embedding=embedding,
            yaws=[1.0, 1.0, 1.0, 1.0],
            ears=[0.30, 0.12, 0.28, 0.29],  # bitta kadr yopiq ko'z (blink)
        )
        app.state.liveness_engine = FakeLivenessEngine(score=0.9)
        response = self._post(client, auth_headers, embedding)
        body = response.json()
        assert body["match"] is True
        assert body["challenge_passed"] is True

    def test_spoof_photo_fails_liveness(self, client, auth_headers) -> None:
        embedding = make_embedding(seed=7)
        app.state.face_engine = FakeFaceEngine(
            embedding=embedding, yaws=[-9.0, -2.0, 5.0, 8.0]
        )
        app.state.liveness_engine = FakeLivenessEngine(score=0.2)
        response = self._post(client, auth_headers, embedding)
        body = response.json()
        assert body["match"] is False
        assert body["error"] == "LIVENESS_FAILED"
        assert body["liveness_passed"] is False

    def test_wrong_person_not_recognized(self, client, auth_headers) -> None:
        app.state.face_engine = FakeFaceEngine(
            embedding=make_embedding(seed=1), yaws=[-9.0, -2.0, 5.0, 8.0]
        )
        app.state.liveness_engine = FakeLivenessEngine(score=0.9)
        response = self._post(client, auth_headers, make_embedding(seed=2))
        body = response.json()
        assert body["match"] is False
        assert body["error"] == "FACE_NOT_RECOGNIZED"

    def test_no_face_in_frames(self, client, auth_headers) -> None:
        app.state.face_engine = FakeFaceEngine(face_found=False)
        app.state.liveness_engine = FakeLivenessEngine(score=0.9)
        response = self._post(client, auth_headers, make_embedding())
        body = response.json()
        assert body["match"] is False
        assert body["frames_valid"] == 0
        assert body["error"] == "FACE_NOT_FOUND"

    def test_challenge_none_skips_challenge(self, client, auth_headers) -> None:
        embedding = make_embedding(seed=7)
        app.state.face_engine = FakeFaceEngine(embedding=embedding)
        app.state.liveness_engine = FakeLivenessEngine(score=0.9)
        response = self._post(client, auth_headers, embedding, challenge="none")
        body = response.json()
        assert body["match"] is True
        assert body["challenge_passed"] is True

    def test_invalid_embeddings_json_is_422(self, client, auth_headers) -> None:
        app.state.face_engine = FakeFaceEngine()
        files = [("frames", ("f.png", _real_png_bytes(), "image/png"))] * 3
        response = client.post(
            "/verify-live",
            files=files,
            data={"embeddings": "not-json"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_single_frame_is_422(self, client, auth_headers) -> None:
        app.state.face_engine = FakeFaceEngine()
        response = self._post(client, auth_headers, make_embedding(), frames=1)
        assert response.status_code == 422

    def test_requires_api_key(self, client) -> None:
        response = client.post(
            "/verify-live",
            files=[("frames", ("f.png", b"x", "image/png"))] * 3,
            data={"embeddings": "[]"},
        )
        assert response.status_code == 401


class TestLivenessEndpoint:
    def test_pass(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine()
        app.state.liveness_engine = FakeLivenessEngine(score=0.95)
        response = client.post(
            "/liveness", json={"image_b64": image_b64}, headers=auth_headers
        )
        body = response.json()
        assert body["liveness_score"] == 0.95
        assert body["passed"] is True

    def test_fail(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine()
        app.state.liveness_engine = FakeLivenessEngine(score=0.2)
        response = client.post(
            "/liveness", json={"image_b64": image_b64}, headers=auth_headers
        )
        body = response.json()
        assert body["liveness_score"] == 0.2
        assert body["passed"] is False

    def test_face_not_found(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine(face_found=False)
        app.state.liveness_engine = FakeLivenessEngine()
        response = client.post(
            "/liveness", json={"image_b64": image_b64}, headers=auth_headers
        )
        body = response.json()
        assert body["passed"] is False
        assert body["error"] == "FACE_NOT_FOUND"


class TestAnalyze:
    """POST /analyze — real-time kadr tahlili (WS oqimi uchun)."""

    def test_face_found_with_normalized_bbox(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine(yaws=[5.0], ears=[0.3])
        app.state.liveness_engine = FakeLivenessEngine(score=0.88)
        response = client.post(
            "/analyze", json={"image_b64": image_b64}, headers=auth_headers
        )
        assert response.status_code == 200
        body = response.json()
        assert body["found"] is True
        assert body["multiple"] is False
        # Fixture rasmi 32x32; fake bbox [10,10,120,130] → normalizatsiya
        assert body["frame_width"] == 32
        assert body["frame_height"] == 32
        assert 0.0 <= body["x"] <= 1.0
        assert body["width"] > 0
        assert body["yaw"] == 5.0
        assert body["ear"] == 0.3
        # Sifat metrikalari va har-kadr passiv anti-spoof skori
        assert body["brightness"] is not None
        assert body["sharpness"] is not None
        assert body["liveness_score"] == 0.88
        assert body["landmarks"] is None  # fake engine landmark bermaydi

    def test_check_liveness_false_skips_score(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine()
        app.state.liveness_engine = FakeLivenessEngine(score=0.9)
        response = client.post(
            "/analyze",
            json={"image_b64": image_b64, "check_liveness": False},
            headers=auth_headers,
        )
        assert response.json()["liveness_score"] is None

    def test_no_face(self, client, auth_headers, image_b64) -> None:
        app.state.face_engine = FakeFaceEngine(face_found=False)
        response = client.post(
            "/analyze", json={"image_b64": image_b64}, headers=auth_headers
        )
        body = response.json()
        assert body["found"] is False
        assert body["error"] is None
        assert body["frame_width"] == 32

    def test_invalid_image(self, client, auth_headers) -> None:
        app.state.face_engine = FakeFaceEngine()
        response = client.post(
            "/analyze", json={"image_b64": "%%%"}, headers=auth_headers
        )
        body = response.json()
        assert body["found"] is False
        assert body["error"] == "INVALID_IMAGE"

    def test_requires_api_key(self, client, image_b64) -> None:
        response = client.post("/analyze", json={"image_b64": image_b64})
        assert response.status_code == 401


class TestHealth:
    def test_degraded_without_models(self, client) -> None:
        response = client.get("/health")
        body = response.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False
        assert body["db"] == "error"
        assert body["liveness"] == "disabled"

    def test_ok_with_everything_loaded(self, client) -> None:
        app.state.face_engine = FakeFaceEngine()
        app.state.liveness_engine = FakeLivenessEngine()
        app.state.db_pool = FakePool()
        response = client.get("/health")
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["db"] == "ok"
        assert body["liveness"] == "ok"

    def test_liveness_disabled_is_reported(self, client) -> None:
        app.state.face_engine = FakeFaceEngine()
        app.state.liveness_engine = FakeLivenessEngine(available=False)
        app.state.db_pool = FakePool()
        response = client.get("/health")
        assert response.json()["liveness"] == "disabled"
