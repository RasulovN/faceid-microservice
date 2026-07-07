"""Unit tests for cosine similarity and pgvector helpers."""

from __future__ import annotations

import asyncio

import numpy as np

from app.services.matcher import (
    IDENTIFY_SQL,
    best_similarity,
    cosine_similarity,
    identify_top,
    to_pgvector,
)
from tests.conftest import FakePool, make_embedding


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        vec = make_embedding(seed=3)
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-9

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-12

    def test_opposite_vectors(self) -> None:
        a = [0.5, -0.5, 0.25]
        b = [-0.5, 0.5, -0.25]
        assert abs(cosine_similarity(a, b) + 1.0) < 1e-9

    def test_robust_to_unnormalized_inputs(self) -> None:
        a = np.array([3.0, 0.0])
        b = np.array([300.0, 0.0])
        assert abs(cosine_similarity(a, b) - 1.0) < 1e-12

    def test_zero_vector_gives_zero(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestBestSimilarity:
    def test_picks_maximum(self) -> None:
        target = make_embedding(seed=11)
        other = make_embedding(seed=12)
        assert abs(best_similarity(target, [other, target]) - 1.0) < 1e-9

    def test_empty_candidates(self) -> None:
        assert best_similarity(make_embedding(), []) == 0.0


class TestToPgvector:
    def test_format(self) -> None:
        text = to_pgvector([0.5, -1.0, 0.25])
        assert text.startswith("[") and text.endswith("]")
        parts = text[1:-1].split(",")
        assert [float(p) for p in parts] == [0.5, -1.0, 0.25]

    def test_512_dim_has_512_components(self) -> None:
        text = to_pgvector(make_embedding())
        assert len(text[1:-1].split(",")) == 512


class TestIdentifyTop:
    def test_returns_ordered_pairs_and_passes_params(self) -> None:
        pool = FakePool(
            rows=[
                {"employee_id": "11111111-1111-1111-1111-111111111111", "similarity": 0.91},
                {"employee_id": "22222222-2222-2222-2222-222222222222", "similarity": 0.42},
            ]
        )
        embedding = make_embedding(seed=5)
        company_id = "33333333-3333-3333-3333-333333333333"
        result = asyncio.run(identify_top(pool, embedding, company_id, limit=5))

        assert result == [
            ("11111111-1111-1111-1111-111111111111", 0.91),
            ("22222222-2222-2222-2222-222222222222", 0.42),
        ]
        sql, args = pool.queries[0]
        assert sql == IDENTIFY_SQL
        assert args[0].startswith("[") and args[0].endswith("]")
        assert args[1] == company_id
        assert args[2] == 5

    def test_empty_result(self) -> None:
        pool = FakePool(rows=[])
        result = asyncio.run(identify_top(pool, make_embedding(), "cid", limit=5))
        assert result == []


class TestIdentifySql:
    def test_uses_agreed_naming_convention(self) -> None:
        """Tables snake_case, columns camelCase quoted (TypeORM default)."""
        assert "FROM face_embeddings" in IDENTIFY_SQL
        assert "JOIN employees" in IDENTIFY_SQL
        assert '"employeeId"' in IDENTIFY_SQL
        assert '"companyId"' in IDENTIFY_SQL
        assert '"deletedAt"' in IDENTIFY_SQL
        assert "'FIRED'" in IDENTIFY_SQL
        assert "<=>" in IDENTIFY_SQL
