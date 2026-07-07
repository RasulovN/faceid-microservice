"""Cosine similarity and 1:N pgvector identification.

Naming convention (documented in README): tables are snake_case
(``face_embeddings``, ``employees``), columns are TypeORM-default camelCase and
therefore quoted (``"employeeId"``, ``"companyId"``, ``"deletedAt"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover — asyncpg is only needed at runtime
    from asyncpg import Pool

#: 1:N search: top-N nearest embeddings within one company, excluding fired
#: and soft-deleted employees. ``<=>`` is the pgvector cosine-distance operator,
#: so ``1 - distance`` is the cosine similarity.
IDENTIFY_SQL = """
SELECT fe."employeeId" AS employee_id,
       1 - (fe.embedding <=> $1::vector) AS similarity
FROM face_embeddings fe
JOIN employees e ON e.id = fe."employeeId"
WHERE e."companyId" = $2
  AND e.status != 'FIRED'
  AND e."deletedAt" IS NULL
  AND ($4::uuid IS NULL OR e."branchId" = $4)
ORDER BY fe.embedding <=> $1::vector
LIMIT $3
"""


def _as_unit_vector(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the vector as a float64 unit vector (zero vector stays zero)."""
    arr = np.asarray(vector, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def cosine_similarity(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> float:
    """Cosine similarity in [-1, 1]; robust to non-normalized inputs."""
    return float(np.dot(_as_unit_vector(a), _as_unit_vector(b)))


def best_similarity(
    embedding: Sequence[float] | np.ndarray,
    candidates: Sequence[Sequence[float]],
) -> float:
    """Maximum cosine similarity between ``embedding`` and each candidate."""
    if not candidates:
        return 0.0
    return max(cosine_similarity(embedding, candidate) for candidate in candidates)


def to_pgvector(embedding: Sequence[float] | np.ndarray) -> str:
    """Serialize an embedding to the pgvector text format ``[f1,f2,...]``."""
    arr = np.asarray(embedding, dtype=np.float32).ravel()
    return "[" + ",".join(f"{value:.8f}" for value in arr) + "]"


async def identify_top(
    pool: "Pool",
    embedding: Sequence[float] | np.ndarray,
    company_id: str,
    limit: int = 5,
    branch_id: str | None = None,
) -> list[tuple[str, float]]:
    """Run the pgvector 1:N search; returns ``[(employee_id, similarity), ...]``.

    Scoped to one company; if ``branch_id`` is given, further restricted to that
    branch (so an employee of another branch/company is never a candidate).
    Results are ordered by descending similarity (pgvector orders by distance).
    """
    rows = await pool.fetch(
        IDENTIFY_SQL, to_pgvector(embedding), company_id, limit, branch_id
    )
    return [(str(row["employee_id"]), float(row["similarity"])) for row in rows]
