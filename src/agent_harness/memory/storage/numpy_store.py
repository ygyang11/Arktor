"""Lightweight vector store using numpy for similarity search.

Zero external dependencies beyond numpy. Suitable for small-to-medium
datasets (up to ~100k documents).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from agent_harness.memory.storage.base import (
    BaseVectorStore,
    VectorDocument,
    VectorSearchResult,
)


class NumpyVectorStore(BaseVectorStore):
    """Vector store backed by numpy arrays.

    Uses cosine similarity for search. Stores all vectors in memory.

    Example:
        store = NumpyVectorStore()
        await store.upsert([VectorDocument(id="1", content="hello", embedding=[0.1, 0.2])])
        results = await store.search(query_embedding=[0.1, 0.2], top_k=5)
    """

    def __init__(self, metric: str = "cosine") -> None:
        """Initialize the store.

        Args:
            metric: Distance metric. "cosine" or "euclidean".
        """
        self._metric = metric
        self._documents: dict[str, VectorDocument] = {}
        self._embeddings: np.ndarray | None = None
        self._id_index: list[str] = []  # maps matrix row -> doc id
        self._dirty = True  # whether the matrix needs rebuilding

    async def upsert(self, documents: list[VectorDocument]) -> None:
        for doc in documents:
            self._documents[doc.id] = doc
        self._dirty = True

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        if not self._documents:
            return []

        self._rebuild_index_if_needed()
        assert self._embeddings is not None

        query = np.array(query_embedding, dtype=np.float32)

        if self._metric == "cosine":
            scores = self._cosine_similarity(query, self._embeddings)
        else:
            # Euclidean: negate distance so higher = more similar
            distances = np.linalg.norm(self._embeddings - query, axis=1)
            scores = -distances

        # Apply metadata filter
        if filter_metadata:
            mask = np.ones(len(self._id_index), dtype=bool)
            for i, doc_id in enumerate(self._id_index):
                doc = self._documents[doc_id]
                for key, value in filter_metadata.items():
                    if doc.metadata.get(key) != value:
                        mask[i] = False
                        break
            scores = np.where(mask, scores, -np.inf)

        # Get top-k indices
        k = min(top_k, len(self._id_index))
        top_indices = np.argsort(scores)[-k:][::-1]

        results: list[VectorSearchResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score == -np.inf:
                continue
            doc_id = self._id_index[idx]
            results.append(VectorSearchResult(
                document=self._documents[doc_id],
                score=score,
            ))

        return results

    async def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._documents.pop(doc_id, None)
        self._dirty = True

    async def count(self) -> int:
        return len(self._documents)

    async def clear(self) -> None:
        self._documents.clear()
        self._embeddings = None
        self._id_index.clear()
        self._dirty = True

    async def list_all(self) -> list[VectorDocument]:
        """Return all stored documents."""
        return list(self._documents.values())

    def _rebuild_index_if_needed(self) -> None:
        if not self._dirty:
            return

        self._id_index = list(self._documents.keys())
        if not self._id_index:
            self._embeddings = None
            self._dirty = False
            return

        vectors = [self._documents[doc_id].embedding for doc_id in self._id_index]
        self._embeddings = np.array(vectors, dtype=np.float32)
        self._dirty = False

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query vector and a matrix of vectors."""
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(matrix.shape[0])

        matrix_norms = np.linalg.norm(matrix, axis=1)
        # Avoid division by zero
        matrix_norms = np.maximum(matrix_norms, 1e-10)

        result: np.ndarray = (matrix @ query) / (matrix_norms * query_norm)
        return result
