"""Hybrid retrieval: TF-IDF + keyword fallback with time decay scoring."""
from __future__ import annotations

import logging
import math
from datetime import datetime

from agent_harness.memory.base import MemoryItem

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining TF-IDF semantic search with keyword fallback.

    Scoring formula:
        final_score = (similarity * time_decay) * (0.8 + importance_score * 0.4)

    Where time_decay = exp(-decay_rate * hours_since_creation)
    """

    def __init__(self, decay_rate: float = 0.01) -> None:
        self._decay_rate = decay_rate

    def retrieve(
        self,
        query: str,
        items: list[MemoryItem],
        top_k: int = 5,
    ) -> list[MemoryItem]:
        """Retrieve the most relevant items using hybrid scoring.

        Args:
            query: The search query.
            items: Candidate items to score and rank.
            top_k: Maximum number of results to return.

        Returns:
            Sorted list of MemoryItems with updated importance_score.
        """
        if not items or not query.strip():
            return []

        similarities = self._tfidf_similarity(query, items)
        if similarities is None:
            similarities = self._keyword_similarity(query, items)

        now = datetime.now()
        scored: list[tuple[float, MemoryItem]] = []

        for item, sim in zip(items, similarities):
            hours = (now - item.timestamp).total_seconds() / 3600
            time_decay = math.exp(-self._decay_rate * max(hours, 0))
            importance = item.importance_score if item.importance_score is not None else 0.5
            final_score = (sim * time_decay) * (0.8 + importance * 0.4)
            updated = item.model_copy(update={
                "importance_score": final_score,
                "time_score": time_decay,
            })
            scored.append((final_score, updated))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def _tfidf_similarity(
        self, query: str, items: list[MemoryItem]
    ) -> list[float] | None:
        """Compute TF-IDF cosine similarity between query and items."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
            from sklearn.metrics.pairwise import cosine_similarity  # noqa: PLC0415
        except ImportError:
            logger.debug("scikit-learn not available, using keyword fallback")
            return None

        if len(items) < 2:
            return None

        try:
            corpus = [item.content for item in items]
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vec = vectorizer.transform([query])
            scores = cosine_similarity(query_vec, tfidf_matrix)[0]
            scores_list: list[float] = scores.tolist()
            return scores_list
        except Exception:
            logger.debug("TF-IDF vectorization failed, falling back to keywords")
            return None

    def _keyword_similarity(
        self, query: str, items: list[MemoryItem]
    ) -> list[float]:
        """Simple keyword overlap similarity as fallback."""
        query_words = set(query.lower().split())
        if not query_words:
            return [0.0] * len(items)

        scores: list[float] = []
        for item in items:
            item_words = set(item.content.lower().split())
            overlap = len(query_words & item_words)
            score = overlap / len(query_words) if query_words else 0.0
            scores.append(score)
        return scores
