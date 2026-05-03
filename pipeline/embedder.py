"""
pipeline/embedder.py
Sentence-BERT embeddings using sentence-transformers.
Falls back to TF-IDF vectors when the library is unavailable.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence-BERT loaded: all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"sentence-transformers unavailable ({e}). "
                           "Falling back to TF-IDF.")
            self._model = "tfidf"

    # ─────────────────────────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        self._ensure_loaded()

        if self._model == "tfidf":
            return self._tfidf_embed(text)

        try:
            vec = self._model.encode(text, convert_to_numpy=True)
            return vec
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return self._tfidf_embed(text)

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _tfidf_embed(text: str) -> np.ndarray:
        """Minimal character-ngram bag-of-words fallback (dim=256)."""
        vec = np.zeros(256, dtype=np.float32)
        for i, ch in enumerate(text.lower()):
            vec[ord(ch) % 256] += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
