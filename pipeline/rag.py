"""
pipeline/rag.py
RAG module: FAISS-based retrieval of similar poems for style context.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Sample poetry corpus for demo purposes
SAMPLE_POEMS = [
    {
        "title": "The Road Not Taken (excerpt)",
        "author": "Robert Frost",
        "text": "Two roads diverged in a yellow wood, And sorry I could not travel both",
        "style": "contemplative, metaphorical",
    },
    {
        "title": "I carry your heart (excerpt)",
        "author": "E.E. Cummings",
        "text": "i carry your heart with me i carry it in my heart i am never without it",
        "style": "romantic, intimate",
    },
    {
        "title": "Do Not Go Gentle (excerpt)",
        "author": "Dylan Thomas",
        "text": "Do not go gentle into that good night, rage, rage against the dying of the light",
        "style": "passionate, defiant",
    },
    {
        "title": "Still I Rise (excerpt)",
        "author": "Maya Angelou",
        "text": "You may write me down in history with your bitter, twisted lies",
        "style": "empowering, triumphant",
    },
    {
        "title": "Annabel Lee (excerpt)",
        "author": "Edgar Allan Poe",
        "text": "It was many and many a year ago, In a kingdom by the sea",
        "style": "melancholic, lyrical",
    },
]


class RAGModule:
    def __init__(self):
        self._index   = None
        self._embedder = None

    def _build_index(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [p["text"] for p in SAMPLE_POEMS]
            embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")

            dim   = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)

            self._index   = index
            self._embedder = model
            logger.info("FAISS index built with sample poems.")
        except Exception as e:
            logger.warning(f"FAISS/sentence-transformers unavailable: {e}. "
                           "RAG will use keyword matching.")
            self._index = "keyword"

    def retrieve(self, query: str, top_k: int = 2) -> str:
        if self._index is None:
            self._build_index()

        if self._index == "keyword":
            return self._keyword_retrieve(query, top_k)

        try:
            query_vec = self._embedder.encode([query],
                                              convert_to_numpy=True).astype("float32")
            distances, indices = self._index.search(query_vec, top_k)
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(SAMPLE_POEMS):
                    p = SAMPLE_POEMS[idx]
                    results.append(
                        f"[{p['title']} — {p['author']}]\n"
                        f"Style: {p['style']}\n{p['text']}"
                    )
            return "\n\n".join(results)
        except Exception as e:
            logger.warning(f"FAISS retrieval failed: {e}")
            return self._keyword_retrieve(query, top_k)

    @staticmethod
    def _keyword_retrieve(query: str, top_k: int = 2) -> str:
        query_words = set(query.lower().split())
        scored = []
        for poem in SAMPLE_POEMS:
            poem_words = set(poem["text"].lower().split())
            score = len(query_words & poem_words)
            scored.append((score, poem))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, p in scored[:top_k]:
            results.append(
                f"[{p['title']} — {p['author']}]\n"
                f"Style: {p['style']}\n{p['text']}"
            )
        return "\n\n".join(results)
