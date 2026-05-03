"""
pipeline/sentiment.py
Sentiment & emotion analysis: VADER + optional transformer model.
"""

import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self):
        self._vader   = None
        self._emotion = None

    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_vader(self):
        if self._vader is not None:
            return
        try:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"VADER unavailable: {e}")
            self._vader = "unavailable"

    def _ensure_emotion(self):
        if self._emotion is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            self._emotion = hf_pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-emotion",
                top_k=None,
                truncation=True,
            )
        except Exception as e:
            logger.warning(f"Emotion model unavailable: {e}")
            self._emotion = "unavailable"

    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self, text: str) -> dict:
        self._ensure_vader()
        self._ensure_emotion()

        vader_scores = {}
        if self._vader != "unavailable":
            try:
                vader_scores = self._vader.polarity_scores(text)
            except Exception as e:
                logger.warning(f"VADER scoring failed: {e}")

        emotion_scores = {}
        if self._emotion != "unavailable":
            try:
                raw = self._emotion(text[:512])
                emotion_scores = {item["label"]: round(item["score"], 4)
                                  for item in raw[0]}
            except Exception as e:
                logger.warning(f"Emotion scoring failed: {e}")

        # Derive a simple label
        compound  = vader_scores.get("compound", 0.0)
        sentiment = "positive" if compound >= 0.05 else \
                    "negative" if compound <= -0.05 else "neutral"

        return {
            "label":         sentiment,
            "compound":      round(compound, 4),
            "vader":         vader_scores,
            "emotions":      emotion_scores,
        }

    @staticmethod
    def drift(orig: dict, trans: dict) -> dict:
        """Compute the shift in compound score & dominant emotion."""
        orig_c  = orig.get("compound", 0.0)
        trans_c = trans.get("compound", 0.0)
        drift   = round(abs(orig_c - trans_c), 4)

        orig_emotion  = max(orig.get("emotions",  {"neutral": 1.0}),
                            key=orig.get("emotions",  {"neutral": 1.0}).get,
                            default="neutral")
        trans_emotion = max(trans.get("emotions", {"neutral": 1.0}),
                            key=trans.get("emotions", {"neutral": 1.0}).get,
                            default="neutral")
        return {
            "compound_drift":   drift,
            "preserved":        drift < 0.15,
            "original_emotion": orig_emotion,
            "translated_emotion": trans_emotion,
        }
