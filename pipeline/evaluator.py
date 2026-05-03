"""
pipeline/evaluator.py
Evaluation framework: Cosine Similarity, BLEU, BERTScore, Semantic Drift.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self):
        self._bertscore_loaded = None   # None = not tried yet

    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, original: str, translated: str,
                 orig_emb: np.ndarray, trans_emb: np.ndarray) -> dict:
        return {
            "cosine_similarity": float(self._cosine(orig_emb, trans_emb)),
            "semantic_drift":    float(self._semantic_drift(orig_emb, trans_emb)),
            "bleu_score":        float(self._bleu(original, translated)),
            "bert_score":        self._bert_score(original, translated),
        }

    # ── Cosine similarity ─────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    # ── Semantic drift ────────────────────────────────────────────────────────

    @staticmethod
    def _semantic_drift(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    # ── BLEU ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _bleu(reference: str, hypothesis: str) -> float:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            nltk.download("punkt", quiet=True)
            ref_tokens  = reference.lower().split()
            hyp_tokens  = hypothesis.lower().split()
            smoothie    = SmoothingFunction().method4
            score = sentence_bleu([ref_tokens], hyp_tokens,
                                  smoothing_function=smoothie)
            return round(score, 4)
        except Exception as e:
            logger.warning(f"BLEU failed: {e}")
            # Manual unigram overlap as absolute fallback
            ref_set = set(reference.lower().split())
            hyp_set = set(hypothesis.lower().split())
            if not hyp_set:
                return 0.0
            return round(len(ref_set & hyp_set) / len(hyp_set), 4)

    # ── BERTScore ─────────────────────────────────────────────────────────────

    def _bert_score(self, reference: str, hypothesis: str) -> dict:
        if self._bertscore_loaded is False:
            return self._fallback_bert_score(reference, hypothesis)
        try:
            from bert_score import score as bs_score
            P, R, F1 = bs_score(
                [hypothesis], [reference],
                lang="en",
                rescale_with_baseline=False,
                verbose=False,
            )
            self._bertscore_loaded = True
            return {
                "precision": round(float(P.mean()), 4),
                "recall":    round(float(R.mean()), 4),
                "f1":        round(float(F1.mean()), 4),
            }
        except Exception as e:
            logger.warning(f"BERTScore failed ({e}), using fallback.")
            self._bertscore_loaded = False
            return self._fallback_bert_score(reference, hypothesis)

    @staticmethod
    def _fallback_bert_score(reference: str, hypothesis: str) -> dict:
        """Token overlap F1 as fallback."""
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        if not ref_tokens or not hyp_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        precision = len(ref_tokens & hyp_tokens) / len(hyp_tokens)
        recall    = len(ref_tokens & hyp_tokens) / len(ref_tokens)
        f1        = (2 * precision * recall / (precision + recall)
                     if precision + recall > 0 else 0.0)
        return {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        }
