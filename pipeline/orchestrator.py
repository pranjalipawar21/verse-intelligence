"""
pipeline/orchestrator.py
Orchestrates the full Poetry NMT pipeline.
"""

import logging
from .preprocessor   import Preprocessor
from .translator     import Translator
from .embedder       import Embedder
from .evaluator      import Evaluator
from .sentiment      import SentimentAnalyzer
from .llm_enhancer   import LLMEnhancer
from .rag            import RAGModule

logger = logging.getLogger(__name__)


class PoetryPipeline:
    """Main pipeline: preprocess → translate → embed → evaluate → enhance."""

    def __init__(self):
        logger.info("Initialising pipeline components…")
        self.preprocessor = Preprocessor()
        self.translator   = Translator()
        self.embedder     = Embedder()
        self.evaluator    = Evaluator()
        self.sentiment    = SentimentAnalyzer()
        self.llm          = LLMEnhancer()
        self.rag          = RAGModule()
        logger.info("Pipeline ready.")

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, poem: str, src_lang: str, tgt_lang: str,
            mode: str, llm_api_key: str, use_rag: bool) -> dict:

        result = {}

        # ── 1. Preprocessing ────────────────────────────────────────────────
        logger.info("Stage 1: Preprocessing")
        preprocessed = self.preprocessor.process(poem)
        result["preprocessing"] = preprocessed

        # ── 2. RAG context retrieval (optional) ─────────────────────────────
        rag_context = ""
        if use_rag:
            logger.info("Stage 2a: RAG retrieval")
            rag_context = self.rag.retrieve(poem)
            result["rag_context"] = rag_context

        # ── 3. Neural Translation ────────────────────────────────────────────
        logger.info("Stage 3: Translation")
        raw_translation = self.translator.translate(
            poem, src_lang=src_lang, tgt_lang=tgt_lang
        )
        result["raw_translation"] = raw_translation

        # ── 4. LLM Enhancement (context-aware mode) ──────────────────────────
        enhanced_translation = raw_translation
        if mode == "context_aware" and llm_api_key:
            logger.info("Stage 4: LLM Enhancement")
            enhanced_translation = self.llm.enhance(
                original=poem,
                raw_translation=raw_translation,
                rag_context=rag_context,
                api_key=llm_api_key,
                tgt_lang=tgt_lang,
            )
        result["enhanced_translation"] = enhanced_translation

        # ── 5. Embeddings ────────────────────────────────────────────────────
        logger.info("Stage 5: Embedding")
        orig_emb  = self.embedder.embed(poem)
        trans_emb = self.embedder.embed(enhanced_translation)
        result["embeddings"] = {
            "original_shape":    list(orig_emb.shape),
            "translated_shape":  list(trans_emb.shape),
        }

        # ── 6. Evaluation ────────────────────────────────────────────────────
        logger.info("Stage 6: Evaluation")
        metrics = self.evaluator.evaluate(
            original=poem,
            translated=enhanced_translation,
            orig_emb=orig_emb,
            trans_emb=trans_emb,
        )
        result["metrics"] = metrics

        # ── 7. Sentiment Analysis ─────────────────────────────────────────────
        logger.info("Stage 7: Sentiment")
        orig_sent  = self.sentiment.analyze(poem)
        trans_sent = self.sentiment.analyze(enhanced_translation)
        result["sentiment"] = {
            "original":    orig_sent,
            "translated":  trans_sent,
            "drift":       self.sentiment.drift(orig_sent, trans_sent),
        }

        return result
