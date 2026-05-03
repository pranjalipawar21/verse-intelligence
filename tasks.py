"""
tasks.py
Celery task definitions for async poetry translation pipeline.
"""

import os
from celery import Celery

# ── Celery app ────────────────────────────────────────────────────────────────

BROKER  = os.environ.get("CELERY_BROKER_URL",  "redis://localhost:6379/0")
BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery = Celery("verse_intelligence", broker=BROKER, backend=BACKEND)

celery.conf.update(
    task_serializer        = "json",
    result_serializer      = "json",
    accept_content         = ["json"],
    task_track_started     = True,
    task_acks_late         = True,
    worker_prefetch_multiplier = 1,
    result_expires         = 3600,   # results kept 1 hour
)


# ── Main translation task ─────────────────────────────────────────────────────

@celery.task(bind=True, name="tasks.translate")
def translate_task(self, poem: str, src_lang: str, tgt_lang: str,
                   mode: str, llm_api_key: str, use_rag: bool) -> dict:
    """
    Run the full poetry NMT pipeline asynchronously.
    Progress is reported via self.update_state() and polled by /api/task/<id>.
    """

    def progress(stage: str, pct: int):
        self.update_state(
            state="PROGRESS",
            meta={"stage": stage, "percent": pct},
        )

    try:
        progress("Initialising pipeline", 5)
        from pipeline.orchestrator import PoetryPipeline
        pipeline = PoetryPipeline()

        progress("Preprocessing (NLTK + spaCy)", 15)
        # Orchestrator runs all stages internally; we trust its logging for sub-progress.
        result = _run_with_progress(pipeline, poem, src_lang, tgt_lang,
                                    mode, llm_api_key, use_rag, progress)
        return result

    except Exception as exc:
        self.update_state(
            state="FAILURE",
            meta={"error": str(exc)},
        )
        raise


def _run_with_progress(pipeline, poem, src_lang, tgt_lang,
                       mode, llm_api_key, use_rag, progress):
    """Thin wrapper that emits progress checkpoints around pipeline stages."""
    from pipeline.preprocessor import Preprocessor
    from pipeline.translator   import Translator
    from pipeline.embedder     import Embedder
    from pipeline.evaluator    import Evaluator
    from pipeline.sentiment    import SentimentAnalyzer
    from pipeline.llm_enhancer import LLMEnhancer
    from pipeline.rag          import RAGModule

    result = {}

    progress("Preprocessing (NLTK + spaCy)", 10)
    result["preprocessing"] = Preprocessor().process(poem)

    rag_context = ""
    if use_rag:
        progress("RAG retrieval (FAISS)", 20)
        rag_context = RAGModule().retrieve(poem)
        result["rag_context"] = rag_context

    progress("Neural translation (MarianMT / mBART / NLLB)", 35)
    raw_translation = Translator().translate(poem, src_lang=src_lang, tgt_lang=tgt_lang)
    result["raw_translation"] = raw_translation

    enhanced = raw_translation
    if mode == "context_aware" and llm_api_key:
        progress("LLM enhancement (GPT-4o / Claude)", 55)
        enhanced = LLMEnhancer().enhance(
            original=poem, raw_translation=raw_translation,
            rag_context=rag_context, api_key=llm_api_key, tgt_lang=tgt_lang,
        )
    result["enhanced_translation"] = enhanced

    progress("Generating embeddings (Sentence-BERT)", 70)
    embedder  = Embedder()
    orig_emb  = embedder.embed(poem)
    trans_emb = embedder.embed(enhanced)
    result["embeddings"] = {
        "original_shape":   list(orig_emb.shape),
        "translated_shape": list(trans_emb.shape),
    }

    progress("Computing evaluation metrics", 85)
    result["metrics"] = Evaluator().evaluate(
        original=poem, translated=enhanced,
        orig_emb=orig_emb, trans_emb=trans_emb,
    )

    progress("Sentiment analysis", 95)
    sa = SentimentAnalyzer()
    orig_sent  = sa.analyze(poem)
    trans_sent = sa.analyze(enhanced)
    result["sentiment"] = {
        "original":   orig_sent,
        "translated": trans_sent,
        "drift":      sa.drift(orig_sent, trans_sent),
    }

    return result
