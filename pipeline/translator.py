"""
pipeline/translator.py
Multi-engine Neural Machine Translation:
  - MarianMT   → European languages (fast, ~300 MB each)
  - mBART-50   → Hindi (hi) — Facebook multilingual model
  - NLLB-200   → Marathi (mr) — Meta 200-language distilled model (~2.4 GB)
Falls back to a labelled mock if a model cannot be downloaded.
"""

import logging

logger = logging.getLogger(__name__)

# ── MarianMT pairs (European) ─────────────────────────────────────────────────
MARIAN_MODELS = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
    ("en", "ro"): "Helsinki-NLP/opus-mt-en-ro",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
}

# ── mBART-50 language codes ───────────────────────────────────────────────────
MBART_MODEL    = "facebook/mbart-large-50-many-to-many-mmt"
MBART_LANG_MAP = {
    "en": "en_XX",
    "hi": "hi_IN",
    "fr": "fr_XX",
    "de": "de_DE",
    "es": "es_XX",
    "it": "it_IT",
    "zh": "zh_CN",
    "ar": "ar_AR",
    "ru": "ru_RU",
}

# ── NLLB-200 language codes ───────────────────────────────────────────────────
NLLB_MODEL    = "facebook/nllb-200-distilled-600M"
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "mr": "mar_Deva",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
}


class Translator:
    def __init__(self):
        self._marian_cache = {}
        self._mbart_cache  = {}
        self._nllb_cache   = {}

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src = src_lang.lower()
        tgt = tgt_lang.lower()

        # Marathi always uses NLLB-200
        if "mr" in (src, tgt):
            return self._nllb_translate(text, src, tgt)

        # Hindi uses mBART-50 (falls back to NLLB if mBART fails)
        if "hi" in (src, tgt):
            result = self._mbart_translate(text, src, tgt)
            if result.startswith("[Mock"):
                result = self._nllb_translate(text, src, tgt)
            return result

        # European pairs use MarianMT
        model_name = MARIAN_MODELS.get((src, tgt))
        if model_name:
            return self._marian_translate(text, model_name)

        logger.warning(f"No model for {src}→{tgt}")
        return self._mock_translate(text, tgt)

    # ── MarianMT ──────────────────────────────────────────────────────────────

    def _marian_translate(self, text: str, model_name: str) -> str:
        tok, model = self._load_marian(model_name)
        if tok is None:
            return self._mock_translate(text, model_name.split("-")[-1])
        try:
            import torch
            inputs = tok(text, return_tensors="pt",
                         padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                out = model.generate(**inputs, num_beams=4,
                                     max_length=512, early_stopping=True)
            return tok.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"MarianMT error: {e}")
            return self._mock_translate(text, "target")

    def _load_marian(self, model_name):
        if model_name in self._marian_cache:
            return self._marian_cache[model_name]
        try:
            from transformers import MarianMTModel, MarianTokenizer
            logger.info(f"Downloading MarianMT: {model_name}")
            tok   = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            self._marian_cache[model_name] = (tok, model)
            return tok, model
        except Exception as e:
            logger.warning(f"MarianMT load failed: {e}")
            return None, None

    # ── mBART-50 ──────────────────────────────────────────────────────────────

    def _mbart_translate(self, text: str, src: str, tgt: str) -> str:
        src_code = MBART_LANG_MAP.get(src)
        tgt_code = MBART_LANG_MAP.get(tgt)
        if not src_code or not tgt_code:
            return self._mock_translate(text, tgt)

        tok, model = self._load_mbart()
        if tok is None:
            return self._mock_translate(text, tgt)
        try:
            import torch
            tok.src_lang = src_code
            inputs = tok(text, return_tensors="pt",
                         padding=True, truncation=True, max_length=512)
            forced_bos = tok.lang_code_to_id[tgt_code]
            with torch.no_grad():
                out = model.generate(**inputs,
                                     forced_bos_token_id=forced_bos,
                                     num_beams=4, max_length=512,
                                     early_stopping=True)
            return tok.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"mBART error: {e}")
            return self._mock_translate(text, tgt)

    def _load_mbart(self):
        if "mbart" in self._mbart_cache:
            return self._mbart_cache["mbart"]
        try:
            from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
            logger.info(f"Downloading mBART-50 (~2.3 GB, first run only)…")
            tok   = MBart50TokenizerFast.from_pretrained(MBART_MODEL)
            model = MBartForConditionalGeneration.from_pretrained(MBART_MODEL)
            self._mbart_cache["mbart"] = (tok, model)
            return tok, model
        except Exception as e:
            logger.warning(f"mBART load failed: {e}")
            self._mbart_cache["mbart"] = (None, None)
            return None, None

    # ── NLLB-200 ──────────────────────────────────────────────────────────────

    def _nllb_translate(self, text: str, src: str, tgt: str) -> str:
        src_code = NLLB_LANG_MAP.get(src)
        tgt_code = NLLB_LANG_MAP.get(tgt)
        if not src_code or not tgt_code:
            return self._mock_translate(text, tgt)

        tok, model = self._load_nllb()
        if tok is None:
            return self._mock_translate(text, tgt)
        try:
            import torch
            inputs = tok(text, return_tensors="pt",
                         src_lang=src_code,
                         padding=True, truncation=True, max_length=512)
            forced_bos = tok.convert_tokens_to_ids(tgt_code)
            with torch.no_grad():
                out = model.generate(**inputs,
                                     forced_bos_token_id=forced_bos,
                                     num_beams=4, max_length=512,
                                     early_stopping=True)
            return tok.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"NLLB error: {e}")
            return self._mock_translate(text, tgt)

    def _load_nllb(self):
        if "nllb" in self._nllb_cache:
            return self._nllb_cache["nllb"]
        try:
            from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
            logger.info(f"Downloading NLLB-200-distilled (~2.4 GB, first run only)…")
            tok   = NllbTokenizerFast.from_pretrained(NLLB_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
            self._nllb_cache["nllb"] = (tok, model)
            return tok, model
        except Exception as e:
            logger.warning(f"NLLB load failed: {e}")
            self._nllb_cache["nllb"] = (None, None)
            return None, None

    @staticmethod
    def _mock_translate(text: str, tgt_lang: str) -> str:
        return (
            f"[Mock {tgt_lang.upper()} translation — model not loaded]\n\n{text}"
        )
