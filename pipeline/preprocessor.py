"""
pipeline/preprocessor.py
NLTK + spaCy preprocessing: tokenisation, lemmatisation, POS tagging, NER.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Lazy imports so the app starts even without models downloaded ─────────────

def _load_nltk():
    import nltk
    for pkg in ["punkt", "averaged_perceptron_tagger", "stopwords",
                "wordnet", "punkt_tab"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return nltk

def _load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True
        )
        return spacy.load("en_core_web_sm")


class Preprocessor:
    def __init__(self):
        self._nltk  = None
        self._spacy = None

    def _ensure_loaded(self):
        if self._nltk is None:
            self._nltk = _load_nltk()
        if self._spacy is None:
            self._spacy = _load_spacy()

    # ─────────────────────────────────────────────────────────────────────────

    def process(self, text: str) -> dict:
        self._ensure_loaded()
        nltk  = self._nltk
        nlp   = self._spacy

        # ── NLTK pipeline ────────────────────────────────────────────────────
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.corpus   import stopwords
        from nltk.stem     import WordNetLemmatizer
        from nltk          import pos_tag

        sentences  = sent_tokenize(text)
        tokens     = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        filtered_tokens = [t for t in tokens if t.lower() not in stop_words
                           and t.isalpha()]
        lemmas          = [lemmatizer.lemmatize(t.lower()) for t in filtered_tokens]
        pos_tags        = pos_tag(tokens, lang='eng')

        # ── spaCy pipeline ───────────────────────────────────────────────────
        doc = nlp(text)

        entities = [
            {"text": ent.text, "label": ent.label_, "description": ent.label_}
            for ent in doc.ents
        ]
        dep_parse = [
            {"token": token.text, "dep": token.dep_, "head": token.head.text}
            for token in doc
            if not token.is_space
        ][:30]   # truncate for display

        return {
            "sentence_count":   len(sentences),
            "token_count":      len(tokens),
            "tokens":           tokens[:50],
            "filtered_tokens":  filtered_tokens[:40],
            "lemmas":           lemmas[:40],
            "pos_tags":         [{"word": w, "tag": t} for w, t in pos_tags[:30]],
            "named_entities":   entities,
            "dependency_parse": dep_parse,
        }
