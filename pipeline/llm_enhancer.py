"""
pipeline/llm_enhancer.py
LLM-based contextual refinement of poetry translations.
Supports OpenAI (gpt-4o / gpt-3.5-turbo) and Anthropic Claude.
"""

import logging

logger = logging.getLogger(__name__)

ENHANCE_PROMPT = """\
You are an expert literary translator and poet.

Below is an original poem and its raw machine translation.
Your task is to rewrite the translation so it:
- Preserves the emotional tone and mood of the original
- Maintains poetic rhythm and flow (adapt line breaks where helpful)
- Retains metaphorical and symbolic meaning
- Respects cultural nuance (adapt idioms naturally to the target culture)
- Remains faithful to the core message
{rag_section}

Return ONLY the improved translated poem — no explanation, no preamble.

──────────────────────────────────────────
ORIGINAL POEM:
{original}

──────────────────────────────────────────
RAW MACHINE TRANSLATION (target language: {tgt_lang}):
{raw_translation}

──────────────────────────────────────────
IMPROVED TRANSLATION:
"""


class LLMEnhancer:
    """Refines a machine-translated poem using an LLM."""

    def enhance(self, original: str, raw_translation: str,
                rag_context: str, api_key: str, tgt_lang: str) -> str:

        rag_section = ""
        if rag_context:
            rag_section = (
                f"\nFor stylistic reference, here are similar poems "
                f"retrieved from our poetry database:\n{rag_context}\n"
            )

        prompt = ENHANCE_PROMPT.format(
            original=original,
            raw_translation=raw_translation,
            tgt_lang=tgt_lang,
            rag_section=rag_section,
        )

        # Try OpenAI first
        enhanced = self._call_openai(prompt, api_key)
        if enhanced:
            return enhanced

        # Try Anthropic Claude
        enhanced = self._call_anthropic(prompt, api_key)
        if enhanced:
            return enhanced

        logger.warning("LLM enhancement failed — returning raw translation.")
        return raw_translation

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _call_openai(prompt: str, api_key: str) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.debug(f"OpenAI call failed: {e}")
            return ""

    @staticmethod
    def _call_anthropic(prompt: str, api_key: str) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.debug(f"Anthropic call failed: {e}")
            return ""
