"""
core/nlp/japanese_engine.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Japanese morphological analysis engine backed by Janome.

Overview
--------
Janome is a pure-Python Japanese morphological analyser that ships with a
bundled MeCab-compatible dictionary (IPAdic).  This module wraps Janome's
:class:`janome.tokenizer.Tokenizer` to:

1. Tokenise arbitrary Japanese text into morphemes.
2. Extract ``surface``, ``base_form``, ``reading`` and a **normalised**
   ``part_of_speech`` label for each morpheme.
3. Filter out punctuation-only tokens by default.

Part-of-speech normalisation
-----------------------------
Janome returns a comma-separated feature string whose first field is the
coarse POS tag in Japanese (e.g. ``"動詞"``, ``"名詞"``, ``"助詞"`` …).
The engine keeps the coarse tag as-is for maximal transparency, mapping
a fixed set of known tags to canonical English equivalents is handled by
the downstream ``VocabularyEntry`` model if needed.

Usage
-----
>>> from core.nlp.japanese_engine import JapaneseTokenizer
>>> tok = JapaneseTokenizer()
>>> tokens = tok.tokenize("食べたい")
>>> tokens[0].base_form
'食べる'
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.base import BaseTokenizer, Token
from core.exceptions import EngineInitialisationError, TokenizationError

if TYPE_CHECKING:  # pragma: no cover
    pass  # keep the import tree clean

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Coarse POS tags that we consider "meaningful" vocabulary candidates.
# Tokens whose coarse POS is NOT in this set are still returned but carry
# the original Japanese tag so callers can filter as needed.
_MEANINGFUL_POS = frozenset({
    "動詞",    # verb
    "名詞",    # noun
    "形容詞",  # i-adjective
    "形容動詞", # na-adjective
    "副詞",    # adverb
    "接続詞",  # conjunction
    "感動詞",  # interjection
})

# Surface forms / POS tags to skip when filter_punctuation=True
_SKIP_POS = frozenset({
    "記号",    # symbols / punctuation
    "BOS/EOS", # sentence boundaries injected by Janome
})


class JapaneseTokenizer(BaseTokenizer):
    """
    Japanese morphological analysis engine using Janome.

    Parameters
    ----------
    filter_punctuation:
        If ``True`` (default) punctuation tokens (POS ``"記号"``) are
        excluded from the output.
    wakati:
        If ``True`` uses wakati-goshi (surface-only) mode.  Defaults to
        ``False`` (full morphological analysis with feature extraction).

    Raises
    ------
    EngineInitialisationError
        If Janome cannot be imported or its internal dictionary fails to load.
    """

    lang_code = "ja"

    def __init__(
        self,
        *,
        filter_punctuation: bool = True,
        wakati: bool = False,
    ) -> None:
        self.filter_punctuation = filter_punctuation
        self._wakati = wakati

        try:
            from janome.tokenizer import Tokenizer as JanomeTokenizer  # type: ignore[import]
            self._tokenizer = JanomeTokenizer(wakati=wakati)
        except ImportError as exc:
            raise EngineInitialisationError(
                "Janome is not installed. Run: pip install janome",
                {"original_error": str(exc)},
            ) from exc
        except Exception as exc:  # pragma: no cover
            raise EngineInitialisationError(
                f"Failed to initialise Janome tokeniser: {exc}",
                {"original_error": str(exc)},
            ) from exc

    # ── BaseTokenizer.tokenize() ─────────────────────────────────────────────

    def tokenize(self, text: str) -> list[Token]:
        """
        Tokenise Japanese ``text`` into a list of :class:`~core.base.Token` objects.

        Each token contains:
        - ``surface``      – the exact surface form from the text
        - ``base_form``    – dictionary form (原形)
        - ``reading``      – hiragana reading (読み) if available
        - ``part_of_speech`` – coarse Janome POS label (e.g. ``"動詞"``)
        - ``extra``        – dict with ``"pos_detail"`` (fine-grained sub-POS),
                             ``"conjugation_type"``, and ``"conjugation_form"``

        Parameters
        ----------
        text:
            Arbitrary Japanese string (may include kanji, kana, mixed).

        Returns
        -------
        list[Token]

        Raises
        ------
        TokenizationError
            On unexpected Janome failure.
        """
        if not text:
            return []

        try:
            raw_tokens = list(self._tokenizer.tokenize(text))
        except Exception as exc:
            raise TokenizationError(
                f"Janome tokenisation failed: {exc}",
                {"text": text[:80], "original_error": str(exc)},
            ) from exc

        tokens: list[Token] = []
        for tok in raw_tokens:
            surface = tok.surface

            if self._wakati:
                # In wakati mode Janome returns surface-only strings, not objects
                tokens.append(Token(surface=surface, base_form=surface))
                continue

            # Janome Token exposes named attributes directly:
            #   tok.part_of_speech → comma-joined feature string (IPAdic)
            #   tok.reading        → katakana reading (direct attribute)
            #   tok.base_form      → dictionary form (direct attribute)
            try:
                part_info = tok.part_of_speech.split(",")
            except AttributeError:
                part_info = []

            def _get(idx: int, fallback: str = "") -> str:
                try:
                    v = part_info[idx].strip()
                    return "" if v == "*" else v
                except IndexError:
                    return fallback

            coarse_pos = _get(0)
            pos_detail = _get(1)
            conj_type = _get(2)
            conj_form = _get(3)

            # Use direct Janome token attributes for base_form and reading
            # (more reliable than parsing the feature string by index)
            base_form: str = getattr(tok, "base_form", "") or ""
            if not base_form or base_form == "*":
                base_form = surface

            reading_raw: str = getattr(tok, "reading", "") or ""
            # Janome returns katakana; preserve as-is (downstream can convert)
            reading = "" if reading_raw == "*" else reading_raw


            if self.filter_punctuation and coarse_pos in _SKIP_POS:
                continue

            tokens.append(Token(
                surface=surface,
                base_form=base_form if base_form else surface,
                reading=reading,
                part_of_speech=coarse_pos,
                extra={
                    "pos_detail": pos_detail,
                    "conjugation_type": conj_type,
                    "conjugation_form": conj_form,
                },
            ))

        return tokens
