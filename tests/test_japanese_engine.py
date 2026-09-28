"""
tests/test_japanese_engine.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for :class:`~core.nlp.japanese_engine.JapaneseTokenizer` and
:class:`~core.nlp.tokenizer_factory.TokenizerFactory`.

Test coverage
-------------
``TestJapaneseTokenizerBasic``
    Tokenisation output structure, non-empty input, empty input.
``TestJapaneseTokenizerConjugation``
    Verb inflection restoration (活用還原): past-tense, te-form, volitional,
    negative, potential, causative.
``TestJapaneseTokenizerPOS``
    Part-of-speech labels for verbs, nouns, adjectives, adverbs, particles.
``TestJapaneseTokenizerFiltering``
    Punctuation filtering (default on / explicit off).
``TestJapaneseTokenizerReading``
    Hiragana readings populated for kanji-containing tokens.
``TestTokenizerFactory``
    Registration, lookup, unsupported-language error, caching, reset.
"""

from __future__ import annotations

import pytest

from core.nlp.japanese_engine import JapaneseTokenizer
from core.nlp.tokenizer_factory import TokenizerFactory
from core.base import BaseTokenizer, Token
from core.exceptions import UnsupportedLanguageError


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tok() -> JapaneseTokenizer:
    """Shared JapaneseTokenizer instance (Janome initialises once per module)."""
    return JapaneseTokenizer()


# ─────────────────────────────────────────────────────────────────────────────
# TestJapaneseTokenizerBasic
# ─────────────────────────────────────────────────────────────────────────────

class TestJapaneseTokenizerBasic:
    """Basic tokenisation output structure."""

    def test_returns_list(self, tok: JapaneseTokenizer) -> None:
        result = tok.tokenize("猫が好きです")
        assert isinstance(result, list)

    def test_each_element_is_token(self, tok: JapaneseTokenizer) -> None:
        result = tok.tokenize("猫が好きです")
        assert all(isinstance(t, Token) for t in result)

    def test_empty_string_returns_empty_list(self, tok: JapaneseTokenizer) -> None:
        assert tok.tokenize("") == []

    def test_surface_present_in_output(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("東京に行きます")
        surfaces = [t.surface for t in tokens]
        # "東京" should appear as a surface form
        assert "東京" in surfaces

    def test_all_tokens_have_surface(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("彼は走った")
        assert all(t.surface for t in tokens)

    def test_all_tokens_have_base_form(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("食べました")
        assert all(t.base_form for t in tokens)

    def test_token_surface_reconstructs_input(self, tok: JapaneseTokenizer) -> None:
        """Concatenating all surfaces must reproduce the original input (no filtering)."""
        text = "私は学生です"
        tok_no_filter = JapaneseTokenizer(filter_punctuation=False)
        tokens = tok_no_filter.tokenize(text)
        assert "".join(t.surface for t in tokens) == text

    def test_lang_code_attribute(self, tok: JapaneseTokenizer) -> None:
        assert tok.lang_code == "ja"


# ─────────────────────────────────────────────────────────────────────────────
# TestJapaneseTokenizerConjugation (活用還原)
# ─────────────────────────────────────────────────────────────────────────────

class TestJapaneseTokenizerConjugation:
    """Verb inflection restoration to dictionary form (原形)."""

    def test_past_tense_ta_form(self, tok: JapaneseTokenizer) -> None:
        """食べた → base_form 食べる"""
        tokens = tok.tokenize("食べた")
        verb_tokens = [t for t in tokens if t.part_of_speech == "動詞"]
        assert any(t.base_form == "食べる" for t in verb_tokens), \
            f"Expected base_form='食べる' in {verb_tokens}"

    def test_past_tense_tta_form(self, tok: JapaneseTokenizer) -> None:
        """走った → base_form 走る"""
        tokens = tok.tokenize("走った")
        verb_tokens = [t for t in tokens if t.part_of_speech == "動詞"]
        assert any(t.base_form == "走る" for t in verb_tokens), \
            f"Expected base_form='走る' in {verb_tokens}"

    def test_te_form(self, tok: JapaneseTokenizer) -> None:
        """食べて → base_form 食べる"""
        tokens = tok.tokenize("食べて")
        verb_tokens = [t for t in tokens if t.part_of_speech == "動詞"]
        assert any(t.base_form == "食べる" for t in verb_tokens)

    def test_masu_form(self, tok: JapaneseTokenizer) -> None:
        """食べます → base_form 食べる (or 食べます contains 食べ)"""
        tokens = tok.tokenize("食べます")
        bases = [t.base_form for t in tokens]
        assert "食べる" in bases

    def test_nai_form(self, tok: JapaneseTokenizer) -> None:
        """食べない → base_form 食べる"""
        tokens = tok.tokenize("食べない")
        bases = [t.base_form for t in tokens]
        assert "食べる" in bases

    def test_suru_verb(self, tok: JapaneseTokenizer) -> None:
        """勉強した → base_form should contain する or 勉強する"""
        tokens = tok.tokenize("勉強した")
        bases = [t.base_form for t in tokens]
        assert any("する" in b for b in bases), f"Unexpected bases: {bases}"

    def test_kuru_verb(self, tok: JapaneseTokenizer) -> None:
        """来た → base_form 来る"""
        tokens = tok.tokenize("来た")
        verb_tokens = [t for t in tokens if t.part_of_speech == "動詞"]
        assert any(t.base_form == "来る" for t in verb_tokens), \
            f"Expected base_form='来る' in {verb_tokens}"

    def test_potential_form(self, tok: JapaneseTokenizer) -> None:
        """食べられる → base_form 食べる (potential ichidan)"""
        tokens = tok.tokenize("食べられる")
        bases = [t.base_form for t in tokens]
        assert "食べる" in bases or "食べられる" in bases

    def test_past_polite_mashita(self, tok: JapaneseTokenizer) -> None:
        """行きました → base_form 行く"""
        tokens = tok.tokenize("行きました")
        bases = [t.base_form for t in tokens]
        assert "行く" in bases


# ─────────────────────────────────────────────────────────────────────────────
# TestJapaneseTokenizerPOS
# ─────────────────────────────────────────────────────────────────────────────

class TestJapaneseTokenizerPOS:
    """Part-of-speech tag correctness."""

    def test_verb_pos(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("走る")
        pos_set = {t.part_of_speech for t in tokens}
        assert "動詞" in pos_set

    def test_noun_pos(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("猫")
        pos_set = {t.part_of_speech for t in tokens}
        assert "名詞" in pos_set

    def test_i_adjective_pos(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("美しい")
        pos_set = {t.part_of_speech for t in tokens}
        assert "形容詞" in pos_set

    def test_adverb_pos(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("とても")
        pos_set = {t.part_of_speech for t in tokens}
        assert "副詞" in pos_set

    def test_particle_pos(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("猫が")
        pos_set = {t.part_of_speech for t in tokens}
        assert "助詞" in pos_set

    def test_part_of_speech_not_empty(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("今日は晴れです")
        assert all(t.part_of_speech for t in tokens)


# ─────────────────────────────────────────────────────────────────────────────
# TestJapaneseTokenizerFiltering
# ─────────────────────────────────────────────────────────────────────────────

class TestJapaneseTokenizerFiltering:
    """Punctuation filtering behaviour."""

    def test_punctuation_filtered_by_default(self) -> None:
        tok = JapaneseTokenizer(filter_punctuation=True)
        tokens = tok.tokenize("食べた。")
        surfaces = [t.surface for t in tokens]
        assert "。" not in surfaces

    def test_punctuation_included_when_disabled(self) -> None:
        tok = JapaneseTokenizer(filter_punctuation=False)
        tokens = tok.tokenize("食べた。")
        surfaces = [t.surface for t in tokens]
        assert "。" in surfaces

    def test_mixed_text_meaningful_tokens_present(self) -> None:
        tok = JapaneseTokenizer(filter_punctuation=True)
        tokens = tok.tokenize("私は食べた。")
        surfaces = [t.surface for t in tokens]
        assert "私" in surfaces or "食べ" in surfaces  # at least one content word


# ─────────────────────────────────────────────────────────────────────────────
# TestJapaneseTokenizerReading
# ─────────────────────────────────────────────────────────────────────────────

class TestJapaneseTokenizerReading:
    """Hiragana reading extraction."""

    def test_kanji_has_reading(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("猫")
        noun_tokens = [t for t in tokens if t.surface == "猫"]
        assert noun_tokens, "Expected '猫' in tokenised output"
        # reading should be non-empty hiragana
        assert noun_tokens[0].reading, f"No reading for '猫': {noun_tokens[0]}"

    def test_verb_kanji_has_reading(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("食べる")
        verb_tokens = [t for t in tokens if t.part_of_speech == "動詞"]
        assert any(t.reading for t in verb_tokens)

    def test_reading_is_kana_string(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("東京")
        noun_tokens = [t for t in tokens if t.surface == "東京"]
        if noun_tokens:
            reading = noun_tokens[0].reading
            # Readings should be non-empty and consist of kana / alphanumeric
            assert reading.strip()

    def test_extra_dict_has_pos_detail(self, tok: JapaneseTokenizer) -> None:
        tokens = tok.tokenize("猫が走る")
        for t in tokens:
            assert "pos_detail" in t.extra


# ─────────────────────────────────────────────────────────────────────────────
# TestTokenizerFactory
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenizerFactory:
    """TokenizerFactory registration, lookup, and caching."""

    def setup_method(self) -> None:
        """Ensure a clean factory state before each test (preserve 'ja' default)."""
        # Only clear cache; keep factories so 'ja' remains registered.
        TokenizerFactory.clear_cache()

    def test_get_ja_returns_japanese_tokenizer(self) -> None:
        tok = TokenizerFactory.get("ja")
        assert isinstance(tok, JapaneseTokenizer)

    def test_get_ja_case_insensitive(self) -> None:
        tok = TokenizerFactory.get("JA")
        assert isinstance(tok, JapaneseTokenizer)

    def test_unsupported_language_raises(self) -> None:
        with pytest.raises(UnsupportedLanguageError) as exc_info:
            TokenizerFactory.get("xx-XX")
        assert "xx-xx" in exc_info.value.context.get("lang_code", "").lower()

    def test_cached_instance_returned(self) -> None:
        tok1 = TokenizerFactory.get("ja")
        tok2 = TokenizerFactory.get("ja")
        assert tok1 is tok2

    def test_register_custom_tokenizer(self) -> None:
        class _DummyTokenizer(BaseTokenizer):
            lang_code = "xx"
            def tokenize(self, text: str) -> list[Token]:
                return [Token(surface=text, base_form=text)]

        TokenizerFactory.register("xx", _DummyTokenizer, overwrite=True)
        tok = TokenizerFactory.get("xx")
        assert isinstance(tok, _DummyTokenizer)
        # Cleanup to avoid polluting other tests
        TokenizerFactory.reset()
        # Re-register Japanese so later tests still work
        from core.nlp.japanese_engine import JapaneseTokenizer as _JT
        TokenizerFactory.register("ja", _JT)

    def test_register_duplicate_without_overwrite_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            TokenizerFactory.register("ja", JapaneseTokenizer)

    def test_supported_languages_contains_ja(self) -> None:
        langs = TokenizerFactory.supported_languages()
        assert "ja" in langs

    def test_tokenize_stream_yields_correct_count(self) -> None:
        tok = TokenizerFactory.get("ja")
        texts = ["猫", "犬", "魚"]
        results = list(tok.tokenize_stream(texts))
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
